"""Transport-independent ledger for bounded dispatch of absolute robot commands.

One scheduler owns this object. It does not run a controller, validate IK, infer
tracking or task success, or make delivery exactly-once. A dispatch is a one-time
handoff; reading its window again is only for reconciliation, never replay.
"""

from copy import deepcopy
import json
import math
from uuid import uuid4


class ExecutionError(ValueError):
    """A rejected operation; ``code`` is stable and state is unchanged."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _reject(code, message):
    raise ExecutionError(code, message)


def _identifier(value, name):
    if type(value) is not str or not value.strip() or len(value) > 256:
        _reject("invalid_argument", f"{name} must be a nonempty string of at most 256 characters")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        _reject("invalid_argument", f"{name} must be an integer >= {minimum}")


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _commands(commands):
    """Validate already-converted RoboICL absolute commands without altering them."""
    if type(commands) is not list or not commands:
        _reject("invalid_command", "commands must be a nonempty list")
    for index, command in enumerate(commands):
        if type(command) is not dict:
            _reject("invalid_command", f"commands[{index}] must be a mapping")
        expected = set()
        for side in ("left", "right"):
            pose = f"{side}_ee_pose"
            joints = f"{side}_arm_joint_state"
            grip = f"{side}_ee_joint_state"
            if (pose in command) == (joints in command):
                _reject("invalid_command", f"commands[{index}] requires one absolute {side} target")
            target = pose if pose in command else joints
            expected.update((target, grip))
            for key, size in ((target, 7 if target == pose else 6), (grip, 1)):
                values = command.get(key)
                if (type(values) is not list or len(values) != size
                        or not all(_finite_number(value) for value in values)):
                    _reject("invalid_command", f"commands[{index}].{key} requires {size} finite numbers")
            if not 0 <= command[grip][0] <= 1:
                _reject("invalid_command", f"commands[{index}].{grip} must be in [0, 1]")
            if target == pose and abs(math.hypot(*command[pose][3:]) - 1) > 0.05:
                _reject("invalid_command", f"commands[{index}].{pose} quaternion must be unit length")
        if set(command) != expected:
            _reject("invalid_command", f"commands[{index}] contains unsupported fields")
    return deepcopy(commands)


class ExecutionQueue:
    """A bounded, single-writer execution queue and its acknowledgement history.

    ``remaining_steps`` are accepted but never dispatched. ``in_flight_steps``
    were handed off but have no acknowledgement. Cancel/fault moves the former
    to ``discarded_steps`` and the latter to ``unknown_steps``. ``executed_steps``
    only counts caller-supplied acknowledgements, not measured pose attainment.
    """

    SCHEMA_VERSION = 1
    TERMINAL = frozenset(("succeeded", "cancelled", "faulted"))

    def __init__(self, episode_id, *, max_window_steps=32, max_queue_steps=4096):
        _identifier(episode_id, "episode_id")
        _integer(max_window_steps, "max_window_steps", 1)
        _integer(max_queue_steps, "max_queue_steps", 1)
        if max_window_steps > max_queue_steps:
            _reject("invalid_argument", "max_window_steps cannot exceed max_queue_steps")
        self.episode_id = episode_id
        self.max_window_steps = max_window_steps
        self.max_queue_steps = max_queue_steps
        self._seen_episodes = [episode_id]
        self._executions = {}
        self._requests = {}
        self._plans = {}
        self._active = None

    def _episode(self, episode_id):
        _identifier(episode_id, "episode_id")
        if episode_id != self.episode_id:
            _reject("stale_episode", "operation belongs to a different episode")

    def _execution(self, episode_id, execution_id):
        self._episode(episode_id)
        _identifier(execution_id, "execution_id")
        if execution_id not in self._executions:
            _reject("unknown_execution", "execution_id is not present in this episode")
        return self._executions[execution_id]

    @staticmethod
    def _plan(execution, plan_id, plan_version):
        _identifier(plan_id, "plan_id")
        _integer(plan_version, "plan_version", 1)
        if (plan_id, plan_version) != (execution["plan_id"], execution["plan_version"]):
            _reject("stale_plan", "plan identity or version does not match the execution")

    @staticmethod
    def _receipt(execution):
        window = execution["window"]
        in_flight = window["end"] - execution["executed"] if window else 0
        return {
            "schema_version": 1,
            "episode_id": execution["episode_id"],
            "execution_id": execution["execution_id"],
            "plan_id": execution["plan_id"],
            "plan_version": execution["plan_version"],
            "status": execution["status"],
            "accepted_steps": len(execution["commands"]),
            "dispatched_steps": execution["dispatched"],
            "executed_steps": execution["executed"],
            "in_flight_steps": in_flight,
            "remaining_steps": (len(execution["commands"]) - execution["executed"]
                                - in_flight - execution["discarded"] - execution["unknown"]),
            "discarded_steps": execution["discarded"],
            "unknown_steps": execution["unknown"],
            "confirmed_prefix": [0, execution["executed"]],
            "active_window_id": window["window_id"] if window else None,
            "reason": execution["reason"],
            "evidence_ids": sorted(execution["evidence"]),
            "completion_scope": "acknowledged_commands_only",
            "physical_state_verified": False,
            "task_success": None,
        }

    def submit(self, *, episode_id, request_id, plan_id, plan_version, commands,
               expected_tail=0):
        """Accept absolute commands, or append to the current matching plan.

        Appending uses a new request_id and the current accepted count as
        expected_tail. Same request and parameters return the original acceptance
        receipt, even after completion; query() returns current progress.
        """
        self._episode(episode_id)
        _identifier(request_id, "request_id")
        _identifier(plan_id, "plan_id")
        _integer(plan_version, "plan_version", 1)
        _integer(expected_tail, "expected_tail")
        commands = _commands(commands)
        payload = {"plan_id": plan_id, "plan_version": plan_version,
                   "commands": commands, "expected_tail": expected_tail}
        previous = self._requests.get(request_id)
        if previous is not None:
            if previous["payload"] != payload:
                _reject("request_conflict", "request_id was already used with different parameters")
            return deepcopy(previous["receipt"])
        current = self._executions.get(self._active)
        previous_plan = self._plans.get(plan_id)
        if current is not None:
            if current["status"] not in ("queued", "running", "paused"):
                _reject("execution_unresolved", "previous execution must be reconciled before accepting more commands")
            self._plan(current, plan_id, plan_version)
            if expected_tail != len(current["commands"]):
                _reject("tail_conflict", "expected_tail does not match the accepted queue tail")
            if len(current["commands"]) + len(commands) > self.max_queue_steps:
                _reject("queue_limit", "execution exceeds max_queue_steps")
            current["commands"].extend(commands)
        else:
            if expected_tail != 0:
                _reject("tail_conflict", "a new execution must have expected_tail=0")
            if previous_plan is not None and plan_version <= previous_plan:
                _reject("stale_plan", "a completed or invalidated plan version cannot be restarted")
            if len(commands) > self.max_queue_steps:
                _reject("queue_limit", "execution exceeds max_queue_steps")
            execution_id = uuid4().hex
            current = {
                "episode_id": episode_id, "execution_id": execution_id,
                "plan_id": plan_id, "plan_version": plan_version,
                "commands": commands, "status": "queued", "executed": 0,
                "dispatched": 0, "discarded": 0, "unknown": 0,
                "window": None, "windows": {}, "evidence": {}, "reason": None,
            }
            self._executions[execution_id] = current
            self._active = execution_id
            self._plans[plan_id] = plan_version
        receipt = self._receipt(current)
        self._requests[request_id] = {"payload": deepcopy(payload), "receipt": deepcopy(receipt)}
        return receipt

    def query(self, *, episode_id, execution_id):
        return self._receipt(self._execution(episode_id, execution_id))

    def dispatch(self, *, episode_id, execution_id, plan_id, plan_version):
        """Hand off one new window; never implicitly return an in-flight one."""
        execution = self._execution(episode_id, execution_id)
        self._plan(execution, plan_id, plan_version)
        if execution["status"] not in ("queued", "running"):
            _reject("not_dispatchable", "execution is not runnable")
        if execution["window"] is not None:
            _reject("window_in_flight", "reconcile the existing window before dispatching another")
        start = execution["dispatched"]
        end = min(start + self.max_window_steps, len(execution["commands"]))
        if start == end:
            _reject("queue_empty", "there are no undispatched commands")
        window = {"window_id": uuid4().hex, "start": start, "end": end}
        execution["window"] = window
        execution["windows"][window["window_id"]] = deepcopy(window)
        execution["dispatched"] = end
        execution["status"] = "running"
        return self.read_window(episode_id=episode_id, execution_id=execution_id,
                                window_id=window["window_id"])

    def read_window(self, *, episode_id, execution_id, window_id):
        """Read a handoff for reconciliation. This does not authorize replay."""
        execution = self._execution(episode_id, execution_id)
        _identifier(window_id, "window_id")
        if window_id not in execution["windows"]:
            _reject("unknown_window", "window_id does not belong to this execution")
        window = execution["windows"][window_id]
        return {"window_id": window_id, "start_index": window["start"],
                "end_index": window["end"],
                "commands": deepcopy(execution["commands"][window["start"]:window["end"]]),
                "receipt": self._receipt(execution)}

    def acknowledge(self, *, episode_id, execution_id, window_id, executed_steps,
                    evidence_id):
        """Confirm a cumulative prefix within this window using execution evidence.

        The adapter must call this only after actual execution receipts, never
        because commands were returned, time passed, or the gripper was commanded.
        """
        execution = self._execution(episode_id, execution_id)
        _identifier(window_id, "window_id")
        _identifier(evidence_id, "evidence_id")
        _integer(executed_steps, "executed_steps", 1)
        if execution["status"] not in ("queued", "running", "paused", "succeeded"):
            _reject("not_acknowledgeable", "execution no longer accepts execution results")
        evidence = {"window_id": window_id, "executed_steps": executed_steps}
        previous = execution["evidence"].get(evidence_id)
        if previous is not None:
            if previous != evidence:
                _reject("evidence_conflict", "evidence_id already confirms a different result")
            return self._receipt(execution)
        window = execution["window"]
        if window is None or window["window_id"] != window_id:
            _reject("stale_window", "only the current in-flight window accepts new results")
        confirmed = window["start"] + executed_steps
        if confirmed > window["end"] or confirmed <= execution["executed"]:
            _reject("invalid_prefix", "acknowledgement must extend the executed prefix within the window")
        execution["executed"] = confirmed
        execution["evidence"][evidence_id] = evidence
        if confirmed == window["end"]:
            execution["window"] = None
            if confirmed == len(execution["commands"]):
                execution["status"] = "succeeded"
                self._active = None
            elif execution["status"] != "paused":
                execution["status"] = "queued"
        return self._receipt(execution)

    def pause(self, *, episode_id, execution_id, reason):
        """Prevent subsequent dispatch; this is not a physical stop command."""
        execution = self._execution(episode_id, execution_id)
        _identifier(reason, "reason")
        if execution["status"] not in ("queued", "running", "paused"):
            _reject("not_pauseable", "execution cannot be paused")
        execution["status"] = "paused"
        execution["reason"] = reason
        return self._receipt(execution)

    def resume(self, *, episode_id, execution_id, plan_id, plan_version, evidence_id):
        """Resume only after the adapter has revalidated state and reconciled delivery."""
        execution = self._execution(episode_id, execution_id)
        self._plan(execution, plan_id, plan_version)
        _identifier(evidence_id, "evidence_id")
        if execution["status"] != "paused":
            _reject("not_resumable", "only paused executions can resume")
        if execution["window"] is not None:
            _reject("window_in_flight", "unconfirmed dispatched commands must be reconciled or cancelled")
        if evidence_id in execution["evidence"]:
            _reject("evidence_conflict", "resume requires a fresh state validation reference")
        execution["evidence"][evidence_id] = {"operation": "resume"}
        execution["status"] = "queued"
        execution["reason"] = None
        return self._receipt(execution)

    def _terminate(self, episode_id, execution_id, reason, status):
        execution = self._execution(episode_id, execution_id)
        _identifier(reason, "reason")
        if execution["status"] in self.TERMINAL:
            if execution["status"] == status and execution["reason"] == reason:
                return self._receipt(execution)
            _reject("already_terminal", "terminal execution cannot change state")
        execution["discarded"] = len(execution["commands"]) - execution["dispatched"]
        execution["unknown"] = execution["dispatched"] - execution["executed"]
        execution["window"] = None
        execution["status"] = status
        execution["reason"] = reason
        self._active = execution_id if execution["unknown"] else None
        return self._receipt(execution)

    def cancel(self, *, episode_id, execution_id, reason):
        return self._terminate(episode_id, execution_id, reason, "cancelled")

    def fault(self, *, episode_id, execution_id, reason):
        return self._terminate(episode_id, execution_id, reason, "faulted")

    def confirm_discarded(self, *, episode_id, execution_id, evidence_id):
        """Record external proof that the unconfirmed dispatched suffix did not run.

        Only call after the adapter has reconciled actual execution and confirmed
        the suffix cannot run later (for example, a synchronous rollout returned
        and discarded it). This method neither stops a robot nor establishes that
        fact itself. Uncertain outcomes must stay unknown and keep the queue locked.
        """
        execution = self._execution(episode_id, execution_id)
        _identifier(evidence_id, "evidence_id")
        if execution["status"] not in ("cancelled", "faulted"):
            _reject("not_reconcilable", "cancel or fault the execution before confirming discarded commands")
        evidence = {"operation": "confirm_discarded"}
        previous = execution["evidence"].get(evidence_id)
        if previous is not None:
            if previous != evidence:
                _reject("evidence_conflict", "evidence_id already records another operation")
            return self._receipt(execution)
        if execution["unknown"] == 0:
            _reject("nothing_unknown", "execution has no unresolved dispatched commands")
        execution["discarded"] += execution["unknown"]
        execution["unknown"] = 0
        execution["evidence"][evidence_id] = evidence
        if self._active == execution_id:
            self._active = None
        return self._receipt(execution)

    def reset(self, *, episode_id):
        """Start a distinct episode; the adapter separately handles robot stopping."""
        _identifier(episode_id, "episode_id")
        if episode_id in self._seen_episodes:
            _reject("episode_reused", "episode IDs cannot be reused")
        self.episode_id = episode_id
        self._seen_episodes.append(episode_id)
        self._executions = {}
        self._requests = {}
        self._plans = {}
        self._active = None

    def snapshot(self):
        """Return JSON-serializable state; callers own durable/atomic storage."""
        return deepcopy({
            "schema_version": self.SCHEMA_VERSION, "episode_id": self.episode_id,
            "max_window_steps": self.max_window_steps, "max_queue_steps": self.max_queue_steps,
            "seen_episodes": self._seen_episodes, "executions": self._executions,
            "requests": self._requests, "plans": self._plans, "active": self._active,
        })

    @classmethod
    def from_snapshot(cls, snapshot):
        """Restore validated state, latching unfinished executions as unknown.

        Restart never establishes whether a previously handed-off command ran.
        Unknown executions can be queried or cancelled, but cannot resume.
        """
        restored = cls._validated_snapshot(snapshot)
        for execution in restored._executions.values():
            if execution["status"] not in cls.TERMINAL:
                execution["unknown"] = execution["dispatched"] - execution["executed"]
                execution["window"] = None
                execution["status"] = "unknown"
                execution["reason"] = "restored_without_controller_reconciliation"
        return restored

    @classmethod
    def _validated_snapshot(cls, snapshot):
        # Validate the recorded requests, window history and counters in a
        # private candidate before exposing anything to a caller.
        if type(snapshot) is not dict or set(snapshot) != {
                "schema_version", "episode_id", "max_window_steps", "max_queue_steps",
                "seen_episodes", "executions", "requests", "plans", "active"}:
            _reject("invalid_snapshot", "unexpected snapshot fields")
        if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != cls.SCHEMA_VERSION:
            _reject("invalid_snapshot", "unsupported snapshot schema")
        try:
            json.dumps(snapshot, allow_nan=False)
        except (TypeError, ValueError):
            _reject("invalid_snapshot", "snapshot is not finite JSON")
        value = deepcopy(snapshot)
        result = cls(value["episode_id"], max_window_steps=value["max_window_steps"],
                     max_queue_steps=value["max_queue_steps"])
        seen = value["seen_episodes"]
        if (type(seen) is not list or not seen or any(type(item) is not str for item in seen)
                or len(set(seen)) != len(seen) or seen[-1] != result.episode_id):
            _reject("invalid_snapshot", "invalid episode history")
        for item in seen:
            _identifier(item, "seen_episode")
        if any(type(value[key]) is not dict for key in ("executions", "requests", "plans")):
            _reject("invalid_snapshot", "invalid ledger mappings")
        executions = value["executions"]
        active = []
        for execution_id, execution in executions.items():
            _identifier(execution_id, "execution_id")
            required = {"episode_id", "execution_id", "plan_id", "plan_version", "commands",
                        "status", "executed", "dispatched", "discarded", "unknown", "window",
                        "windows", "evidence", "reason"}
            if type(execution) is not dict or set(execution) != required:
                _reject("invalid_snapshot", "invalid execution record")
            if execution["episode_id"] != result.episode_id or execution["execution_id"] != execution_id:
                _reject("invalid_snapshot", "execution identity mismatch")
            _identifier(execution["plan_id"], "plan_id")
            _integer(execution["plan_version"], "plan_version", 1)
            _commands(execution["commands"])
            total = len(execution["commands"])
            for key in ("executed", "dispatched", "discarded", "unknown"):
                _integer(execution[key], key)
            if not (execution["executed"] <= execution["dispatched"] <= total <= result.max_queue_steps):
                _reject("invalid_snapshot", "inconsistent execution counters")
            status = execution["status"]
            if type(status) is not str or status not in cls.TERMINAL | {"queued", "running", "paused", "unknown"}:
                _reject("invalid_snapshot", "invalid execution state")
            if execution["reason"] is not None:
                _identifier(execution["reason"], "reason")
            windows = execution["windows"]
            if type(windows) is not dict or type(execution["evidence"]) is not dict:
                _reject("invalid_snapshot", "invalid window or evidence mapping")
            for window_id, window in windows.items():
                _identifier(window_id, "window_id")
                if type(window) is not dict or set(window) != {"window_id", "start", "end"}:
                    _reject("invalid_snapshot", "invalid window record")
                _integer(window["start"], "window.start")
                _integer(window["end"], "window.end", 1)
                if (window["window_id"] != window_id
                        or not window["start"] < window["end"] <= window["start"] + result.max_window_steps):
                    _reject("invalid_snapshot", "invalid dispatch interval")
            frontier = 0
            for window in sorted(windows.values(), key=lambda item: item["start"]):
                if window["start"] != frontier:
                    _reject("invalid_snapshot", "noncontiguous dispatch history")
                frontier = window["end"]
            if frontier != execution["dispatched"]:
                _reject("invalid_snapshot", "dispatch history differs from counters")
            window = execution["window"]
            if window is not None:
                if type(window) is not dict:
                    _reject("invalid_snapshot", "invalid in-flight window")
                _identifier(window.get("window_id"), "window_id")
                if (window != windows.get(window["window_id"])
                        or window["end"] != execution["dispatched"]
                        or not window["start"] <= execution["executed"] < window["end"]):
                    _reject("invalid_snapshot", "invalid in-flight window")
            if status in ("queued", "running", "paused"):
                if (execution["discarded"] or execution["unknown"] or execution["executed"] == total
                        or (window is None and execution["executed"] != execution["dispatched"])
                        or (status == "queued" and window is not None)
                        or (status == "running" and window is None)):
                    _reject("invalid_snapshot", "state contradicts execution progress")
            else:
                discarded_evidence = any(item == {"operation": "confirm_discarded"}
                                         for item in execution["evidence"].values())
                expected_discarded = ((total - execution["executed"] if discarded_evidence
                                       else total - execution["dispatched"])
                                      if status in ("cancelled", "faulted") else 0)
                expected_unknown = (execution["dispatched"] - execution["executed"]
                                    if status != "succeeded" and not discarded_evidence else 0)
                if (window is not None or execution["discarded"] != expected_discarded
                        or execution["unknown"] != expected_unknown
                        or (status == "succeeded" and execution["executed"] != total)
                        or (discarded_evidence and status not in ("cancelled", "faulted"))):
                    _reject("invalid_snapshot", "terminal or unknown state contradicts progress")
            if status not in cls.TERMINAL or execution["unknown"]:
                active.append(execution_id)
            confirmed = {window_id: 0 for window_id in windows}
            for evidence_id, evidence in execution["evidence"].items():
                _identifier(evidence_id, "evidence_id")
                if evidence in ({"operation": "resume"}, {"operation": "confirm_discarded"}):
                    if evidence["operation"] == "confirm_discarded" and status not in ("cancelled", "faulted"):
                        _reject("invalid_snapshot", "discard confirmation requires a terminated execution")
                    continue
                if type(evidence) is not dict or set(evidence) != {"window_id", "executed_steps"}:
                    _reject("invalid_snapshot", "invalid evidence record")
                _integer(evidence["executed_steps"], "executed_steps", 1)
                _identifier(evidence["window_id"], "window_id")
                referenced = windows.get(evidence["window_id"])
                if (referenced is None or evidence["executed_steps"] > referenced["end"] - referenced["start"]
                        or referenced["start"] + evidence["executed_steps"] > execution["executed"]):
                    _reject("invalid_snapshot", "evidence extends beyond confirmed commands")
                confirmed[evidence["window_id"]] = max(confirmed[evidence["window_id"]], evidence["executed_steps"])
            if any(confirmed[key] != min(window["end"], execution["executed"]) - window["start"]
                   for key, window in windows.items() if window["start"] < execution["executed"]):
                _reject("invalid_snapshot", "confirmed progress lacks execution evidence")
        if len(active) > 1 or value["active"] != (active[0] if active else None):
            _reject("invalid_snapshot", "active execution mismatch")
        expected_plans = {}
        for execution in executions.values():
            key = execution["plan_id"]
            expected_plans[key] = max(expected_plans.get(key, 0), execution["plan_version"])
        if value["plans"] != expected_plans or any(type(v) is not int for v in value["plans"].values()):
            _reject("invalid_snapshot", "plan versions differ from execution history")
        accepted_ranges = {execution_id: [] for execution_id in executions}
        for request_id, request in value["requests"].items():
            _identifier(request_id, "request_id")
            if type(request) is not dict or set(request) != {"payload", "receipt"}:
                _reject("invalid_snapshot", "invalid idempotency record")
            payload, receipt = request["payload"], request["receipt"]
            if type(payload) is not dict or set(payload) != {"plan_id", "plan_version", "commands", "expected_tail"}:
                _reject("invalid_snapshot", "invalid accepted request payload")
            _identifier(payload["plan_id"], "plan_id")
            _integer(payload["plan_version"], "plan_version", 1)
            _integer(payload["expected_tail"], "expected_tail")
            _commands(payload["commands"])
            if type(receipt) is not dict:
                _reject("invalid_snapshot", "invalid acceptance receipt")
            _identifier(receipt.get("execution_id"), "execution_id")
            if receipt["execution_id"] not in executions:
                _reject("invalid_snapshot", "request references an absent execution")
            execution = executions[receipt["execution_id"]]
            start = payload["expected_tail"]
            if (payload["plan_id"] != execution["plan_id"] or payload["plan_version"] != execution["plan_version"]
                    or execution["commands"][start:start + len(payload["commands"])] != payload["commands"]):
                _reject("invalid_snapshot", "request receipt does not match accepted commands")
            end = start + len(payload["commands"])
            accepted_ranges[receipt["execution_id"]].append((start, end))
            cls._validate_saved_receipt(receipt, execution, end)
        for execution_id, intervals in accepted_ranges.items():
            frontier = 0
            for start, end in sorted(intervals):
                if start != frontier:
                    _reject("invalid_snapshot", "acceptance history contains gaps or overlaps")
                frontier = end
            if frontier != len(executions[execution_id]["commands"]):
                _reject("invalid_snapshot", "commands are missing acceptance records")
        result._seen_episodes = seen
        result._executions = executions
        result._requests = value["requests"]
        result._plans = value["plans"]
        result._active = value["active"]
        return result

    @classmethod
    def _validate_saved_receipt(cls, receipt, execution, accepted):
        """Acceptance receipts are frozen historical views, not current state."""
        if set(receipt) != set(cls._receipt(execution)):
            _reject("invalid_snapshot", "unexpected receipt fields")
        for key in ("schema_version", "plan_version", "accepted_steps", "dispatched_steps",
                    "executed_steps", "in_flight_steps", "remaining_steps", "discarded_steps", "unknown_steps"):
            _integer(receipt[key], key)
        for key in ("episode_id", "execution_id", "plan_id", "plan_version"):
            if receipt[key] != execution[key]:
                _reject("invalid_snapshot", "acceptance receipt identity mismatch")
        if (receipt["schema_version"] != cls.SCHEMA_VERSION
                or receipt["accepted_steps"] != accepted
                or not 0 <= receipt["executed_steps"] <= receipt["dispatched_steps"] <= accepted
                or receipt["executed_steps"] >= accepted
                or receipt["in_flight_steps"] != receipt["dispatched_steps"] - receipt["executed_steps"]
                or receipt["remaining_steps"] != accepted - receipt["dispatched_steps"]
                or receipt["discarded_steps"] or receipt["unknown_steps"]):
            _reject("invalid_snapshot", "inconsistent acceptance receipt counters")
        status = receipt["status"]
        if type(status) is not str or status not in ("queued", "running", "paused"):
            _reject("invalid_snapshot", "invalid acceptance receipt state")
        window_id = receipt["active_window_id"]
        if receipt["in_flight_steps"]:
            _identifier(window_id, "window_id")
            window = execution["windows"].get(window_id)
            if (window is None or window["end"] != receipt["dispatched_steps"]
                    or window["start"] > receipt["executed_steps"] or status == "queued"):
                _reject("invalid_snapshot", "acceptance receipt has an inconsistent window")
        elif window_id is not None or status == "running":
            _reject("invalid_snapshot", "acceptance receipt has no matching in-flight commands")
        prefix = receipt["confirmed_prefix"]
        if (type(prefix) is not list or len(prefix) != 2 or any(type(item) is not int for item in prefix)
                or prefix != [0, receipt["executed_steps"]]
                or receipt["completion_scope"] != "acknowledged_commands_only"
                or receipt["physical_state_verified"] is not False or receipt["task_success"] is not None):
            _reject("invalid_snapshot", "invalid acceptance receipt evidence claims")
        if receipt["reason"] is not None:
            _identifier(receipt["reason"], "reason")
        ids = receipt["evidence_ids"]
        if (type(ids) is not list or any(type(item) is not str for item in ids)
                or len(set(ids)) != len(ids) or any(item not in execution["evidence"] for item in ids)):
            _reject("invalid_snapshot", "invalid acceptance receipt evidence references")
