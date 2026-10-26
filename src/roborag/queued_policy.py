"""Opt-in synchronous segment queue on the existing RoboICL observation loop.

One model act owns one complete segment and one final tool receipt. Dispatch
windows are an implementation limit, not a reason to infer or replay a suffix.
This is not the future Astra/Luna supervisor or an asynchronous safety loop.
"""

from copy import deepcopy
import json

from roboicl.policy.action_feedback import TrackingGuard
from roboicl.policy.astra_policy import plain, validate_actions
from roboicl.policy.bounded_policy import BudgetExceeded
from roboicl.policy.dialogue_policy import Model as DialogueModel

from .execution import ExecutionQueue
from .robot_memory import RetrievalModel
from .waypoints import expand_segment


QUEUE_INSTRUCTIONS = """
QUEUED SEGMENT CONTROL
Choose control_mode="transit" for a clearly free-space segment: 1..6 sparse
waypoint displacement rows, each at most 0.30m and 0.60rad per arm. Preserve both
grippers at their latest commanded openings. The executor expands all rows into
bounded motor steps and converts them to absolute targets once. It dispatches
at most 32 motor steps per local window and automatically continues the remaining
authorized segment from that same plan, without asking the model at each window.

Choose control_mode="precision" for approach, uncertain geometry, contact,
grasp/release or retention checks: 1..8 rows, each at most 0.01m and 0.15rad per
arm. Gripper changes must be explicit. Reobserve after this short segment.
Only submit a segment that can be executed until the next semantic checkpoint.
The local executor never selects a new target or task stage. Existing IK and
arm tracking guards can interrupt a segment; an unresolved large residual at a
window boundary also stops continuation. The entire remaining suffix then
becomes invalid. Replan from the latest observation, without compensating for
unexecuted motion. A receipt proves command execution, not grasp or task success.
Keep execution_note concise and include the current task stage and checkpoint.
Historical TRAIN action rows are recorded motor steps, not sparse waypoints;
their original time scale is preserved. Use the current control_mode contract
for every new LIVE act, even when a historical example lacks that field.
""".strip()


class _QueuedControlMixin:
    def __init__(self, cfg):
        super().__init__(cfg)
        c = self.dialogue_config
        if c.action_space != "ee_delta" or c.fixed_predict_horizon:
            raise ValueError("queued requires variable-horizon ee_delta actions")
        if c.execute_horizon > 32:
            raise ValueError("queued dispatch windows cannot exceed 32 steps")
        original = (f"Per step, translation norm <= {c.max_translation_m}m and "
                    f"rotation norm <= {c.max_rotation_rad}rad. ")
        horizon = (f"Predict 1..{c.predict_horizon} steps per act; "
                   f"execute at most {c.execute_horizon}. ")
        if original not in self.prompt or horizon not in self.prompt:
            raise ValueError("Could not locate upstream action contract")
        self.prompt = self.prompt.replace(original, "Bounds depend on control_mode below. ", 1)
        self.prompt = self.prompt.replace(horizon, "Submit one bounded segment per act. ", 1)
        self.prompt += "\n\n" + QUEUE_INSTRUCTIONS
        self.base_prompt = self.prompt

    def reset(self):
        super().reset()
        self.execution_queue = ExecutionQueue(
            self.episode_id, max_window_steps=min(self.dialogue_config.execute_horizon, 32),
            max_queue_steps=64)
        self._execution = None
        self._window = None
        self._prepared_commands = None
        self._converting_response = False
        self._response_mode = None
        self._segment_start = None
        self._confirmed_step = None
        self._segment_tracking = None

    def _tools(self):
        tools = super()._tools()
        tool = tools[0]
        tool["description"] = "Submit one transit/precision segment; receive its complete executed-prefix receipt."
        schema = tool["parameters"]
        schema["properties"]["control_mode"] = {"type": "string", "enum": ["transit", "precision"]}
        schema["properties"]["actions"].update(minItems=1, maxItems=8)
        schema["required"].append("control_mode")
        return tools

    def _request(self, payload):
        response = super()._request(payload)
        # The base class parses actions and execution_note. Capture the extra
        # schema field here; validate it in _convert_tensor so normal bounded
        # action-correction handling also covers a missing or invalid mode.
        self._converting_response = True
        self._response_mode = None
        calls = [v for v in response.get("output", []) if v.get("type") == "function_call"]
        if len(calls) == 1:
            try:
                args = json.loads(calls[0]["arguments"])
                if isinstance(args, dict):
                    self._response_mode = args.get("control_mode")
            except (ValueError, TypeError, KeyError):
                pass
        return response

    def _convert_tensor(self, rows, state):
        if not self._converting_response:
            # Upstream initialization may serialize frozen TRAIN actions.
            return super()._convert_tensor(rows, state)
        motor = expand_segment(rows, state, mode=self._response_mode, max_motor_steps=64)
        raw = [{"mode": "ee_delta", "left": row[:6], "left_gripper": row[6],
                "right": row[7:13], "right_gripper": row[13]} for row in motor]
        commands = validate_actions(raw, max_chunk=64, state=state)
        self._prepared_commands = plain(commands)
        return commands

    def _identity(self):
        return {"episode_id": self.episode_id, "execution_id": self._execution["execution_id"]}

    def _save_execution(self, event):
        receipt = self.execution_queue.query(**self._identity())
        self._log("executions.jsonl", {"event": event, "step": self.step,
                                      "call_id": self._execution["request_id"], **receipt})
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.episode_dir / "queue-state.tmp"
        temporary.write_text(json.dumps(self.execution_queue.snapshot(), allow_nan=False), encoding="utf-8")
        temporary.replace(self.episode_dir / "queue-state.json")

    def _dispatch(self):
        self._window = self.execution_queue.dispatch(
            **self._identity(), plan_id=self._execution["plan_id"],
            plan_version=self._execution["plan_version"])
        # Upstream rejection/receipt indexing covers the whole model segment,
        # whereas the simulator's action_index is local to this window.
        self.pending["returned"] = self._window["end_index"]
        self._save_execution("window_dispatched")
        return deepcopy(self._window["commands"])

    def _get_action_once(self, allow_correction=False):
        self._prepared_commands = None
        self._converting_response = False
        try:
            returned = super()._get_action_once(allow_correction=allow_correction)
        finally:
            self._converting_response = False
        if returned is None or self._prepared_commands is None or self.pending.get("local_safety_hold"):
            return returned
        call_id = self.pending["call_id"]
        receipt = self.execution_queue.submit(
            episode_id=self.episode_id, request_id=call_id, plan_id=call_id,
            plan_version=1, commands=self._prepared_commands)
        self._execution = {**receipt, "request_id": call_id}
        self._segment_start = self.step
        self._confirmed_step = self.step
        self._segment_tracking = TrackingGuard()
        self._save_execution("segment_accepted")
        return self._dispatch()

    def _terminate_segment(self, reason, *, fault=False):
        operation = self.execution_queue.fault if fault else self.execution_queue.cancel
        receipt = operation(**self._identity(), reason=reason)
        if receipt["unknown_steps"]:
            # Only called after the synchronous rollout returns control to
            # get_action/finalize, and after its executed prefix was reconciled.
            self.execution_queue.confirm_discarded(
                **self._identity(), evidence_id=f"rollout-returned:{self.step}:{reason}")
        self._save_execution("segment_interrupted")
        self._window = None

    def _reconcile(self, *, terminal=False):
        if self._execution is None:
            return
        receipt = self.execution_queue.query(**self._identity())
        if receipt["status"] in ExecutionQueue.TERMINAL:
            return
        records = [r for r in self.ledger if self._confirmed_step < r["resulting_step"] <= self.step]
        if [r["resulting_step"] for r in records] != list(range(self._confirmed_step + 1, self.step + 1)):
            raise ValueError("Missing execution observations; cannot continue the queue")
        interrupted = bool(self.pending.get("controller_rejection"))
        for record in records:
            index = record["resulting_step"] - self._segment_start - 1
            local_index = index - self._window["start_index"]
            if not 0 <= local_index < len(self._window["commands"]):
                raise ValueError("Observation exceeds dispatched queue window")
            if plain(record["executed_action"]) != self._window["commands"][local_index]:
                raise ValueError("Observed command differs from the dispatched queue target")
            self.execution_queue.acknowledge(
                **self._identity(), window_id=self._window["window_id"],
                executed_steps=local_index + 1,
                evidence_id=f"observation:{record['resulting_step']}")
            self._confirmed_step = record["resulting_step"]
            if self._segment_tracking.observe(record.get("tracking", {})):
                # Upstream emits no event if the guard fires on a window's
                # last action. Recheck across windows so that boundary cannot
                # accidentally authorize the next window.
                record.setdefault("controller_event", {
                    "type": "queued_segment_tracking_error", "source": "measured_arm_tracking",
                    "reason": "Consecutive large arm residuals; remaining segment invalidated."})
            interrupted = interrupted or bool(record.get("controller_event"))
        receipt = self.execution_queue.query(**self._identity())
        if (receipt["remaining_steps"] and not receipt["in_flight_steps"]
                and self._segment_tracking.count):
            # The simulator starts a fresh guard for each dispatch window.
            # Do not carry a pending large residual into that fresh guard:
            # otherwise the first bad sample in the next window could be
            # missed, and this policy only sees it after that window returns.
            if records:
                records[-1].setdefault("controller_event", {
                    "type": "queued_boundary_tracking_error", "source": "measured_arm_tracking",
                    "reason": "Large residual at dispatch boundary; reobserve before continuing."})
            interrupted = True
        if receipt["status"] == "succeeded":
            self._window = None
            self._save_execution("segment_acknowledged")
        elif interrupted:
            self._terminate_segment("controller_interruption", fault=True)
        elif terminal:
            self._terminate_segment("episode_finalized")
        elif receipt["in_flight_steps"]:
            raise ValueError("Dispatched window lacks complete receipts; duplicate dispatch prohibited")
        else:
            self._window = None
            self._save_execution("window_acknowledged")

    def get_action(self):
        if self._execution is not None:
            self._reconcile()
            receipt = self.execution_queue.query(**self._identity())
            if receipt["status"] in ("queued", "running"):
                if self._time_limit_reached():
                    self._terminate_segment("harness_timeout")
                    self._commit_feedback(terminal=True)
                    raise BudgetExceeded("harness_timeout: no remaining queued actions dispatched")
                return self._dispatch()
        return super().get_action()

    def notify_action_rejection(self, obs):
        value = deepcopy(obs)
        if self._execution is not None and self._window is not None:
            value["event"]["action_index"] += self._window["start_index"]
        return super().notify_action_rejection(value)

    def _commit_feedback(self, terminal=False):
        if self._execution is not None:
            self._reconcile(terminal=terminal)
            status = self.execution_queue.query(**self._identity())["status"]
            if status not in ExecutionQueue.TERMINAL:
                raise ValueError("Cannot close a model act while authorized actions remain queued")
        super()._commit_feedback(terminal=terminal)
        self._execution = None
        self._window = None


class QueuedDialogueModel(_QueuedControlMixin, DialogueModel):
    """A single-model, no-retrieval queued execution pilot."""


class QueuedRetrievalModel(_QueuedControlMixin, RetrievalModel):
    """Same execution pilot with optional existing retrieval conditions."""
