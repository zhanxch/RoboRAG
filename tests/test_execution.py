import copy
import json
import unittest

from roborag.execution import ExecutionError, ExecutionQueue


def commands(count):
    return [{"left_ee_pose": [index / 100, 0, 0, 1, 0, 0, 0],
             "left_ee_joint_state": [0.5],
             "right_arm_joint_state": [0, 0, 0, 0, 0, 0],
             "right_ee_joint_state": [1]} for index in range(count)]


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.queue = ExecutionQueue("episode", max_window_steps=3, max_queue_steps=20)
        self.submit_args = dict(episode_id="episode", request_id="request", plan_id="plan",
                                plan_version=1, commands=commands(8))

    def submit(self, **changes):
        return self.queue.submit(**(self.submit_args | changes))

    def identity(self, receipt):
        return {key: receipt[key] for key in ("episode_id", "execution_id")}

    def dispatch(self, receipt):
        return self.queue.dispatch(**self.identity(receipt), plan_id=receipt["plan_id"],
                                   plan_version=receipt["plan_version"])

    def ack(self, receipt, window, count, evidence="receipt-1"):
        return self.queue.acknowledge(**self.identity(receipt), window_id=window["window_id"],
                                      executed_steps=count, evidence_id=evidence)

    def unchanged_rejection(self, code, callback):
        before = self.queue.snapshot()
        with self.assertRaises(ExecutionError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(self.queue.snapshot(), before)

    def test_multiple_windows_preserve_every_absolute_command_without_replay(self):
        receipt = self.submit()
        dispatched = []
        for index, count in enumerate((3, 3, 2)):
            window = self.dispatch(receipt)
            dispatched.extend(window["commands"])
            self.assertEqual(window["start_index"], index * 3)
            self.assertEqual(window["receipt"]["executed_steps"], index * 3)
            self.assertEqual(window["receipt"]["in_flight_steps"], count)
            receipt = self.ack(receipt, window, count, f"evidence-{index}")
        self.assertEqual(dispatched, commands(8))
        self.assertEqual(receipt["status"], "succeeded")
        self.assertEqual(receipt["executed_steps"], 8)
        self.assertEqual(receipt["remaining_steps"], 0)
        self.assertIsNone(receipt["task_success"])
        self.assertFalse(receipt["physical_state_verified"])
        self.unchanged_rejection("not_dispatchable", lambda: self.dispatch(receipt))

    def test_request_dedup_returns_original_receipt_and_rejects_payload_change(self):
        original = self.submit()
        window = self.dispatch(original)
        self.ack(original, window, 1)
        self.assertEqual(self.submit(), original)
        self.unchanged_rejection("request_conflict", lambda: self.submit(commands=commands(2)))
        self.assertEqual(self.queue.query(**self.identity(original))["executed_steps"], 1)

    def test_more_than_32_steps_use_three_dispatch_windows_without_truncation(self):
        self.queue = ExecutionQueue("episode", max_window_steps=32, max_queue_steps=100)
        receipt = self.submit(commands=commands(70))
        collected = []
        for index, size in enumerate((32, 32, 6)):
            window = self.dispatch(receipt)
            self.assertEqual(len(window["commands"]), size)
            collected.extend(window["commands"])
            receipt = self.ack(receipt, window, size, f"executed-{index}")
        self.assertEqual(collected, commands(70))
        self.assertEqual(receipt["accepted_steps"], receipt["executed_steps"])

    def test_append_checks_tail_and_does_not_mutate_original_idempotency_record(self):
        original = self.submit(commands=commands(2))
        appended = self.submit(request_id="append", expected_tail=2, commands=commands(3))
        self.assertEqual(appended["execution_id"], original["execution_id"])
        self.assertEqual(appended["accepted_steps"], 5)
        self.assertEqual(self.submit(commands=commands(2)), original)
        self.unchanged_rejection("tail_conflict", lambda: self.submit(
            request_id="stale-append", expected_tail=2, commands=commands(1)))
        self.unchanged_rejection("stale_plan", lambda: self.submit(
            request_id="wrong-version", expected_tail=5, plan_version=2))
        self.unchanged_rejection("queue_limit", lambda: self.submit(
            request_id="too-long", expected_tail=5, commands=commands(16)))

    def test_mutating_caller_commands_or_returned_window_cannot_change_queue(self):
        receipt = self.submit()
        self.submit_args["commands"][0]["left_ee_pose"][0] = 99
        window = self.dispatch(receipt)
        window["commands"][0]["left_ee_pose"][0] = 98
        reread = self.queue.read_window(**self.identity(receipt), window_id=window["window_id"])
        self.assertEqual(reread["commands"][0]["left_ee_pose"][0], 0)

    def test_dispatch_in_flight_rejected_read_is_passive_and_ack_is_cumulative(self):
        receipt = self.submit()
        window = self.dispatch(receipt)
        before = self.queue.snapshot()
        self.assertEqual(self.queue.read_window(**self.identity(receipt),
                                                window_id=window["window_id"]), window)
        self.assertEqual(self.queue.snapshot(), before)
        self.unchanged_rejection("window_in_flight", lambda: self.dispatch(receipt))
        receipt = self.ack(receipt, window, 1)
        self.assertEqual(receipt["executed_steps"], 1)
        self.assertEqual(self.ack(receipt, window, 1), receipt)
        self.unchanged_rejection("evidence_conflict", lambda: self.ack(receipt, window, 2))
        self.unchanged_rejection("invalid_prefix", lambda: self.ack(receipt, window, 1, "new"))
        self.unchanged_rejection("invalid_prefix", lambda: self.ack(receipt, window, 4, "outside"))
        receipt = self.ack(receipt, window, 3, "receipt-3")
        self.assertEqual(self.ack(receipt, window, 3, "receipt-3"), receipt)
        next_window = self.dispatch(receipt)
        self.unchanged_rejection("stale_window", lambda: self.ack(receipt, window, 2, "late"))
        self.assertEqual(next_window["start_index"], 3)

    def test_pause_accepts_in_flight_ack_but_requires_reconciliation_before_resume(self):
        receipt = self.submit()
        window = self.dispatch(receipt)
        receipt = self.queue.pause(**self.identity(receipt), reason="new observation required")
        self.unchanged_rejection("not_dispatchable", lambda: self.dispatch(receipt))
        resume = lambda evidence: self.queue.resume(**self.identity(receipt),
            plan_id="plan", plan_version=1, evidence_id=evidence)
        self.unchanged_rejection("window_in_flight", lambda: resume("fresh"))
        receipt = self.ack(receipt, window, 3)
        self.assertEqual(receipt["status"], "paused")
        self.unchanged_rejection("evidence_conflict", lambda: resume("receipt-1"))
        receipt = resume("fresh")
        self.assertEqual(receipt["status"], "queued")
        self.assertEqual(self.dispatch(receipt)["start_index"], 3)

    def test_cancel_preserves_confirmed_prefix_and_distinguishes_unknown_from_discarded(self):
        receipt = self.submit()
        window = self.dispatch(receipt)
        receipt = self.ack(receipt, window, 1)
        receipt = self.queue.cancel(**self.identity(receipt), reason="cancel requested")
        self.assertEqual([receipt[key] for key in (
            "executed_steps", "unknown_steps", "discarded_steps", "remaining_steps", "in_flight_steps")],
            [1, 2, 5, 0, 0])
        self.assertEqual(self.queue.cancel(**self.identity(receipt), reason="cancel requested"), receipt)
        self.unchanged_rejection("not_acknowledgeable", lambda: self.ack(receipt, window, 2, "late"))
        self.unchanged_rejection("not_dispatchable", lambda: self.dispatch(receipt))
        self.unchanged_rejection("execution_unresolved", lambda: self.submit(request_id="late-new-request"))
        self.unchanged_rejection("execution_unresolved", lambda: self.submit(
            request_id="different-plan", plan_id="different", plan_version=5))
        stopped = self.queue.confirm_discarded(**self.identity(receipt), evidence_id="rollout-stopped")
        self.assertEqual(stopped["unknown_steps"], 0)
        self.assertEqual(stopped["discarded_steps"], 7)
        self.assertEqual(stopped["remaining_steps"], 0)
        self.assertEqual(self.queue.confirm_discarded(
            **self.identity(receipt), evidence_id="rollout-stopped"), stopped)
        self.unchanged_rejection("stale_plan", lambda: self.submit(request_id="late-new-request"))
        newer = self.submit(request_id="new-version", plan_version=2)
        self.assertNotEqual(newer["execution_id"], receipt["execution_id"])
        self.queue.confirm_discarded(**self.identity(receipt), evidence_id="rollout-stopped")
        self.unchanged_rejection("stale_plan", lambda: self.submit(
            request_id="new-third-plan", plan_id="third-plan"))

    def test_fault_cannot_resume_or_accept_late_results(self):
        receipt = self.submit()
        window = self.dispatch(receipt)
        receipt = self.queue.fault(**self.identity(receipt), reason="tracking rejection")
        self.assertEqual(receipt["status"], "faulted")
        self.assertEqual(receipt["unknown_steps"], 3)
        self.unchanged_rejection("not_resumable", lambda: self.queue.resume(
            **self.identity(receipt), plan_id="plan", plan_version=1, evidence_id="continue"))
        self.unchanged_rejection("not_acknowledgeable", lambda: self.ack(receipt, window, 1))

    def test_new_plan_id_may_start_at_any_positive_version(self):
        receipt = self.submit(commands=commands(1))
        receipt = self.ack(receipt, self.dispatch(receipt), 1)
        newer = self.submit(request_id="new", plan_id="different", plan_version=7)
        self.assertEqual(newer["plan_version"], 7)

    def test_reset_isolates_old_submissions_results_and_ids(self):
        receipt = self.submit()
        window = self.dispatch(receipt)
        self.queue.reset(episode_id="episode-2")
        self.unchanged_rejection("stale_episode", lambda: self.submit(request_id="late"))
        self.unchanged_rejection("stale_episode", lambda: self.ack(receipt, window, 1))
        self.unchanged_rejection("episode_reused", lambda: self.queue.reset(episode_id="episode"))
        newer = self.submit(episode_id="episode-2")
        self.assertEqual(newer["status"], "queued")

    def test_invalid_tail_is_validated_before_accepting_any_commands(self):
        variants = []
        for key, value in (("left_ee_pose", [0, 0, 0, 0, 0, 0, 0]),
                           ("left_ee_pose", [0, 0, 0, True, 0, 0, 0]),
                           ("left_ee_pose", [float("nan"), 0, 0, 1, 0, 0, 0]),
                           ("left_ee_joint_state", [1.1]),
                           ("left_ee_joint_state", [True]), ("delta", [1])):
            variant = commands(8)
            variant[-1][key] = value
            variants.append(variant)
        for variant in variants:
            with self.subTest(command=variant[-1]):
                self.unchanged_rejection("invalid_command", lambda: self.submit(commands=variant))
        self.unchanged_rejection("invalid_argument", lambda: self.submit(plan_version=True))
        self.unchanged_rejection("invalid_argument", lambda: self.submit(expected_tail=False))
        self.unchanged_rejection("invalid_argument", lambda: self.submit(request_id=" "))

    def test_numeric_boolean_config_and_acknowledgements_are_rejected(self):
        for kwargs in ({"max_window_steps": True}, {"max_queue_steps": False},
                       {"max_window_steps": 1.0}, {"max_window_steps": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ExecutionError):
                ExecutionQueue("episode", **kwargs)
        receipt = self.submit()
        window = self.dispatch(receipt)
        self.unchanged_rejection("invalid_argument", lambda: self.ack(receipt, window, True))
        self.unchanged_rejection("invalid_argument", lambda: self.ack(receipt, window, 0))

    def test_snapshot_restore_latches_pending_work_and_keeps_completed_receipts(self):
        receipt = self.submit()
        window = self.dispatch(receipt)
        self.ack(receipt, window, 1)
        snapshot = json.loads(json.dumps(self.queue.snapshot()))
        restored = ExecutionQueue.from_snapshot(snapshot)
        result = restored.query(**self.identity(receipt))
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["executed_steps"], 1)
        self.assertEqual(result["unknown_steps"], 2)
        self.assertEqual(result["remaining_steps"], 5)
        self.assertEqual(snapshot, self.queue.snapshot())
        self.assertEqual(restored.submit(**self.submit_args), receipt)
        with self.assertRaises(ExecutionError):
            restored.dispatch(**self.identity(receipt), plan_id="plan", plan_version=1)
        with self.assertRaises(ExecutionError):
            restored.resume(**self.identity(receipt), plan_id="plan", plan_version=1,
                            evidence_id="fresh")
        with self.assertRaises(ExecutionError):
            restored.acknowledge(**self.identity(receipt), window_id=window["window_id"],
                                 executed_steps=3, evidence_id="late")
        twice = ExecutionQueue.from_snapshot(json.loads(json.dumps(restored.snapshot())))
        self.assertEqual(twice.query(**self.identity(receipt)), result)
        stopped = restored.cancel(**self.identity(receipt), reason="reconcile externally")
        self.assertEqual(stopped["discarded_steps"], 5)

    def test_snapshot_restores_completed_execution_without_claiming_physical_success(self):
        receipt = self.submit(commands=commands(2))
        receipt = self.ack(receipt, self.dispatch(receipt), 2)
        restored = ExecutionQueue.from_snapshot(self.queue.snapshot())
        self.assertEqual(restored.query(**self.identity(receipt)), receipt)
        self.assertIsNone(receipt["task_success"])

    def test_corrupt_snapshot_states_and_counters_are_rejected(self):
        receipt = self.submit()
        self.dispatch(receipt)
        snapshot = self.queue.snapshot()
        for key, value in (("status", "stopped-ish"), ("executed", True),
                           ("executed", 4), ("dispatched", 9), ("unknown", 2),
                           ("status", "succeeded"), ("window", None)):
            candidate = copy.deepcopy(snapshot)
            candidate["executions"][receipt["execution_id"]][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ExecutionError):
                ExecutionQueue.from_snapshot(candidate)
        self.assertEqual(self.queue.snapshot(), snapshot)

    def test_snapshot_keeps_uncertain_termination_locked_until_discard_evidence(self):
        receipt = self.submit()
        self.dispatch(receipt)
        self.queue.fault(**self.identity(receipt), reason="controller lost")
        restored = ExecutionQueue.from_snapshot(self.queue.snapshot())
        with self.assertRaises(ExecutionError):
            restored.submit(**(self.submit_args | {"request_id": "new", "plan_id": "new"}))
        restored.confirm_discarded(**self.identity(receipt), evidence_id="controller-confirmed-prefix")
        twice = ExecutionQueue.from_snapshot(restored.snapshot())
        recovered = twice.submit(**(self.submit_args | {"request_id": "new", "plan_id": "new"}))
        self.assertEqual(recovered["status"], "queued")

    def test_snapshot_round_trip_does_not_depend_on_json_key_order(self):
        receipt = self.submit(commands=commands(4), request_id="z-first")
        first_window = self.dispatch(receipt)
        self.ack(receipt, first_window, 1, "first")
        self.submit(commands=commands(2), request_id="a-append", expected_tail=4)
        self.ack(receipt, first_window, 3, "last")
        last_window = self.dispatch(receipt)
        receipt = self.ack(receipt, last_window, 3, "final")
        snapshot = json.loads(json.dumps(self.queue.snapshot(), sort_keys=True))
        self.assertEqual(ExecutionQueue.from_snapshot(snapshot).query(**self.identity(receipt)), receipt)

    def test_snapshot_rejects_fabricated_receipt_success_bool_counts_and_missing_evidence(self):
        receipt = self.submit()
        self.ack(receipt, self.dispatch(receipt), 1)
        snapshot = self.queue.snapshot()
        for key, value in (("status", "succeeded"), ("task_success", True),
                           ("physical_state_verified", True), ("accepted_steps", True),
                           ("confirmed_prefix", [False, 0])):
            candidate = copy.deepcopy(snapshot)
            candidate["requests"]["request"]["receipt"][key] = value
            with self.subTest(key=key), self.assertRaises(ExecutionError):
                ExecutionQueue.from_snapshot(candidate)
        candidate = copy.deepcopy(snapshot)
        candidate["executions"][receipt["execution_id"]]["evidence"] = {}
        with self.assertRaises(ExecutionError):
            ExecutionQueue.from_snapshot(candidate)
        candidate = copy.deepcopy(snapshot)
        candidate["requests"] = {}
        with self.assertRaises(ExecutionError):
            ExecutionQueue.from_snapshot(candidate)


if __name__ == "__main__":
    unittest.main()
