"""Exercise the real upstream policy/receipt code with deterministic fake inference."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec("numpy") is not None and (ROOT / "vendor/RoboICL").is_dir()
if AVAILABLE:
    sys.path.insert(0, str(ROOT / "vendor/RoboICL"))
    import numpy as np
    from roboicl.policy.astra_policy import CAMERAS, plain
    from roboicl.policy.dialogue_policy import Model as DialogueModel
    from roborag.control_profiles import apply_control_profile
    from roborag.queued_policy import QueuedDialogueModel


@unittest.skipUnless(AVAILABLE, "optional robot runtime not installed")
class QueuedPolicyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        config = {"max_turns": 1, "max_action_corrections": 0, "max_seconds": 600,
                  "cache_mode": "omit", "feedback_frames": 1,
                  "include_head_calibration": False}
        apply_control_profile(config, "queued", task="put_bottles_into_dustbin", reference_horizon=10)
        (self.root / "config.json").write_text(json.dumps(config))
        environment = patch.dict(os.environ, {
            "ASTRA_HARNESS_CONFIG": str(self.root / "config.json"),
            "ASTRA_LOG_DIR": str(self.root / "policy"), "ASTRA_API_KEY": "test-only",
            "ASTRA_ENDPOINT": "http://127.0.0.1:1/v1/responses", "ASTRA_API_MODE": "responses",
            "ASTRA_DEMO_MANIFEST": "", "ASTRA_REASONING_EFFORT": "xhigh",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.requests = []
        self.rows = [[.30, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]] * 4
        self.mode = "transit"
        def reply(model, payload):
            self.requests.append(deepcopy(payload))
            return {"output": [{"type": "function_call", "name": "act", "call_id": "segment-1",
                               "arguments": json.dumps({"actions": self.rows, "control_mode": self.mode,
                                                        "execution_note": "Test authorized transit."})}]}
        inference = patch.object(DialogueModel, "_request", reply)
        inference.start()
        self.addCleanup(inference.stop)
        self.model = QueuedDialogueModel({"model": "fake", "task_name": "put_bottles_into_dustbin"})
        self.state = {f"{side}_ee_pose": [0., 0., .5, 1., 0., 0., 0.] for side in ("left", "right")}
        self.state.update({f"{side}_ee_joint_state": [1.] for side in ("left", "right")})
        self.state.update({f"{side}_arm_joint_state": [0.] * 6 for side in ("left", "right")})
        self.step = 0
        self.observe()

    def observe(self, action=None, *, bad_tracking=False, event=None):
        if action is not None:
            self.step += 1
            self.state.update(deepcopy(plain(action)))
        state = deepcopy(self.state)
        if bad_tracking:
            state["left_ee_pose"][0] -= .10
        meta = {"step": self.step, "max_episode_steps": 500}
        if action is not None:
            meta["executed_action"] = deepcopy(action)
        if event:
            meta["controller_event"] = event
        self.model.update_obs({"astra_meta": meta, "state": state, "instruction": "Test bottle disposal",
                               "vision": {camera: {"color": np.full((24, 24, 3), 120, dtype=np.uint8)}
                                          for camera in CAMERAS}})

    def receipt(self):
        rows = [json.loads(line) for line in (self.model.episode_dir / "tool_results.jsonl").read_text().splitlines()]
        return rows[-1]

    def execution(self):
        rows = [json.loads(line) for line in (self.model.episode_dir / "executions.jsonl").read_text().splitlines()]
        return rows[-1]

    def test_two_windows_use_one_model_call_and_one_complete_receipt(self):
        first = self.model.get_action()
        self.assertEqual(len(first), 32)
        for action in first:
            self.observe(action)
        second = self.model.get_action()
        self.assertEqual(len(second), 8)
        self.assertEqual(len(self.requests), 1)
        self.assertAlmostEqual(second[0]["left_ee_pose"][0], .99, places=5)
        for action in second:
            self.observe(action)
        self.model.finalize()
        receipt = self.receipt()
        self.assertEqual((receipt["predicted_steps"], receipt["dispatched_steps"], receipt["executed_steps"]), (40, 40, 40))
        self.assertEqual(receipt["unexecuted_steps"], 0)
        self.assertEqual(self.execution()["status"], "succeeded")
        self.assertIsNone(self.execution()["task_success"])
        self.assertAlmostEqual(self.state["left_ee_pose"][0], 1.2, places=5)
        self.assertTrue((self.model.episode_dir / "queue-state.json").is_file())

    def test_duplicate_get_without_observation_never_redispatches(self):
        self.model.get_action()
        with self.assertRaisesRegex(ValueError, "complete receipts"):
            self.model.get_action()
        self.assertEqual(len(self.requests), 1)

    def test_partial_bad_tracking_does_not_invent_a_stopped_window(self):
        first = self.model.get_action()
        self.observe(first[0], bad_tracking=True)
        with self.assertRaisesRegex(ValueError, "complete receipts"):
            self.model.get_action()
        receipt = self.model.execution_queue.query(**self.model._identity())
        self.assertEqual(receipt["in_flight_steps"], 31)
        self.assertEqual(receipt["discarded_steps"], 0)
        self.assertEqual(receipt["executed_steps"], 1)
        self.assertEqual(len(self.requests), 1)

    def test_ik_rejection_in_second_window_has_global_index_and_no_replay(self):
        for action in self.model.get_action():
            self.observe(action)
        second = self.model.get_action()
        for action in second[:2]:
            self.observe(action)
        self.model.notify_action_rejection({"step": self.step, "event": {
            "type": "action_rejected_ik_continuity", "rejected_action_executed": False,
            "action_index": 2, "discarded_commands": 6}})
        self.model.finalize()
        receipt = self.receipt()
        self.assertEqual(receipt["executed_steps"], 34)
        self.assertEqual(receipt["unexecuted_steps"], 6)
        self.assertEqual(receipt["controller_events"][-1]["action_index"], 34)
        self.assertEqual(self.execution()["discarded_steps"], 6)
        self.assertEqual(self.execution()["unknown_steps"], 0)

    def test_tracking_guard_on_window_boundary_invalidates_undispatched_tail(self):
        actions = self.model.get_action()
        for index, action in enumerate(actions):
            self.observe(action, bad_tracking=index >= 30)
        self.model._commit_feedback()
        receipt = self.receipt()
        self.assertEqual(receipt["executed_steps"], 32)
        self.assertEqual(receipt["unexecuted_steps"], 8)
        self.assertEqual(self.execution()["discarded_steps"], 8)
        self.assertEqual(self.execution()["status"], "faulted")
        self.assertEqual(receipt["controller_events"][-1]["type"], "queued_segment_tracking_error")

    def test_episode_end_preserves_executed_prefix_and_discards_rest(self):
        first = self.model.get_action()
        for action in first[:3]:
            self.observe(action)
        self.model.finalize()
        self.assertEqual(self.receipt()["executed_steps"], 3)
        self.assertEqual(self.execution()["discarded_steps"], 37)
        self.assertEqual(self.execution()["unknown_steps"], 0)
        self.assertEqual(self.execution()["status"], "cancelled")

    def test_one_bad_boundary_sample_does_not_cross_reset_simulator_guard(self):
        for index, action in enumerate(self.model.get_action()):
            self.observe(action, bad_tracking=index == 31)
        self.model._commit_feedback()
        self.assertEqual(self.execution()["discarded_steps"], 8)
        self.assertEqual(self.execution()["status"], "faulted")
        self.assertEqual(self.receipt()["controller_events"][-1]["type"], "queued_boundary_tracking_error")

    def test_mismatched_execution_evidence_cannot_ack_or_continue(self):
        first = self.model.get_action()
        wrong = deepcopy(first[0])
        wrong["left_ee_pose"][0] += .1
        self.observe(wrong)
        with self.assertRaisesRegex(ValueError, "differs"):
            self.model.get_action()
        self.assertEqual(self.model.execution_queue.query(**self.model._identity())["executed_steps"], 0)

    def test_precision_schema_and_mode_bounds_are_enforced(self):
        schema = self.model.frozen_tools[0]["parameters"]
        self.assertIn("control_mode", schema["required"])
        self.mode = "precision"
        self.rows = [[.005, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
        actions = self.model.get_action()
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["left_ee_joint_state"], [0.])

    def test_reference_conversion_bypasses_live_expansion(self):
        rows = [[.09, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]]
        result = self.model._convert_tensor(rows, self.state)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["left_ee_pose"][0], .09)
        self.assertIsNone(self.model._prepared_commands)

    def test_reset_clears_old_queue_and_execution(self):
        self.model.get_action()
        old_id = self.model.episode_id
        self.model.reset()
        self.assertNotEqual(old_id, self.model.episode_id)
        self.assertIsNone(self.model._execution)
        self.assertEqual(self.model.execution_queue.snapshot()["executions"], {})


if __name__ == "__main__":
    unittest.main()
