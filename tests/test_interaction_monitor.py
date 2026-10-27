import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from roborag.interaction_monitor import snapshot


class InteractionMonitorTests(unittest.TestCase):
    def test_pending_motion_and_feedback_do_not_assert_plan_stage_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exchange = root / "workspace"
            (exchange / "inbox").mkdir(parents=True)
            (exchange / "outbox").mkdir()
            (root / "agent-workspace.json").write_text(json.dumps(dict(path=str(exchange))))
            (root / "gpt-plan.json").write_text(
                json.dumps(dict(steps=[dict(subgoal="fold")], adaptation=""))
            )
            request = dict(
                subgoal="check grip with small lift",
                expected_feedback="cloth follows",
                actions=[[0] * 14],
            )
            (exchange / "inbox/001.json").write_text(json.dumps(request))
            pending = snapshot(root)
            self.assertEqual(pending["current_interaction"]["label"], request["subgoal"])
            self.assertEqual(pending["current_interaction"]["status"], "awaiting_feedback")
            self.assertIsNone(pending["current_interaction"]["feedback"])
            (exchange / "outbox/001.json").write_text(
                json.dumps(dict(step_after=8, result=dict(plan_ok=True)))
            )
            before = {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file()
            }
            observed = snapshot(root)
            self.assertEqual(observed["current_interaction"]["step"], 8)
            self.assertEqual(observed["current_interaction"]["status"], "feedback_available")
            self.assertIsNone(observed["native_result"])
            self.assertEqual(observed["plan"]["steps"], [dict(subgoal="fold")])
            self.assertEqual(
                before,
                {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.rglob("*")
                    if p.is_file()
                },
            )

    def test_missing_partial_and_interrupted_logs_do_not_invent_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(snapshot(root)["status"], "waiting_for_logs")
            (root / "progress.json").write_text('{"subgoal":')
            self.assertTrue(snapshot(root)["unreadable"])
            (root / "exit.json").write_text(
                json.dumps(dict(complete=False, error="KeyboardInterrupt"))
            )
            self.assertEqual(snapshot(root)["status"], "interrupted")
            self.assertIsNone(snapshot(root)["native_result"])

    def test_archived_episode_uses_recorded_subgoal_and_native_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "progress.json").write_text(
                json.dumps(dict(subgoal="release", step_after=500, result=dict(episode_over=True)))
            )
            (root / "result.json").write_text(
                json.dumps(dict(complete=True, native=dict(score=20, success_rate=0)))
            )
            observed = snapshot(root)
            self.assertEqual(observed["status"], "completed")
            self.assertEqual(observed["current_interaction"]["label"], "release")
            self.assertEqual(observed["native_result"]["success_rate"], 0)
