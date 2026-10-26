import json
import tempfile
import unittest
from pathlib import Path

from roborag.evaluation.episode import summarize, workflow_completed


class EpisodeAccountingTests(unittest.TestCase):
    def test_motion_survives_missing_mcp_reply_and_partial_log_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "actions.jsonl").write_text(
                "".join(json.dumps({"step": i, "delta_row": [0] * 14}) + "\n" for i in range(1, 11))
                + '{"step":11'
            )
            (root / "mcp-calls.jsonl").write_text('{"tool":')
            report = summarize(root)
            self.assertEqual(report["native_control_steps"], 10)
            self.assertEqual(report["executed_delta_rows"], 10)
            self.assertEqual(report["control_steps_status"], "lower_bound")
            self.assertEqual(report["malformed_mcp_records"], 1)
            self.assertFalse(report["native_control_steps_complete"])
            (root / "episode_end.json").write_text('{"step":11}')
            self.assertEqual(summarize(root)["native_control_steps"], 11)
            self.assertTrue(summarize(root)["native_control_steps_complete"])

    def test_missing_counter_is_unknown_and_zero_action_episode_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(summarize(root)["native_control_steps"])
            (root / "episode_end.json").write_text('{"step":0}')
            self.assertEqual(summarize(root)["native_control_steps"], 0)
            self.assertFalse(workflow_completed([]))


if __name__ == "__main__":
    unittest.main()
