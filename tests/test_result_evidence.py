"""Regression checks for misleading simulator exit codes and interrupted scores."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_runs import summarize


class NativeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.write("run.json", {"arguments": {"mode": "dual", "task": "deposit_coin", "layout": 0},
                                "evaluation_role": "development"})

    def write(self, name, value):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_zero_exit_without_native_result_is_not_a_scored_episode(self):
        self.write("exit.json", {"returncode": 0})
        result = summarize(self.folder)
        self.assertFalse(result["valid_native_episode"])
        self.assertIsNone(result["score_0_100"])

    def test_native_progress_is_distinct_from_full_success(self):
        self.write("native/_result.json", {"eval_time": 1, "score": 20., "success_rate": 0.,
                   "details": {"0": {"score": .2, "success": False}}})
        result = summarize(self.folder)
        self.assertTrue(result["valid_native_episode"])
        self.assertEqual(result["score_0_100"], 20.)
        self.assertEqual(result["success_rate"], 0.)

    def test_interrupted_high_score_is_excluded(self):
        self.write("native/_result.json", {"eval_time": 1, "score": 100., "success_rate": 0.,
                   "details": {"0": {"score": 1., "interrupted": True}}})
        result = summarize(self.folder)
        self.assertFalse(result["valid_native_episode"])
        self.assertIsNone(result["score_0_100"])


if __name__ == "__main__":
    unittest.main()
