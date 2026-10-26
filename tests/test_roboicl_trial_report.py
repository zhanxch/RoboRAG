"""Incomplete native results and rejected model calls remain visible in audits."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "trial_report", Path(__file__).resolve().parents[1] / "scripts/summarize_roboicl_trial.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class TrialReportTests(unittest.TestCase):
    def test_score_is_withheld_without_clean_native_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            native = folder / "upstream-results/eval_result/trial/_result.json"
            native.parent.mkdir(parents=True)
            native.write_text(json.dumps({"eval_time": 1, "score": 100, "success_rate": 1,
                                          "details": {"0": {"layout_id": 0, "success": True}}}))
            result = report.summarize(folder, model="gpt-6-sol", effort="xhigh", layout=0)
            self.assertFalse(result["valid_native_episode"])
            self.assertIsNone(result["native_score_0_100"])
            self.assertIsNone(result["native_success"])

    def test_failed_call_usage_is_included_and_model_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            call = folder / "decisions/call-0001"
            call.mkdir(parents=True)
            (call / "invocation.json").write_text(json.dumps({"model": "gpt-6-sol", "effort": "xhigh",
                                                             "command": ["codex", "--model", "gpt-6-astra"]}))
            (call / "events.jsonl").write_text(json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 30, "output_tokens": 5, "cached_input_tokens": 10}}) + "\n")
            result = report.summarize(folder, model="gpt-6-sol", effort="xhigh", layout=0)
            self.assertEqual(result["known_total_tokens"], 35)
            self.assertEqual(result["completed_model_calls"], 0)
            self.assertEqual(result["attempted_model_calls"], 1)
            self.assertEqual(result["calls_without_usage"], 0)
            self.assertFalse(result["model_configuration_matches"])

    def test_missing_usage_stays_unknown_and_command_effort_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            call = folder / "decisions/call-0001"
            call.mkdir(parents=True)
            (call / "invocation.json").write_text(json.dumps({"model": "gpt-6-sol", "effort": "xhigh",
                "command": ["codex", "--model", "gpt-6-sol", "-c", 'model_reasoning_effort="medium"']}))
            (call / "events.jsonl").write_text(json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 30, "output_tokens": None}}) + "\n")
            result = report.summarize(folder, model="gpt-6-sol", effort="xhigh", layout=0)
            self.assertIsNone(result["cached_input_tokens"])
            self.assertIsNone(result["total_tokens"])
            self.assertEqual(result["known_total_tokens"], 30)
            self.assertEqual(result["calls_without_usage"], 1)
            self.assertEqual(result["calls_without_complete_cache_usage"], 1)
            self.assertEqual(result["calls"][0]["command_effort"], "medium")
            self.assertFalse(result["model_configuration_matches"])


if __name__ == "__main__":
    unittest.main()
