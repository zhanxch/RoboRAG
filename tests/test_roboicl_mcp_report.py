import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("mcp_report", Path(__file__).resolve().parents[1] / "scripts/summarize_roboicl_mcp.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class MCPReportTests(unittest.TestCase):
    def fixture(self, folder):
        save(folder / "frozen-plan.json", {"mode": "model_evaluation", "model": "gpt-6-astra", "effort": "xhigh", "layout": 0})
        call = folder / "decisions/call-0001"
        save(call / "invocation.json", {"model": "gpt-6-astra", "effort": "xhigh",
            "command": ["codex", "--model", "gpt-6-astra", "-c", 'model_reasoning_effort="xhigh"']})
        save(call / "execution.json", {"wall_seconds": 30})
        save(call / "response.json", {"status": "completed"})
        return call

    def test_no_native_result_or_missing_usage_remain_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            save(root / "exit.json", {"status": "failed", "wall_seconds": 40})
            result = report.summarize(root)
        self.assertFalse(result["valid_native_episode"])
        self.assertIsNone(result["native_score_0_100"])
        self.assertIsNone(result["total_tokens"])
        self.assertEqual(result["calls_without_complete_usage"], 1)

    def test_snapshot_audit_detects_missing_and_changed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source").mkdir()
            path = root / "source/policy.py"
            path.write_text("original")
            plan = {"source_sha256": {"policy.py": hashlib.sha256(path.read_bytes()).hexdigest()}}
            self.assertTrue(report.source_audit(root, plan)["all_match"])
            path.write_text("changed")
            self.assertEqual(report.source_audit(root, plan)["changed_or_missing"], ["policy.py"])
            path.unlink()
            self.assertFalse(report.source_audit(root, plan)["all_match"])

    def test_scored_run_requires_step_aligned_video_and_includes_cached_input_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            call = self.fixture(root)
            (call / "events.jsonl").write_text(json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 100, "output_tokens": 10, "cached_input_tokens": 50,
                "reasoning_output_tokens": 6, "cache_write_input_tokens": 0}}) + "\n")
            save(root / "exit.json", {"status": "native_completed", "steps": 1, "wall_seconds": 40})
            native = root / "native/_result.json"
            save(native, {"eval_time": 1, "score": 20, "success_rate": 0, "details": {"0": {"layout_id": 0}}})
            save(root / "native/evaluation.json", {"status": "scored", "steps": 1, "result": str(native)})
            (root / "mcp-calls.jsonl").write_text(json.dumps({"tool": "control_step", "status": "returned", "seconds": .1}) + "\n")
            for name in ("head", "left_wrist", "right_wrist"):
                (native.parent / f"episode_cam_{name}_fail.mp4").touch()
            with patch.object(report.subprocess, "check_output", return_value=json.dumps({"streams": [{"nb_frames": "2"}]})):
                result = report.summarize(root)
            self.assertTrue(result["valid_native_episode"])
            self.assertEqual(result["total_tokens"], 110)
            self.assertEqual(result["reasoning_output_tokens"], 6)
            self.assertEqual(result["native_score_0_100"], 20)
            with patch.object(report.subprocess, "check_output", return_value=json.dumps({"streams": [{"nb_frames": "3"}]})):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    report.summarize(root)
