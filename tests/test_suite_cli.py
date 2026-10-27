import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SuiteCommandTests(unittest.TestCase):
    def test_registered_pour_task_uses_native_results_and_preserves_pending_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            cases = [dict(seed=1, variant="standard", layout=i) for i in range(3)]
            for layout, success in enumerate((1, 0)):
                run = runs / f"layout-{layout}"
                run.mkdir(parents=True)
                config = dict(
                    task="pour_balls_into_vase",
                    model="test-model",
                    layout_sha256=f"scene-{layout}",
                    prompt_sha256="frozen-prompt",
                    source_sha256={},
                    args=dict(
                        **cases[layout],
                        effort="xhigh",
                        controller="delta",
                        no_rag=False,
                    ),
                )
                (run / "run.json").write_text(json.dumps(config))
                # Progress alone does not establish full task success.
                native = dict(eval_time=1, score=100, success_rate=success)
                native_path = run / "native.json"
                native_path.write_text(json.dumps(native))
                (run / "result.json").write_text(
                    json.dumps(
                        dict(
                            complete=True,
                            native=native,
                            native_result=str(native_path),
                            native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
                        )
                    )
                )
            case_path, report_path = root / "cases.json", root / "report.json"
            case_path.write_text(json.dumps(cases))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "roborag",
                    "evaluate",
                    "--task",
                    "pour_balls_into_vase",
                    "--runs",
                    str(runs),
                    "--cases",
                    str(case_path),
                    "--output",
                    str(report_path),
                ],
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["task"], "pour_balls_into_vase")
            self.assertEqual(report["overall"]["successes"], 1)
            self.assertEqual(report["overall"]["scored"], 2)
            self.assertEqual(report["overall"]["scored_success_rate"], 0.5)
            self.assertIsNone(report["overall"]["success_rate"])
            self.assertEqual(report["pending"], [cases[2]])


if __name__ == "__main__":
    unittest.main()
