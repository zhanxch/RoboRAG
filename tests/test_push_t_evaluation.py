import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from roborag.evaluation.comparison import candidate_rows
from roborag.evaluation.suite import paired, suite


def row(layout=0, success=True, complete=True):
    return dict(
        seed=1,
        variant="random",
        layout=layout,
        complete=complete,
        success=success,
        progress=100 if success else 0,
        steps=200,
        wall_seconds=100,
        agent_wall_seconds=80,
        usage=dict(
            input_tokens=1000, cached_input_tokens=800, output_tokens=50, reasoning_output_tokens=20
        ),
        layout_sha256="same-scene",
        model="same-model",
        effort="xhigh",
        protocol=dict(controller="delta", no_rag=False),
    )


class PushTEvaluationTests(unittest.TestCase):
    def test_pending_incomplete_and_failed_runs_remain_distinct(self):
        cases = [dict(seed=1, variant="random", layout=i) for i in range(3)]
        result = suite([row(0), row(1, success=None, complete=False)], cases)
        self.assertEqual(result["pending"], [cases[2]])
        self.assertEqual(result["overall"]["scored"], 1)
        self.assertIsNone(result["overall"]["success_rate"])
        self.assertEqual(result["overall"]["sum_episode_wall_seconds"], 200)
        with self.assertRaises(ValueError):
            suite([row(), row()], cases)
        with self.assertRaises(ValueError):
            suite([row(4)], cases)

    def test_matched_comparison_checks_scene_and_does_not_reward_failed_early_exit(self):
        baseline = [row(0), row(1)]
        candidate = [row(0), row(1, success=False)]
        candidate[1]["wall_seconds"] = 1
        result = paired(baseline, candidate)
        self.assertEqual(result["candidate"]["success_rate"], 0.5)
        self.assertEqual(result["jointly_successful"]["count"], 1)
        self.assertEqual(result["jointly_successful"]["candidate"]["mean_wall_seconds"], 100)
        for key in ("layout_sha256", "model", "effort"):
            changed = copy.deepcopy(candidate)
            changed[0][key] = "different"
            with self.assertRaises(ValueError):
                paired(baseline, changed)
        with self.assertRaises(ValueError):
            paired(baseline[:1], candidate)

    def test_complete_claim_requires_matching_native_result_and_valid_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config = dict(
                task="push_T_random",
                model="test",
                layout_sha256="scene",
                prompt_sha256="prompt",
                source_sha256={},
                args=dict(
                    seed=1,
                    variant="random",
                    layout=0,
                    effort="xhigh",
                    controller="delta",
                    no_rag=False,
                ),
            )
            (path / "run.json").write_text(json.dumps(config))
            native = dict(eval_time=1, score=100, success_rate=1)
            native_path = path / "native.json"
            native_path.write_text(json.dumps(native))
            result = dict(
                complete=True,
                native=native,
                native_result=str(native_path),
                native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
            )
            (path / "result.json").write_text(json.dumps(result))
            self.assertTrue(candidate_rows([path], task="push_T")[0]["success"])
            native_path.write_text(json.dumps(dict(native, score=0)))
            with self.assertRaisesRegex(ValueError, "changed"):
                candidate_rows([path], task="push_T")
            native["eval_time"] = 0
            native_path.write_text(json.dumps(native))
            result["native_sha256"] = hashlib.sha256(native_path.read_bytes()).hexdigest()
            (path / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "Invalid native"):
                candidate_rows([path], task="push_T")


if __name__ == "__main__":
    unittest.main()
