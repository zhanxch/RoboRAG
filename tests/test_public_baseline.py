"""Public comparisons must use completed, correctly identified rollout evidence."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_roboicl_suite import aggregate, public_layout
from import_roboicl_baseline import normalize


class PublicBaselineTests(unittest.TestCase):
    def raw(self):
        row = {"id": 2, "variant": "standard", "score": 20, "success": False,
               "officialScore": 20, "officialSuccess": False,
               "includedInAggregate": True, "trace": "trace.json"}
        return {"version": 1, "snapshotDate": "test", "summary": {
                    "fullEvaluationRolloutCount": 1, "partialEvaluationRolloutCount": 0,
                    "overallScore": 20},
                "tasks": [{"id": "swap_t", "name": "Swap T", "category": "Memory",
                           "shots": 1, "officialScore": 20, "officialLayoutCount": 1,
                           "evaluationStatus": "complete", "layouts": [row]}]}

    def test_progress_is_not_success_and_task_case_is_normalized(self):
        baseline = normalize(self.raw())
        task = baseline["tasks"][0]
        self.assertEqual(task["task"], "swap_T")
        self.assertEqual(task["completed_success_rate"], 0)
        self.assertEqual(public_layout(baseline, "swap_T", 0, "standard", 2)["official_score_0_100"], 20)
        with self.assertRaises(StopIteration):
            public_layout(baseline, "swap_T", 0, "random", 2)

    def test_partial_rollout_cannot_be_baseline(self):
        raw = self.raw()
        row = copy.deepcopy(raw["tasks"][0]["layouts"][0])
        row.update(id=3, score=100, success=True, includedInAggregate=False)
        raw["tasks"][0]["layouts"].append(row)
        raw["summary"]["partialEvaluationRolloutCount"] = 1
        baseline = normalize(raw)
        self.assertEqual(baseline["tasks"][0]["completed_success_rate"], 0)
        with self.assertRaises(ValueError):
            public_layout(baseline, "swap_T", 0, "standard", 3)

    def test_published_mean_must_match_episode_evidence(self):
        raw = self.raw()
        raw["tasks"][0]["officialScore"] = 50
        with self.assertRaises(ValueError):
            normalize(raw)

    def test_infrastructure_failure_is_never_zero_score(self):
        self.assertIsNone(aggregate([{"status": "incomplete_or_infrastructure_failure"}]))


if __name__ == "__main__":
    unittest.main()
