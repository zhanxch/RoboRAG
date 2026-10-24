import unittest
from roborag.fold_comparison import aggregate


class FoldComparisonTests(unittest.TestCase):
    def episode(self, success=True, complete=True, usage=None):
        return dict(complete=complete, success=success, progress=100 if success else 20,
                    steps=300, wall_seconds=240, agent_wall_seconds=200,
                    usage=usage if usage is not None else dict(input_tokens=1000,cached_input_tokens=900,output_tokens=20,reasoning_output_tokens=5))

    def test_pending_or_incomplete_cases_do_not_become_final_success_rate(self):
        report=aggregate([self.episode()], planned=20)
        self.assertIsNone(report['success_rate'])
        report=aggregate([self.episode(), self.episode(complete=False)], planned=2)
        self.assertIsNone(report['success_rate'])
        self.assertEqual(report['scored'],1)

    def test_token_breakdowns_are_not_added_twice(self):
        report=aggregate([self.episode(),self.episode(False)])
        self.assertEqual(report['success_rate'],.5)
        self.assertEqual(report['usage']['input_tokens'],2000)
        self.assertEqual(report['usage']['output_tokens'],40)
        self.assertEqual(report['uncached_input_tokens'],200)
        self.assertEqual(report['input_cache_fraction'],.9)

    def test_unreported_usage_stays_unknown(self):
        report=aggregate([self.episode(),self.episode(usage={})])
        self.assertIsNone(report['usage']['input_tokens'])
        self.assertEqual(report['usage']['input_tokens_known_lower_bound'],1000)

    def test_more_results_than_planned_is_not_a_valid_denominator(self):
        with self.assertRaises(ValueError):
            aggregate([self.episode(), self.episode()], planned=1)

    def test_incomplete_episode_wall_cost_is_kept_and_missing_is_unknown(self):
        complete, failed = self.episode(), self.episode(complete=False)
        self.assertEqual(aggregate([complete, failed])["sum_episode_wall_seconds"], 480)
        failed["wall_seconds"] = None
        summary = aggregate([complete, failed])
        self.assertIsNone(summary["sum_episode_wall_seconds"])
        self.assertEqual(summary["sum_episode_wall_seconds_known_lower_bound"], 240)


if __name__ == '__main__':
    unittest.main()
