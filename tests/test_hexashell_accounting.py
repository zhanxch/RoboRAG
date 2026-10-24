import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from roborag.hexashell_accounting import event_accounting, read_events, resource_accounting


class AccountingTests(unittest.TestCase):
    def test_no_terminal_usage_is_unknown(self):
        result = event_accounting([{"type": "turn.failed"}])
        self.assertIsNone(result["reported_usage"]["input_tokens"])
        self.assertFalse(result["terminal_usage_available"])

    def test_cached_and_reasoning_counts_are_not_double_added(self):
        usage = dict(input_tokens=100, cached_input_tokens=80, cache_write_input_tokens=0,
                     output_tokens=20, reasoning_output_tokens=10)
        result = event_accounting([dict(type="turn.completed", usage=usage)] * 2)
        self.assertEqual(result["reported_usage"]["input_tokens"], 200)
        self.assertEqual(result["reported_usage"]["output_tokens"], 40)
        self.assertIsNone(result["model_request_count"])

    def test_truncated_stream_preserves_known_usage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_text(json.dumps(dict(type="turn.completed", usage=dict(input_tokens=100)))+'\n{"type":')
            result = event_accounting(*read_events(path))
        self.assertEqual(result["reported_usage"]["input_tokens"], 100)
        self.assertIsNone(result["reported_usage"]["output_tokens"])
        self.assertFalse(result["terminal_usage_available"])

    def test_energy_uses_elapsed_time_and_does_not_fill_gaps(self):
        samples = [dict(epoch=t, gpu="0, 100, 0", gpu_power_w=p) for t,p in [(0,100),(10,200),(60,200),(70,None)]]
        result = resource_accounting(samples)
        self.assertAlmostEqual(result["estimated_gpu_board_energy_wh"], 1500/3600)
        self.assertEqual(result["power_sample_coverage_seconds"], 10)
        self.assertIsNone(result["peak_simultaneous_owned_rss_bytes"])


if __name__ == "__main__":
    unittest.main()
