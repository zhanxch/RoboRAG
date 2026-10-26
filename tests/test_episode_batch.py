import contextlib
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from roborag.config import RuntimeConfig
from roborag.runtime import batch as episode_batch


class EpisodeBatchTests(unittest.TestCase):
    def test_seeded_layouts_have_distinct_keys_and_reject_duplicates(self):
        cases = episode_batch.normalize_cases(
            [
                dict(variant="random", layout=0),
                dict(seed=1, variant="random", layout=0),
            ]
        )
        self.assertEqual(
            [episode_batch.case_key(c) for c in cases], ["random-l0", "seed1-random-l0"]
        )
        self.assertEqual(
            episode_batch.normalize_cases([dict(variant="standard", layout=0)], 2)[0]["seed"], 2
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            episode_batch.normalize_cases(
                [dict(variant="random", layout=0), dict(seed=0, variant="random", layout=0)]
            )
        for invalid in (True, -1, "1"):
            with self.assertRaises(ValueError):
                episode_batch.normalize_cases([dict(seed=invalid, variant="random", layout=0)])

    def test_active_process_is_not_reported_finished_before_receipt_exists(self):
        sleep = time.sleep
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            (root / "local.runtime.json").write_text(
                json.dumps({name: sys.executable for name in RuntimeConfig.__dataclass_fields__})
            )
            (root / "scripts/run_rag_mcp.py").write_text("""import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args,_=p.parse_known_args()
args.output.mkdir();(args.output/'ready.json').write_text('{}')
time.sleep(.25)
(args.output/'result.json').write_text(json.dumps(dict(native=dict(score=100))))
""")
            out = root / "batch"
            argv = [
                "run_rag_batch.py",
                "--task",
                "deposit_coin",
                "--layouts",
                "0",
                "--seed",
                "1",
                "--parallel",
                "1",
                "--output",
                str(out),
            ]
            with (
                patch.object(episode_batch, "ROOT", root),
                patch.object(sys, "argv", argv),
                patch.object(
                    episode_batch, "resource_sample", return_value=dict(gpu="24000, 0, 0")
                ),
                patch.object(episode_batch, "protocol_sources", return_value={}),
                patch.object(episode_batch.time, "sleep", side_effect=lambda _: sleep(0.03)),
                contextlib.redirect_stdout(io.StringIO()) as logs,
            ):
                episode_batch.main()
            status = json.loads((out / "status.json").read_text())
            self.assertTrue(status["complete"])
            self.assertEqual(status["active"], {})
            self.assertEqual(len(status["finished"]), 1)
            self.assertEqual(status["finished"][0]["result"]["native"]["score"], 100)
            record = status["finished"][0]
            self.assertEqual(record["seed"], 1)
            self.assertEqual(Path(record["output"]).name, "seed1-standard-l0")
            self.assertEqual(record["command"][record["command"].index("--seed") + 1], "1")
            messages = [json.loads(line) for line in logs.getvalue().splitlines()]
            self.assertEqual(sum(x.get("finished", False) for x in messages), 1)


if __name__ == "__main__":
    unittest.main()
