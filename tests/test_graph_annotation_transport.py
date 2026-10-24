"""Exercise graph annotation through the real transport with a fake CLI."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("jsonschema"), "Optional transport runtime")
class GraphAnnotationTransportTests(unittest.TestCase):
    def test_fresh_annotation_passes_requested_model_to_cli(self):
        spec = importlib.util.spec_from_file_location("graph_builder", ROOT / "scripts/build_graph_bank.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake-cli"
            fake.write_text(f'#!{sys.executable}\n' + '''import json,sys
from pathlib import Path
sys.stdin.read()
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps({"title":"test","summary":"test","phases":[]}))
print(json.dumps({"type":"turn.started"}))
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":3,"output_tokens":2}}))
''')
            fake.chmod(0o700)
            with patch.dict(os.environ, {"COMPARE_CODEX_BIN":str(fake)}):
                value, usage = builder.annotate({"instruction":"fold","frames":10,"frequency":25}, [], {},
                    root / "call", "gpt-6-astra", "xhigh")
            invocation = json.loads((root / "call/invocation.json").read_text())
            command = invocation["command"]
            self.assertEqual(command[command.index("--model")+1], "gpt-6-astra")
            self.assertEqual(value["title"], "test")
            self.assertEqual(usage["total_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
