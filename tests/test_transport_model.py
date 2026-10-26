"""The requested model must reach the CLI, rather than merely label the run."""

from http.server import HTTPServer
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from roborag.transports.responses_bridge import Handler
from roborag.transports.codex_cli import normalize_usage


class TransportModelTests(unittest.TestCase):
    def test_missing_usage_is_not_reported_as_zero(self):
        value = normalize_usage({})
        self.assertIsNone(value["input_tokens"])
        self.assertIsNone(value["output_tokens"])
        self.assertIsNone(value["total_tokens"])
        self.assertIsNone(value["input_tokens_details"]["cached_tokens"])
        known = normalize_usage({"input_tokens": 10, "output_tokens": 2})
        self.assertEqual(known["total_tokens"], 12)
        self.assertIsNone(known["input_tokens_details"]["cached_tokens"])

    def test_invalid_usage_does_not_bypass_budget_or_cache_accounting(self):
        value = normalize_usage({"input_tokens": True, "output_tokens": -1, "cached_input_tokens": 0})
        self.assertIsNone(value["total_tokens"])
        known = normalize_usage({"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 11})
        self.assertIsNone(known["input_tokens_details"]["cached_tokens"])

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "optional transport runtime not installed")
    def test_response_model_and_effort_reach_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fake = folder / "fake-cli"
            fake.write_text(
                f"#!{sys.executable}\n"
                "import json,sys\nfrom pathlib import Path\n"
                "sys.stdin.read()\n"
                "Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('{}')\n"
                "print(json.dumps({'type':'turn.started'}))\n"
                "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':11,'output_tokens':3}}))\n"
            )
            fake.chmod(0o700)
            server = HTTPServer(("127.0.0.1", 0), Handler)
            server.output, server.calls = folder / "calls", 0
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with patch.dict(os.environ, {"COMPARE_CODEX_BIN": str(fake)}):
                    for number, model in enumerate(("gpt-6-sol", None), 1):
                        body = {"instructions": "test", "input": [],
                                "tools": [{"name": "act", "parameters": {"type": "object"}}],
                                "reasoning": {"effort": "xhigh"}}
                        if model is not None:
                            body["model"] = model
                        request = Request(f"http://127.0.0.1:{server.server_port}/v1/responses",
                                          json.dumps(body).encode(), method="POST")
                        with urlopen(request, timeout=10) as response:
                            self.assertEqual(json.load(response)["usage"]["total_tokens"], 14)
                        invocation = json.loads((folder / f"calls/call-{number:04d}/invocation.json").read_text())
                        expected = model or "gpt-6-astra"
                        self.assertEqual(invocation["model"], expected)
                        command = invocation["command"]
                        self.assertEqual(command[command.index("--model") + 1], expected)
                        self.assertIn('model_reasoning_effort="xhigh"', command)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
