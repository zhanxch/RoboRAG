"""A canceled robot run must not leave an account-inference subprocess alive."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request


@unittest.skipUnless(sys.platform.startswith("linux"), "uses Linux process evidence")
class TransportCleanupTests(unittest.TestCase):
    def test_sigterm_reaps_inflight_inference_child(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            child_file = folder / "child.pid"
            fake = folder / "fake-cli"
            fake.write_text(f"#!{sys.executable}\nimport os,sys,time\nfrom pathlib import Path\n"
                            "sys.stdin.read()\nPath(os.environ['TRANSPORT_TEST_PID']).write_text(str(os.getpid()))\ntime.sleep(120)\n")
            fake.chmod(0o700)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            environment = dict(os.environ, PYTHONPATH=str(root / "src"),
                               COMPARE_CODEX_BIN=str(fake), TRANSPORT_TEST_PID=str(child_file))
            with (folder / "server.log").open("w") as log:
                server = subprocess.Popen([sys.executable, "-m", "roborag.transports.responses_bridge",
                                           "--port", str(port), "--output", str(folder / "calls")],
                                          env=environment, stdout=log, stderr=log)
                child = None
                try:
                    deadline = time.monotonic() + 10
                    while True:
                        self.assertIsNone(server.poll(), "test bridge exited before request")
                        try:
                            with socket.create_connection(("127.0.0.1", port), timeout=.1):
                                break
                        except OSError:
                            self.assertLess(time.monotonic(), deadline)
                            time.sleep(.02)
                    def request():
                        body = {"instructions": "test", "input": [],
                                "tools": [{"name": "act", "parameters": {"type": "object"}}]}
                        try:
                            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/responses",
                                                         json.dumps(body).encode(), method="POST")
                            urllib.request.urlopen(req, timeout=10).close()
                        except Exception:
                            pass  # Cancellation intentionally closes this request.
                    worker = threading.Thread(target=request, daemon=True)
                    worker.start()
                    while child is None:
                        self.assertLess(time.monotonic(), deadline, "fake inference did not start")
                        try:
                            child = int(child_file.read_text())
                        except (FileNotFoundError, ValueError):
                            time.sleep(.02)
                    self.assertTrue(Path(f"/proc/{child}").exists())
                    server.send_signal(signal.SIGTERM)
                    server.wait(timeout=5)
                    worker.join(timeout=2)
                    self.assertFalse(Path(f"/proc/{child}").exists(), "orphan inference process")
                    record = json.loads((folder / "calls/call-0001/execution.json").read_text())
                    self.assertEqual(record["error"], "SystemExit")
                finally:
                    if server.poll() is None:
                        server.terminate()
                        try:
                            server.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            server.kill()
                            server.wait(timeout=5)
                    if child and Path(f"/proc/{child}").exists():
                        os.kill(child, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
