"""Loopback Responses transport; preserves upstream RoboICL action/history logic."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import argparse
import json
from pathlib import Path
import signal
import traceback
from .codex_cli import infer, write_json


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/responses":
            self.send_error(404)
            return
        self.server.calls += 1
        folder = self.server.output / f"call-{self.server.calls:04d}"
        try:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            tools = payload.get("tools", [])
            if len(tools) != 1 or tools[0].get("name") != "act":
                raise ValueError("Expected upstream's single act function")
            # The model sees exact upstream instructions, role history and RGB.
            model_input = {key: payload[key] for key in ("instructions", "tools", "input")}
            action, usage = infer(model_input, tools[0]["parameters"], folder,
                effort=payload.get("reasoning", {}).get("effort", "xhigh"))
            response = {"id": f"codex_local_{self.server.calls}", "status": "completed",
                "output": [{"type": "function_call", "name": "act",
                            "call_id": f"act_{self.server.calls}",
                            "arguments": json.dumps(action)}], "usage": usage}
            write_json(folder / "response.json", response)
            body = json.dumps(response).encode()
            self.send_response(200)
        except Exception as error:
            print(traceback.format_exc(), flush=True)
            body = json.dumps({"error": {"code": "local_transport_error", "message": str(error)}}).encode()
            self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    def interrupted(signum, frame):
        # Unwind infer() so its child process group is terminated and reaped.
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    server = HTTPServer(("127.0.0.1", a.port), Handler)
    server.output = a.output.resolve()
    server.calls = 0
    print(f"READY 127.0.0.1:{a.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
