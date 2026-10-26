#!/usr/bin/env python3
"""Serve the local RoboRAG research progress app (Python standard library only)."""
from __future__ import annotations

import argparse
import csv
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import sqlite3
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from roborag.research_progress import COLLECTIONS, Conflict, Store

WEB = ROOT / "web/research-progress"


def csv_export(state):
    stream = io.StringIO(newline="")
    fields = ["id", "title", "date", "task", "method", "protocol", "condition", "model", "effort", "split", "layout", "seed",
              "transport", "mcp_server", "mcp_version", "status", "success", "completion", "wall_seconds", "retrieval_seconds",
              "execution_seconds", "input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "total_tokens", "usage", "evidence", "notes"]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for run in state["experiments"] + state["historical"]:
        # Keep arbitrary titles safe when opened by spreadsheet software.
        safe = {key: ("'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
                for key, value in run.items()}
        writer.writerow(safe)
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, store, **kwargs):
        self.store = store
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass

    def send(self, data, content_type="application/json; charset=utf-8", status=200, filename=None):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'self'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def allowed_host(self):
        return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def do_GET(self):
        if not self.allowed_host():
            return self.send({"error": "仅支持本地访问"}, status=403)
        route = urlsplit(self.path)
        try:
            if route.path == "/api/state":
                return self.send(self.store.state())
            if route.path == "/api/export":
                state = self.store.state()
                if parse_qs(route.query).get("format") == ["csv"]:
                    return self.send(csv_export(state), "text/csv; charset=utf-8", filename="roborag-experiments.csv")
                return self.send(state, filename="roborag-research-progress.json")
            if route.path == "/api/evidence":
                name = parse_qs(route.query).get("path", [""])[0]
                path = (ROOT / name).resolve()
                allowed = any(path.is_relative_to((ROOT / prefix).resolve()) for prefix in ["docs", "results", "experiments", "src/roborag", "tests"])
                if not allowed or path.suffix not in {".md", ".json", ".py", ".csv", ".txt"} or not path.is_file() or path.stat().st_size > 5_000_000:
                    return self.send({"error": "此证据仅保留路径，请在项目中查看"}, status=404)
                return self.send(path.read_bytes(), "text/plain; charset=utf-8")
            assets = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/metrics.js": "metrics.js", "/style.css": "style.css"}
            if route.path in assets:
                path = WEB / assets[route.path]
                return self.send(path.read_bytes(), (mimetypes.guess_type(path.name)[0] or "text/plain") + "; charset=utf-8")
            return self.send({"error": "未找到"}, status=404)
        except (OSError, ValueError, KeyError) as error:
            return self.send({"error": str(error)}, status=500)

    def mutate(self, method):
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        if not self.allowed_host() or (origin and origin != f"http://{host}") or self.headers.get("X-RoboRAG-Client") != "research-progress":
            return self.send({"error": "仅允许本地页面写入"}, status=403)
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.send({"error": "需要 application/json"}, status=415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 100_000:
                raise ValueError("请求大小无效")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("请求必须是对象")
            parts = urlsplit(self.path).path.strip("/").split("/")
            if len(parts) not in (2, 3) or parts[0] != "api" or parts[1] not in COLLECTIONS:
                raise KeyError("未知记录类型")
            collection = parts[1]
            record_id = parts[2] if len(parts) == 3 else None
            if method == "POST" and record_id is None:
                return self.send(self.store.save(collection, data), status=201)
            if method == "PUT" and record_id:
                return self.send(self.store.save(collection, data, record_id))
            if method == "DELETE" and record_id:
                self.store.delete(collection, record_id, data.get("revision"))
                return self.send({"deleted": record_id})
            raise KeyError("未知操作")
        except Conflict as error:
            return self.send({"error": str(error)}, status=409)
        except (ValueError, TypeError) as error:
            return self.send({"error": str(error)}, status=400)
        except KeyError as error:
            return self.send({"error": str(error)}, status=404)
        except (OSError, sqlite3.Error):
            return self.send({"error": "写入失败，请检查磁盘与目录权限；输入仍保留"}, status=500)

    def do_POST(self):
        self.mutate("POST")

    def do_PUT(self):
        self.mutate("PUT")

    def do_DELETE(self):
        self.mutate("DELETE")


def make_server(store, port=8794):
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, store=store))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument("--db", type=Path, default=ROOT / "data/research-progress.sqlite3")
    args = parser.parse_args()
    store = Store(args.db.resolve())
    server = make_server(store, args.port)
    print(f"Local: http://127.0.0.1:{server.server_port}\nData: {store.path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
