"""HTTP browser for recorded graph evidence."""

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from roborag.bank.store import BankStore, frame_bytes
from roborag.config import PROJECT_ROOT as ROOT
from roborag.config import RESOURCE_ROOT

WEB = RESOURCE_ROOT / "graph-bank"


def phase_trace(source, phase):
    import h5py

    start, end = phase["start_frame"], phase["end_frame"]
    with h5py.File(source, "r") as data:
        result = {
            "start_frame": start,
            "end_frame": end,
            "frequency": float(data["additional_info/frequency"][()]),
            "frames": list(range(start, end)),
            "state": {},
            "action": {},
        }
        for arm in ("left", "right"):
            for section in ("state", "action"):
                for field in ("ee_poses", "ee_joint_states", "arm_joint_states"):
                    key = f"{arm}_{field}"
                    result[section][key] = data[f"{section}/{key}"][start:end].tolist()
    return result


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_bytes(self, data, content_type, *, status=200, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, value, **kwargs):
            return self.send_bytes(
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                "application/json; charset=utf-8",
                **kwargs,
            )

        def do_POST(self):
            if urlparse(self.path).path != "/api/retrieve":
                return self.send_json({"error": "Unknown endpoint"}, status=404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1000000:
                    raise ValueError("Expected a JSON request smaller than 1 MB")
                request = json.loads(self.rfile.read(length))
                return self.send_json(store.retriever().retrieve(request))
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                return self.send_json({"error": str(error)}, status=400)
            except (KeyError, FileNotFoundError) as error:
                return self.send_json({"error": str(error)}, status=404)

        def do_GET(self):
            route = urlparse(self.path)
            params = parse_qs(route.query)

            def get(key, default=""):
                return params.get(key, [default])[0]

            try:
                if route.path == "/api/bank":
                    return self.send_json(store.bank())
                if route.path == "/api/retrieval-info":
                    index = store.retriever().index
                    return self.send_json(
                        {
                            k: index[k]
                            for k in (
                                "schema",
                                "built_at",
                                "method",
                                "counts",
                                "horizon",
                                "stride",
                                "limitations",
                            )
                        }
                    )
                if route.path == "/api/chunk":
                    return self.send_json(
                        store.retriever().chunk(get("id")),
                        filename="action-chunk.json" if get("download") else None,
                    )
                if route.path == "/api/query-example":
                    return self.send_json(store.retriever().example_query(get("id"), get("node")))
                if route.path == "/api/search":
                    query = get("q")[:500]
                    return self.send_json(
                        {
                            "query": query,
                            "method": "BM25",
                            "results": store.search(
                                query, get("task") or None, min(50, max(1, int(get("limit", "20"))))
                            ),
                        }
                    )
                if route.path in ("/api/graph", "/api/export", "/api/frame", "/api/trace"):
                    graph = store.graph(get("id"))
                    if route.path == "/api/graph":
                        return self.send_json(graph)
                    if route.path == "/api/export":
                        return self.send_json(graph, filename=f"{graph['id']}.json")
                    if route.path == "/api/frame":
                        frame = int(get("frame", "0"))
                        camera = get("camera", "cam_head")
                        if not (
                            0 <= frame < graph["source"]["frames"]
                            and camera in graph["source"]["cameras"]
                        ):
                            raise ValueError("Invalid frame or camera")
                        raw = frame_bytes(str(store.source(graph)), camera, frame)
                        mime = "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg"
                        return self.send_bytes(raw, mime)
                    phase = next(
                        (
                            n
                            for n in graph["nodes"]
                            if n["id"] == get("node") and n["kind"] == "phase"
                        ),
                        None,
                    )
                    if phase is None:
                        raise KeyError("Phase not found")
                    trace = phase_trace(store.source(graph), phase)
                    return self.send_json(
                        trace,
                        filename=f"{graph['id']}-{phase['id']}-actions.json"
                        if get("download")
                        else None,
                    )
                if route.path.startswith("/api/"):
                    raise KeyError("Unknown endpoint")
                relative = route.path.lstrip("/") or "index.html"
                path = (WEB / relative).resolve()
                if not path.is_relative_to(WEB) or not path.is_file():
                    raise KeyError("File not found")
                return self.send_bytes(
                    path.read_bytes(),
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                )
            except (ValueError, TypeError) as error:
                return self.send_json({"error": str(error)}, status=400)
            except (KeyError, FileNotFoundError) as error:
                return self.send_json({"error": str(error)}, status=404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=ROOT / "data/graph-bank/current")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(BankStore(args.bank, ROOT / "data"))
    )
    print(f"Local: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
