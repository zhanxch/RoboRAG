"""Version-aware bank loading and source image access, independent of HTTP."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from roborag.bank.models import search_documents
from roborag.bank.retrieval import GraphRetriever


@lru_cache(maxsize=64)
def _read_version(path, modified, size):
    return json.loads(Path(path).read_text())


def read_json(path):
    stat = path.stat()
    return _read_version(str(path), stat.st_mtime_ns, stat.st_size)


class BankStore:
    def __init__(self, directory, data_root):
        self.directory = Path(directory).resolve()
        self.data_root = Path(data_root).resolve()
        self._retriever_version = None
        self._retriever = None

    def bank(self):
        path = self.directory / "bank.json"
        if not path.exists():
            return {
                "name": "RoboRAG · 交互经验图库",
                "graphs": [],
                "counts": {"graphs": 0, "nodes": 0, "edges": 0, "tasks": 0, "phases": 0},
            }
        return read_json(path)

    def graph(self, graph_id):
        if graph_id not in {g["id"] for g in self.bank()["graphs"]}:
            raise KeyError("Graph not found")
        return read_json(self.directory / "graphs" / f"{graph_id}.json")

    def source(self, graph):
        path = (self.data_root / graph["source"]["path"]).resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError("Invalid source path")
        return path

    def search(self, query, task, limit):
        path = self.directory / "search-index.json"
        docs = read_json(path)["documents"] if path.exists() else []
        return search_documents(docs, query, task=task, limit=limit)

    def retriever(self):
        path = self.directory / "execution-index.json"
        stat = path.stat()
        version = (stat.st_mtime_ns, stat.st_size)
        if version != self._retriever_version:
            self._retriever = GraphRetriever(read_json(path), self.graph, self.source)
            self._retriever_version = version
        return self._retriever


@lru_cache(maxsize=256)
def frame_bytes(source, camera, frame):
    import h5py

    with h5py.File(source, "r") as data:
        raw = bytes(data[f"vision/{camera}/colors"][frame])
    return raw
