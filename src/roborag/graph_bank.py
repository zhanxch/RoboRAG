"""A small, inspectable bank of episode graphs and phase retrieval documents.

The backbone follows recorded phases. Cue/action/feedback children form a local
guidance tree; labels remain free text rather than a fixed phase ontology.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

TASK_NAMES = {
    "stack_blocks": "积木堆叠",
    "fold_clothes": "叠衣",
    "cover_blocks": "覆盖积木",
    "put_bottles_into_dustbin": "瓶子收纳",
    "deposit_coin": "硬币投放",
}

EXPERIENCE_SEMANTICS = {
    "role": "recorded_experience",
    "next_edges": "observed temporal order, not mandatory planning dependencies",
    "guidance": "automatically extracted descriptions of a demonstration, not executable instructions",
    "planner": "GPT generates a new plan from the current goal and observations; this graph is reference evidence",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def tokens(text):
    """English words and Chinese unigrams/bigrams for a dependency-free index."""
    text = text.lower()
    result = re.findall(r"[a-z0-9]+", text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        result.extend(run)
        result.extend(run[i:i + 2] for i in range(len(run) - 1))
    return result


def make_graph(meta, annotation, model):
    phases = annotation["phases"]
    if not phases:
        raise ValueError("Annotation has no phases")
    nodes, edges = [], []
    previous = None
    previous_end = 0
    for index, phase in enumerate(phases):
        start, end = phase["start_frame"], phase["end_frame"]
        if not (isinstance(start, int) and isinstance(end, int) and
                0 <= start < end <= meta["frames"] and start >= previous_end):
            raise ValueError(f"Invalid phase interval: {start}:{end}")
        previous_end = end
        node_id = f"p{index + 1:02d}"
        nodes.append({
            "id": node_id, "kind": "phase", "title": phase["title"],
            "summary": phase["summary"], "objects": phase["objects"],
            "start_frame": start, "end_frame": end, "order": index,
            "keyframes": sorted(set([start, (start + end - 1) // 2, end - 1])),
        })
        if previous:
            edges.append({"source": previous, "target": node_id, "kind": "next", "label": "下一阶段"})
        for kind, title, field in [("cue", "场景线索", "cue"),
                                   ("action", "操作", "action"),
                                   ("feedback", "反馈", "feedback")]:
            child_id = f"{node_id}-{kind}"
            nodes.append({"id": child_id, "kind": kind, "title": title,
                          "text": phase[field], "parent": node_id})
            edges.append({"source": node_id, "target": child_id, "kind": "detail", "label": title})
        previous = node_id
    return {
        "schema": "roborag.graph.v1", "id": meta["id"], "task": meta["task"],
        "semantics": dict(EXPERIENCE_SEMANTICS),
        "task_name": TASK_NAMES.get(meta["task"], meta["task"]),
        "title": annotation["title"], "summary": annotation["summary"],
        "instruction": meta["instruction"], "episode": meta["episode"],
        "source": {k: meta[k] for k in ["path", "sha256", "frames", "frequency", "cameras"]},
        "construction": {"method": "codex_multimodal", "model": model,
                         "annotation": "自动提取", "frame_interval": "[start_frame, end_frame)",
                         "created_at": datetime.now(timezone.utc).isoformat()},
        "nodes": nodes, "edges": edges,
    }


def graph_documents(graph):
    children = {}
    for node in graph["nodes"]:
        if node["kind"] != "phase":
            children.setdefault(node["parent"], []).append(node)
    docs = []
    for phase in (n for n in graph["nodes"] if n["kind"] == "phase"):
        guidance = children.get(phase["id"], [])
        body = "\n".join([graph["task_name"], graph["instruction"], phase["title"],
                          phase["summary"], " ".join(phase["objects"]),
                          *[n["text"] for n in guidance]])
        docs.append({"id": f"{graph['id']}:{phase['id']}", "graph_id": graph["id"],
                     "node_id": phase["id"], "task": graph["task"],
                     "task_name": graph["task_name"], "graph_title": graph["title"],
                     "title": phase["title"], "summary": phase["summary"],
                     "start_frame": phase["start_frame"], "text": body,
                     "terms": dict(Counter(tokens(body)))})
    return docs


def publish_bank(root):
    root = Path(root)
    graphs = [json.loads(p.read_text()) for p in sorted((root / "graphs").glob("*.json"))]
    summaries, docs = [], []
    for graph in graphs:
        phases = [n for n in graph["nodes"] if n["kind"] == "phase"]
        summaries.append({k: graph[k] for k in ["id", "task", "task_name", "title", "summary", "episode", "instruction"]} |
                         {"phase_count": len(phases), "node_count": len(graph["nodes"]),
                          "edge_count": len(graph["edges"]), "frames": graph["source"]["frames"],
                          "frequency": graph["source"]["frequency"], "preview_frame": 0})
        docs.extend(graph_documents(graph))
    write_json(root / "search-index.json", {"method": "BM25 · 中文字词 / English tokens", "documents": docs})
    bank = {"schema": "roborag.graph-bank.v1", "name": "RoboRAG · 交互经验图库",
            "updated_at": datetime.now(timezone.utc).isoformat(), "graphs": summaries,
            "counts": {"graphs": len(graphs), "phases": len(docs),
                       "nodes": sum(len(g["nodes"]) for g in graphs),
                       "edges": sum(len(g["edges"]) for g in graphs),
                       "tasks": len({g["task"] for g in graphs})}}
    write_json(root / "bank.json", bank)
    return bank


def search_documents(documents, query, *, task=None, limit=12):
    queries = set(tokens(query))
    if not queries:
        return []
    # Use the whole bank for document frequencies, then apply the task filter.
    count = len(documents)
    if not count:
        return []
    average = sum(sum(d["terms"].values()) for d in documents) / count or 1
    df = Counter(term for d in documents for term in queries if term in d["terms"])
    results = []
    for doc in documents:
        if task and doc["task"] != task:
            continue
        length = sum(doc["terms"].values())
        score = 0.0
        matched = []
        for term in queries:
            frequency = doc["terms"].get(term, 0)
            if not frequency:
                continue
            matched.append(term)
            inverse = math.log(1 + (count - df[term] + 0.5) / (df[term] + 0.5))
            score += inverse * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average))
        if score:
            results.append({k: v for k, v in doc.items() if k != "terms"} |
                           {"score": round(score, 4), "matched_terms": sorted(matched)})
    return sorted(results, key=lambda item: (-item["score"], item["id"]))[:limit]
