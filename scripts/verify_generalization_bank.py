#!/usr/bin/env python3
"""Check full TRAIN access before native testing, without model inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from roborag.graph_server import BankStore, frame_bytes
from roborag.graph_chunks import select_reference_window


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(task, bank):
    import h5py
    import numpy as np
    from PIL import Image
    store = BankStore(bank, ROOT / "data")
    retriever = store.retriever()
    graphs = [g for g in retriever.index["graphs"] if g["task"] == task]
    assert len(graphs) == (6 if task == "stack_blocks" else 3)
    report = dict(task=task, bank=str(bank.resolve()), passed=False,
                  generated_at=datetime.now(timezone.utc).isoformat(),
                  metadata_sha256={str(p.resolve()):digest(p) for p in
                      [*bank.glob("*.json"), *(bank / "graphs").glob("*.json")]},
                  source_sha256={}, demonstrations=[], phases=0, sections=0, images=0)
    for entry in graphs:
        graph = store.graph(entry["id"])
        source = store.source(graph)
        sha = digest(source)
        assert sha == graph["source"]["sha256"]
        report["source_sha256"][str(source)] = sha
        phases = [p for p in retriever.index["phases"] if p["graph_id"] == graph["id"]]
        if task != "stack_blocks":
            assert phases[0]["start"] == 0 and phases[-1]["end"] == graph["source"]["frames"]
            assert all(a["end"] == b["start"] for a, b in zip(phases, phases[1:]))
        with h5py.File(source, "r") as data:
            frames = len(data["state/left_ee_poses"])
            for section in ("state", "action"):
                for arm in ("left", "right"):
                    for field in ("ee_poses", "ee_joint_states"):
                        values = np.asarray(data[f"{section}/{arm}_{field}"])
                        assert len(values) == frames and np.isfinite(values).all()
            for camera in graph["source"]["cameras"]:
                assert len(data[f"vision/{camera}/colors"]) == frames
        # Every phase has readable command-event and boundary references,
        # including original 14D rows and all retained camera images.
        for phase in phases:
            windows = retriever.by_phase[(graph["id"], phase["node_id"])]
            assert windows
            for selector in ("start", "close", "open", "end"):
                window, selection = select_reference_window(source, windows, selector)
                chunk = retriever.chunk(window["id"])
                assert chunk["source_sha256"] == sha
                assert len(chunk["actions"]) == window["end"]-window["start"]
                assert all(len(row) == 14 for row in chunk["actions"])
                for camera in graph["source"]["cameras"]:
                    for frame in (window["start"], window["end"]):
                        Image.open(io.BytesIO(frame_bytes(str(source), camera, frame))).verify()
                        report["images"] += 1
                report["sections"] += 1
            report["phases"] += 1
        result = retriever.retrieve(dict(task=task, mode="plan", goal=graph["instruction"],
            observation=entry["initial_cue"], k=2, max_context_chars=6000))
        assert json.loads(result["model_context"])["memories"]
        report["demonstrations"].append(dict(id=graph["id"], frames=frames, phases=len(phases)))
    report["passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.task, args.bank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k:report[k] for k in ("task", "passed", "phases", "sections", "images")}))


if __name__ == "__main__":
    main()
