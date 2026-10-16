#!/usr/bin/env python3
"""Export compact, versioned native evidence; never derive scores from intentions."""

import argparse
import hashlib
import json
from pathlib import Path

from summarize_runs import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trials = []
    for folder in args.runs:
        summary = summarize(folder)
        if summary is None:
            raise ValueError(f"No run manifest: {folder}")
        run = json.loads((folder / "run.json").read_text())
        retrieval = next(folder.glob("policy/*/retrieval.jsonl"), None)
        first = json.loads(retrieval.open().readline()) if retrieval else {}
        native_path = Path(summary["native_result"]) if summary["native_result"] else None
        hashes = run.get("source_sha256", run.get("policy_source_sha256_observed_during_run", {}))
        trials.append({
            "summary": {k: v for k, v in summary.items() if k != "latest_intent_not_verified_outcome"},
            "model": run["model"], "effort": run["arguments"]["effort"],
            "long_k": run["arguments"]["long_k"], "short_k": run["arguments"]["short_k"],
            "selector": first.get("retrieval_version", "transition-v1" if first else None),
            "corpus_sha256": first.get("corpus_sha256"),
            "selected_source_sha256": first.get("source_episode_sha256"),
            "layout_sha256": run["layout_sha256"], "roborag_commit": run.get("roborag_commit"),
            "source_sha256": hashes, "source_hash_note": run.get("source_hash_note", "Recorded by launcher"),
            "upstream_dependencies": run["dependencies"], "roboicl_commit": run["roboicl_commit"],
            "transport": run["transport"], "asset_audit": run["asset_audit"],
            "native_result": json.loads(native_path.read_text()) if native_path else None,
            "native_result_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest() if native_path else None,
            "videos": [str(p) for p in folder.rglob("*.mp4")],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "Small pilot; development and validation must be reported separately",
                                      "trials": trials}, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
