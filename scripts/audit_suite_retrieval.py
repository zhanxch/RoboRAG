#!/usr/bin/env python3
"""Audit recorded selection boundaries and measured model usage, without scoring intent."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def audit_run(folder, task, plan):
    corpus = json.loads((ROOT / task["corpus"]).read_text())
    digest = hashlib.sha256((ROOT / task["corpus"]).read_bytes()).hexdigest()
    starts = {r["start"] for r in corpus["records"]}
    violations, selections, seen, old_retrievals = [], [], [], 0
    for path in folder.glob("policy/*/retrieval.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            selections.append(row)
            checks = {
                "frozen_corpus": row["corpus_sha256"] == digest,
                "locked_training_episode": row["source_episode_sha256"] == task["source_sha256"],
                "long_budget": len(row["long_selected"]) <= plan["long_k"],
                "short_budget": len(row["short_selected"]) <= plan["short_k"],
                "long_membership": set(row["long_selected"]) <= starts,
                "distinct_long": len(set(row["long_selected"])) == len(row["long_selected"]),
                "distinct_short": len(set(row["short_selected"])) == len(row["short_selected"]),
                "chronological_short": sorted(row["short_selected"]) == row["short_selected"],
                "past_short_starts": all(s < row["step"] for s in row["short_selected"]),
            }
            latest = row["latest_interval"]
            if latest:
                checks["latest_feedback_retained"] = latest[0] in row["short_selected"]
                checks["latest_transition_completed"] = latest[0] < latest[1] <= row["step"]
                if latest[0] not in seen:
                    seen.append(latest[0])
            if len(seen) == row["short_pool_size"] and len(seen) > plan["short_k"]:
                old_retrievals += bool(set(row["short_selected"]) - set(seen[-plan["short_k"]:]))
            violations.extend({"step": row["step"], "check": name}
                              for name, passed in checks.items() if not passed)
    calls = []
    for path in sorted(folder.glob("decisions/call-*")):
        response_path, execution_path = path / "response.json", path / "execution.json"
        if not response_path.exists() or not execution_path.exists():
            continue
        response = json.loads(response_path.read_text())
        if response.get("status") != "completed":
            continue
        execution = json.loads(execution_path.read_text())
        invocation = json.loads((path / "invocation.json").read_text())
        usage = response.get("usage", {})
        calls.append({"call": path.name, "images": len(invocation["images"]),
                      "wall_seconds": execution["wall_seconds"],
                      "input_tokens": usage.get("input_tokens"),
                      "output_tokens": usage.get("output_tokens"),
                      "total_tokens": usage.get("total_tokens"),
                      "cached_input_tokens": usage.get("input_tokens_details", {}).get("cached_tokens")})
    for call in calls:
        if call["images"] > 50:
            violations.append({"call": call["call"], "check": "actual_image_cap"})
    inputs = sum(c["input_tokens"] or 0 for c in calls)
    cached = sum(c["cached_input_tokens"] or 0 for c in calls)
    return {
        "run": folder.name, "logged_selection_queries": len(selections),
        "distinct_long_term_selections": len({tuple(r["long_selected"]) for r in selections}),
        "queries_retrieving_beyond_latest_k": old_retrievals,
        "boundary_violations": violations, "completed_model_calls": len(calls),
        "attempted_call_directories": len(list(folder.glob("decisions/call-*"))),
        "known_total_tokens": sum(c["total_tokens"] or 0 for c in calls),
        "weighted_input_cache_hit_fraction": cached / inputs if inputs else None,
        "mean_completed_call_seconds": statistics.mean(c["wall_seconds"] for c in calls) if calls else None,
        "max_supplied_images": max((c["images"] for c in calls), default=None),
        "calls": calls,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    rows = []
    for task in plan["tasks"]:
        for layout in plan["layouts"]:
            folder = ROOT / plan["output_root"] / f"{task['task']}-l{layout}"
            if (folder / "run.json").exists():
                rows.append(audit_run(folder, task, plan))
    result = {
        "scope": "Recorded retrieval boundaries and measured usage, not task success or causal retrieval quality",
        "short_history_check": "Past selected starts and completed latest transition; unit tests separately cover full temporal isolation",
        "cache_note": "Provider-reported cached input divided by input tokens, weighted over completed responses",
        "runs": rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "calls"} for r in rows], indent=2))


if __name__ == "__main__":
    main()
