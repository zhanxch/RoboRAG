#!/usr/bin/env python3
"""Archive published RoboICL scores without mixing full and partial rollouts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import urllib.request

URL = ("https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/"
       "assets/data/benchmark-rollouts.json?v=20260928-rollouts-v9")
PROJECT = "https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/"
ALIASES = {"swap_t": "swap_T", "play_xylophone": "play_Xylophone"}


def normalize(raw):
    tasks = []
    for task in raw["tasks"]:
        layouts = []
        for row in task["layouts"]:
            layouts.append({
                "layout": row["id"], "variant": row["variant"], "seed": 0,
                "score_0_100": row["score"], "success": row["success"],
                "official_score_0_100": row["officialScore"],
                "official_success": row["officialSuccess"],
                "included_in_official_evaluation": row["includedInAggregate"],
                "trace_url": PROJECT + row["trace"],
            })
        full = [r for r in layouts if r["included_in_official_evaluation"]]
        if len(full) != task["officialLayoutCount"]:
            raise ValueError(f"Official layout count mismatch: {task['id']}")
        if full and abs(statistics.mean(r["official_score_0_100"] for r in full)
                        - task["officialScore"]) > .0051:
            raise ValueError(f"Official score mismatch: {task['id']}")
        identities = {(r["seed"], r["variant"], r["layout"]) for r in layouts}
        if len(identities) != len(layouts):
            raise ValueError(f"Duplicate layout identity: {task['id']}")
        tasks.append({
            "task": ALIASES.get(task["id"], task["id"]), "website_id": task["id"],
            "name": task["name"], "category": task["category"], "shots": task["shots"],
            "published_score_0_100": task["officialScore"],
            "official_episode_count": len(full),
            "evaluation_status": task["evaluationStatus"],
            "completed_success_rate": statistics.mean(r["official_success"] for r in full) if full else None,
            "layouts": layouts,
        })
    full_count = sum(t["official_episode_count"] for t in tasks)
    partial_count = sum(len(t["layouts"]) - t["official_episode_count"] for t in tasks)
    summary = raw["summary"]
    assert full_count == summary["fullEvaluationRolloutCount"]
    assert partial_count == summary["partialEvaluationRolloutCount"]
    mean = statistics.mean(t["published_score_0_100"] for t in tasks)
    assert abs(mean - summary["overallScore"]) <= .0051
    return {
        "schema": "roborag.roboicl-public-baseline.v1",
        "source_url": URL, "paper_url": "https://arxiv.org/html/2609.34261v1",
        "snapshot_date": raw["snapshotDate"], "source_version": raw["version"],
        "metric": "native progress score 0-100; not task success percentage",
        "task_count": len(tasks), "full_episode_count": full_count,
        "partial_episode_count": partial_count,
        "published_overall_0_100": summary["overallScore"],
        "recomputed_task_macro_mean_0_100": mean,
        "aggregation_note": (
            "Published Overall averages all 30 published task scores, including seven "
            "zero placeholders for tasks with no full evaluation. Partial rollout "
            "scores are retained for audit but excluded from same-layout comparisons."
        ),
        "tasks": tasks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.snapshot.exists():
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    payload = args.snapshot.read_bytes()
    result = normalize(json.loads(payload))
    result.update(source_sha256=hashlib.sha256(payload).hexdigest(),
                  imported_at_utc=datetime.now(timezone.utc).isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
