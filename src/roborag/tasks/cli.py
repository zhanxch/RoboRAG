"""Inspect task configuration and separately sourced evaluation evidence."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from roborag.config import PROJECT_ROOT
from roborag.io import read_json
from roborag.tasks.registry import TASK_ROOT, TASK_SPECS


def catalog(root=PROJECT_ROOT):
    source = Path(root) / "results/current.json"
    evidence = read_json(source, {}).get("tasks", {})
    rows = []
    for spec in TASK_SPECS.values():
        row = asdict(spec)
        row["directory"] = str(TASK_ROOT / spec.name)
        recorded = evidence.get(spec.name)
        if recorded:
            metrics = recorded["metrics"]
            row["evaluation"] = dict(
                status="selected_policy_report",
                source=str(source),
                policy_commit=recorded["source_commit"],
                successes=metrics["successes"],
                scored=metrics["scored"],
                planned=metrics["planned"],
            )
        else:
            row["evaluation"] = dict(status="no_selected_suite_report")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_SPECS))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = [row for row in catalog() if args.task is None or row["name"] == args.task]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    for row in rows:
        result = row["evaluation"]
        status = "registered; no selected native suite report"
        if result["status"] == "selected_policy_report":
            status = (
                f"recorded policy {result['policy_commit'][:7]}: "
                f"{result['successes']}/{result['planned']} successes, "
                f"{result['scored']}/{result['planned']} scored; see {result['source']}"
            )
        print(f"{row['name']} [{', '.join(row['capabilities']) or 'core tools'}]: {status}")
