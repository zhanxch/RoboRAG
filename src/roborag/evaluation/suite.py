"""Task-independent declared suites and paired comparisons; never select best retries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roborag.evaluation.comparison import aggregate, candidate_rows
from roborag.io import read_json, write_json
from roborag.tasks import TASKS


def case_key(row):
    return row["seed"], row["variant"], row["layout"]


def suite(rows, cases):
    planned = {case_key(c) for c in cases}
    if len(planned) != len(cases) or not cases:
        raise ValueError("Declare a nonempty, unique case list")
    keys = [case_key(r) for r in rows]
    if len(set(keys)) != len(keys) or not set(keys) <= planned:
        raise ValueError("Duplicate or unplanned results; keep every revision separate")
    by_variant = {
        variant: aggregate(
            [r for r in rows if r["variant"] == variant],
            sum(c["variant"] == variant for c in cases),
        )
        for variant in sorted({c["variant"] for c in cases})
    }
    return dict(
        overall=aggregate(rows, len(cases)),
        by_variant=by_variant,
        pending=[c for c in cases if case_key(c) not in set(keys)],
        cases=rows,
    )


def paired(baseline, candidate):
    before = {case_key(r): r for r in baseline}
    if len(before) != len(baseline) or len({case_key(r) for r in candidate}) != len(candidate):
        raise ValueError("Duplicate cases in paired comparison")
    pairs = []
    for after in candidate:
        key = case_key(after)
        if key not in before:
            raise ValueError(f"Missing matched baseline: {key}")
        first = before[key]
        for field in ("layout_sha256", "model", "effort"):
            if not first.get(field) or first[field] != after.get(field):
                raise ValueError(f"Paired {field} differs: {key}")
        for field in ("controller", "no_rag"):
            if first["protocol"][field] != after["protocol"][field]:
                raise ValueError(f"Paired {field} differs: {key}")
        pairs.append(
            dict(seed=key[0], variant=key[1], layout=key[2], baseline=first, candidate=after)
        )
    joint = [
        p
        for p in pairs
        if p["baseline"]["complete"]
        and p["candidate"]["complete"]
        and p["baseline"]["success"]
        and p["candidate"]["success"]
    ]
    return dict(
        baseline=aggregate([p["baseline"] for p in pairs]),
        candidate=aggregate([p["candidate"] for p in pairs]),
        pairs=pairs,
        jointly_successful=dict(
            count=len(joint),
            baseline=aggregate([p["baseline"] for p in joint]),
            candidate=aggregate([p["candidate"] for p in joint]),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), default="push_T")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = candidate_rows(args.runs, task=args.task)
    report = dict(
        schema="roborag.task-evaluation.v1",
        task=args.task,
        **suite(rows, read_json(args.cases)),
        interpretation=[
            "Native evaluation is authoritative; incomplete runs retain cost and unknown final score.",
            "Frozen revisions are separate; development cases do not establish heldout performance.",
            "System perception, prompts and references can all change; this is not a RAG-only effect.",
            "Input includes cached tokens; output includes reasoning. Billing/API request count are unknown.",
        ],
    )
    if args.baseline:
        report["paired"] = paired(candidate_rows(args.baseline, task=args.task), rows)
    write_json(args.output, report)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
