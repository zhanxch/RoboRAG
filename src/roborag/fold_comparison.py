"""Matched-layout comparison with the archived local HexaShell reproduction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from .episode_files import read_json, write_json
from .rag_accounting import summarize


def average(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.mean(values) if values else None


def aggregate(rows, planned=None):
    planned = len(rows) if planned is None else planned
    if planned < len(rows):
        raise ValueError("Planned case count is smaller than recorded cases")
    complete = [r for r in rows if r["complete"]]
    usage = [r.get("usage", {}) for r in rows]
    totals = {}
    for field in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
        known = [u[field] for u in usage if isinstance(u.get(field), int)]
        totals[field] = sum(known) if len(known) == len(rows) and rows else None
        totals[field + "_known_lower_bound"] = sum(known)
    incoming, cached = totals["input_tokens"], totals["cached_input_tokens"]
    successes = sum(r["success"] for r in complete)
    wall = [r["wall_seconds"] for r in rows if r.get("wall_seconds") is not None]
    return dict(planned=planned, recorded=len(rows), scored=len(complete), successes=successes,
        success_rate=successes/planned if planned and len(complete) == planned else None,
        scored_success_rate=successes/len(complete) if complete else None,
        mean_progress=average(complete, "progress"), mean_control_steps=average(complete, "steps"),
        mean_successful_control_steps=average([r for r in complete if r["success"]], "steps"),
        mean_wall_seconds=average(rows, "wall_seconds"), mean_agent_wall_seconds=average(rows, "agent_wall_seconds"),
        sum_episode_wall_seconds=sum(wall) if rows and len(wall) == len(rows) else None,
        sum_episode_wall_seconds_known_lower_bound=sum(wall),
        usage=totals, uncached_input_tokens=incoming-cached if incoming is not None and cached is not None else None,
        input_cache_fraction=cached/incoming if incoming and cached is not None else None,
        actual_billed_usd=None, model_request_count=None)


def baseline_rows(document):
    rows = []
    for r in document["cases"]:
        path = Path(r["native_result"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["native_sha256"]:
            raise ValueError(f"Baseline native result changed: {path}")
        native = read_json(path)
        if native.get("eval_time") != 1 or native["score"] != r["progress"]:
            raise ValueError(f"Baseline score mismatch: {path}")
        rows.append(dict(r, complete=True, usage=r["accounting"]["reported_usage"]))
    return rows


def candidate_rows(paths):
    rows = []
    for path in paths:
        path = Path(path)
        runs = [path] if (path / "run.json").exists() else sorted(p for p in path.iterdir() if p.is_dir() and (p / "run.json").exists())
        for run in runs:
            r = summarize(run)
            if r["task"] not in ("fold_clothes", "fold_clothes_random"):
                raise ValueError(f"Not a fold_clothes episode: {run}")
            result = read_json(run / "result.json", {})
            config = read_json(run / "run.json")
            if r["complete"]:
                native = Path(result["native_result"])
                if hashlib.sha256(native.read_bytes()).hexdigest() != result["native_sha256"]:
                    raise ValueError(f"Candidate native result changed: {run}")
                if read_json(native) != r["native"]:
                    raise ValueError(f"Candidate embedded result differs from native evidence: {run}")
            rows.append(dict(variant=r["variant"], layout=r["layout"], seed=r["seed"], complete=r["complete"],
                success=r["native"]["success_rate"] == 1 if r["complete"] else None,
                progress=r["native"]["score"] if r["complete"] else None,
                steps=r["native_control_steps"], steps_complete=r["native_control_steps_complete"],
                wall_seconds=r["wall_seconds"], agent_wall_seconds=r["agent_wall_seconds"],
                usage=r["reported_usage"], model=r["model"], effort=r["effort"], directory=r["run"],
                layout_sha256=config["layout_sha256"],
                tool_counts=r["tool_counts"], failure=r["failure"],
                protocol=dict(model=r["model"], effort=r["effort"], controller=config["args"]["controller"],
                    no_rag=config["args"]["no_rag"], prompt=config["prompt_sha256"],
                    sources=config["source_sha256"])))
    keys = [(r["seed"], r["variant"], r["layout"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate candidate cases; compare each frozen revision separately")
    if len({json.dumps(r["protocol"], sort_keys=True) for r in rows}) > 1:
        raise ValueError("Mixed candidate protocols; compare each frozen revision separately")
    return rows


def compare(reference, candidates, planned):
    all_baseline = baseline_rows(reference)
    keys = {(r["seed"], r["variant"], r["layout"]) for r in candidates}
    matched = [r for r in all_baseline if (r["seed"], r["variant"], r["layout"]) in keys]
    if len(matched) != len(candidates):
        raise ValueError("A candidate has no matched local HexaShell case")
    pairs = []
    for candidate in candidates:
        baseline = next(r for r in matched if (r["seed"],r["variant"],r["layout"]) ==
                        (candidate["seed"],candidate["variant"],candidate["layout"]))
        if candidate["model"] != baseline["model"] or candidate["effort"] != reference["reasoning_effort"]:
            raise ValueError("Matched comparison requires the same model and reasoning effort")
        if candidate["layout_sha256"] != baseline["layout_sha256"]:
            raise ValueError("Matching layout IDs have different scene contents")
        pairs.append(dict(case=f"{candidate['variant']}-l{candidate['layout']}", baseline=baseline, candidate=candidate))
    jointly = [p for p in pairs if p["candidate"]["success"] and p["baseline"]["success"]]
    return dict(schema="roborag.fold-comparison.v1", baseline_source=reference["source"],
        baseline_source_sha256=reference["source_sha256"],
        baseline=aggregate(matched), candidate=aggregate(candidates, planned), pairs=pairs,
        jointly_successful=dict(cases=[p["case"] for p in jointly],
            baseline=aggregate([p["baseline"] for p in jointly]), candidate=aggregate([p["candidate"] for p in jointly])),
        interpretation=["Matched public seed/layout/variant; one trial per case. No statistical significance claim.",
            "Compare the same model and reasoning effort; different control representations remain a method difference.",
            "Tool calls/chunks are not model API requests. Billing is unknown; cache tokens are part of input.",
            "Sum of episode wall times is not concurrent batch elapsed time.",
            "Development pilots and prior revisions remain separate from each frozen suite."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--planned", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(read_json(args.reference), candidate_rows(args.runs), args.planned)
    write_json(args.output, report)
    print(json.dumps({k:report[k] for k in ("baseline", "candidate")}, indent=2))


if __name__ == "__main__":
    main()
