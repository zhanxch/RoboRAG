#!/usr/bin/env python3
"""Audit the authorized supplemental fold attempts; fill only missing scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from roborag.rag_accounting import control_accounting
from roborag.episode_files import read_jsonl


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key(case):
    return case["seed"], case["variant"], case["layout"]


def counts(cases):
    n = len(cases)
    successes = sum(c["success"] is True for c in cases)
    failures = sum(c["success"] is False for c in cases)
    return dict(attempted=n, success=successes, failure=failures, unscored=n-successes-failures,
                scored=successes+failures,
                success_rate_scored=successes/(successes+failures) if successes+failures else None,
                confirmed_success_fraction_all_attempts=successes/n if n else None)


def metrics(cases):
    fields = {"steps": [], "wall_seconds": [], "input_tokens": [], "cached_input_tokens": [],
              "output_tokens": [], "reasoning_output_tokens": [], "total_tokens": [],
              "uncached_input_tokens": []}
    for c in cases:
        u = c.get("usage") or {}
        fields["steps"].append(c.get("steps"))
        fields["wall_seconds"].append(c.get("wall_seconds"))
        for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
            fields[k].append(u.get(k))
        i, o, cache = u.get("input_tokens"), u.get("output_tokens"), u.get("cached_input_tokens")
        fields["total_tokens"].append(i+o if i is not None and o is not None else None)
        fields["uncached_input_tokens"].append(i-cache if i is not None and cache is not None else None)
    output = {}
    for field, values in fields.items():
        known = [v for v in values if v is not None]
        output[field] = dict(known_count=len(known), unknown_count=len(values)-len(known),
                             known_sum=sum(known) if known else None,
                             known_mean=statistics.mean(known) if known else None)
    output["steps"]["partial_lower_bound_cases"] = sum(not c.get("steps_complete", False) for c in cases)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    w = args.workspace.resolve()
    parent_tag = "v3-evidence-context-fold-20261028"
    retest_tag = "v3-evidence-context-fold-20261029-unscored-retest"
    original_path = w / "results" / (parent_tag + "-final.json")
    original = read(original_path)
    assert digest(original_path) == read(w / "runs" / (parent_tag + "-launch.json"))["final_report_sha256"]
    sources, audit = {}, dict(native_results=0, first_round_files=0, supplemental_trajectories=0)
    for tag in (parent_tag, retest_tag):
        for path, sha in read(w / "runs" / (tag + "-freeze.json"))["source_sha256"].items():
            assert path not in sources or sources[path] == sha, path
            sources[path] = sha
    for path, sha in sources.items():
        assert digest(path) == sha, path
    first = original["cases"]
    by_key = {key(c): c for c in first}
    assert len(first) == len(by_key) == 50
    for c in first:
        for path, sha in c["files"].items():
            assert digest(path) == sha, path
            audit["first_round_files"] += 1
        if c["native_result"]:
            assert digest(c["native_result"]) == c["native_sha256"]
            n = read(c["native_result"])
            assert n["eval_time"] == 1 and not n.get("policy_error")
            assert (n["success_rate"] == 1) == c["success"]
            audit["native_results"] += 1
    status = read(w / "runs" / retest_tag / "status.json")
    assert status["complete"] and not status["active"] and not status["pending"]
    expected = {key(c) for c in first if c["success"] is None}
    assert len(status["finished"]) == len(expected)
    assert {key(c) for c in status["finished"]} == expected
    supplemental = []
    for entry in status["finished"]:
        old = by_key[key(entry)]
        directory = Path(entry["output"])
        config, old_config = read(directory / "run.json"), read(Path(old["directory"]) / "run.json")
        assert config["git_commit"] == original["candidate_commit"]
        assert {k:v for k,v in config["args"].items() if k != "output"} == {
            k:v for k,v in old_config["args"].items() if k != "output"}
        assert config["prompt_sha256"] == old_config["prompt_sha256"] == digest(directory / "agent-prompt.txt")
        assert config["layout_sha256"] == old["layout_sha256"]
        for path, sha in config["source_sha256"].items():
            assert digest(path) == sha
        calls, malformed = read_jsonl(directory / "mcp-calls.jsonl")
        assert not malformed
        accounting = control_accounting(directory, calls)
        commands, malformed = read_jsonl(directory / "commands.jsonl")
        assert not malformed
        for command in commands:
            assert all((Path(command["observation"]) / f).is_file()
                       for f in ("state.json", "head.png", "wrist_l.png", "wrist_r.png"))
        audit["supplemental_trajectories"] += 1
        result_path = directory / "result.json"
        result = read(result_path) if result_path.exists() else None
        row = dict(seed=entry["seed"], variant=entry["variant"], layout=entry["layout"],
                   directory=str(directory), source_attempt="supplemental", success=None,
                   native_result=None, native_sha256=None, baselines=old["baselines"],
                   steps=accounting["native_control_steps"],
                   steps_complete=accounting["native_control_steps_complete"],
                   layout_sha256=config["layout_sha256"],
                   usage=read(directory / "agent-exit.json").get("usage"),
                   wall_seconds=(result or read(directory / "exit.json"))["wall_seconds"],
                   files={str(p):digest(p) for p in [directory / n for n in (
                       "run.json", "agent-exit.json", "agent-prompt.txt", "commands.jsonl",
                       "actions.jsonl", "mcp-calls.jsonl", "gpt-plan.json", "result.json",
                       "episode_end.json", "exit.json")] if p.is_file()})
        if result:
            n = read(result["native_result"])
            assert result["complete"] and n == result["native"]
            assert n["eval_time"] == 1 and not n.get("policy_error")
            assert digest(result["native_result"]) == result["native_sha256"]
            assert result["native_control_steps"] == row["steps"]
            row.update(success=n["success_rate"] == 1, native_result=result["native_result"],
                       native_sha256=result["native_sha256"])
            audit["native_results"] += 1
        else:
            assert not list(directory.rglob("_result.json")), "Unaccounted native scoring"
        supplemental.append(row)
    replacements = {key(c): c for c in supplemental if c["success"] is not None}
    merged = [replacements.get(key(c), dict(c, source_attempt="first_round")) for c in first]
    paired = {}
    baseline_root = w.parents[1]
    for name in ("v3", "preoptimization", "execution_aligned"):
        candidates, baselines, improvements, losses = [], [], [], []
        for c in merged:
            b = c["baselines"][name]
            if c["success"] is None or b["success"] is None:
                continue
            directory = Path(b["directory"])
            if not directory.is_absolute():
                directory = baseline_root / directory
            result = read(directory / "result.json")
            native = read(result["native_result"])
            assert digest(result["native_result"]) == result["native_sha256"]
            assert native["eval_time"] == 1 and not native.get("policy_error")
            assert b["success"] == (native["success_rate"] == 1)
            assert read(directory / "run.json")["layout_sha256"] == c["layout_sha256"]
            candidates.append(c)
            baselines.append(dict(success=b["success"], steps=result["native_control_steps"],
                steps_complete=result["native_control_steps_complete"], wall_seconds=result["wall_seconds"],
                usage=result["native_agent"].get("usage")))
            case_id = f"seed{c['seed']}-{c['variant']}-l{c['layout']}"
            if c["success"] and not b["success"]:improvements.append(case_id)
            if not c["success"] and b["success"]:losses.append(case_id)
        paired[name] = dict(paired_count=len(candidates), current_success=counts(candidates)["success"],
            baseline_success=counts(baselines)["success"], improvements=improvements, losses=losses,
            candidate_metrics=metrics(candidates), baseline_metrics=metrics(baselines))
    audit.update(passed=True, frozen_sources_verified=len(sources))
    report = dict(schema="roborag.fold-supplement-final.v1", generated_at=datetime.now(timezone.utc).isoformat(),
        candidate_commit=original["candidate_commit"], original_report=str(original_path),
        original_report_sha256=digest(original_path), first_round_counts=counts(first),
        supplemental_counts=counts(supplemental), merged_counts=counts(merged),
        first_round_cases=first, supplemental_attempts=supplemental, cases=merged, paired=paired,
        all_attempt_costs=metrics([*first, *supplemental]), audit=audit,
        rule="Fill only original missing native scores; preserve original successes/failures; all attempts included in cost; unknown usage is not zero; cached input and reasoning output are subsets")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k:report[k] for k in ("first_round_counts", "supplemental_counts", "merged_counts", "audit")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
