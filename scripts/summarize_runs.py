#!/usr/bin/env python3
"""Read native completion evidence without treating missing/interrupted runs as scores."""

import argparse
import json
import math
from pathlib import Path


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def summarize(folder):
    run = load(folder / "run.json")
    if not run:
        return None
    result_paths = list((folder / "native").rglob("_result.json"))
    failures = list((folder / "native").rglob("_infrastructure_failure.json"))
    failures += list((folder / "native").rglob("_policy_error.json"))
    native = load(result_paths[0]) if len(result_paths) == 1 else None
    exit_info = load(folder / "exit.json")
    manual_stop = load(folder / "manual_stop.json")
    decisions = []
    for path in folder.glob("policy/*/decisions.jsonl"):
        for line in path.read_text().splitlines():
            try:
                decisions.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A running append may be incomplete; never infer a completed decision.
    calls = list((folder / "decisions").glob("call-*"))
    completed, known_tokens, missing_usage = 0, 0, 0
    for call in calls:
        response = load(call / "response.json")
        if response and response.get("status") == "completed":
            completed += 1
            usage = response.get("usage", {})
            if usage.get("total_tokens") is not None:
                known_tokens += usage["total_tokens"]
            else:
                missing_usage += 1
        else:
            missing_usage += 1
    score = native.get("score") if native else None
    valid = bool(native and native.get("eval_time") == 1 and not failures and not native.get("policy_error")
                 and type(score) in (float, int) and math.isfinite(score) and 0 <= score <= 100
                 and native.get("success_rate") in (0, 1)
                 and not any(d.get("interrupted") or d.get("policy_error")
                             for d in native.get("details", {}).values()))
    if run.get("capture_only"):
        valid = False
        status = "capture_only"
    elif valid:
        status = "native_completed"
    elif manual_stop:
        status = "interrupted_by_experimenter"
    elif failures or exit_info:
        status = "incomplete_or_infrastructure_failure"
    else:
        status = "running_or_interrupted_no_exit_evidence"
    row = {"run": folder.name, "mode": run["arguments"]["mode"], "task": run["arguments"]["task"],
           "evaluation_role": run.get("evaluation_role", "unspecified"),
           "layout": run["arguments"]["layout"], "status": status, "valid_native_episode": valid,
           "score_0_100": native["score"] if valid else None,
           "success_rate": native["success_rate"] if valid else None,
           "native_result": str(result_paths[0]) if len(result_paths) == 1 else None,
           "completed_model_calls": completed, "attempted_model_calls": len(calls),
           "known_total_tokens": known_tokens, "calls_without_usage": missing_usage,
           "latest_decision_step": decisions[-1]["step"] if decisions else None,
           "latest_intent_not_verified_outcome": decisions[-1]["execution_note"] if decisions else None,
           "exit": exit_info, "manual_stop": manual_stop}
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("runs"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [row for folder in sorted(args.root.iterdir()) if folder.is_dir()
            if (row := summarize(folder)) is not None]
    encoded = json.dumps(rows, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
