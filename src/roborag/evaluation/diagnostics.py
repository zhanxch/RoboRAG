#!/usr/bin/env python3
"""Summarize recorded action/measurement feedback without inferring task success."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from roborag.evaluation.comparison import candidate_rows
from roborag.io import write_json


def read_log(path):
    data = path.read_bytes()
    return [json.loads(line) for line in data.splitlines() if line.strip()], hashlib.sha256(
        data
    ).hexdigest()


def action_summary(call):
    receipt = call["result"]
    result = receipt["result"]
    return dict(
        subgoal=call["request"].get("subgoal"),
        step_before=receipt["step_before"],
        step_after=receipt["step_after"],
        submitted_rows=result.get("submitted_rows"),
        executed_rows=result.get("executed_rows"),
        discarded_rows=result.get("discarded_rows"),
        rejection=result.get("rejection"),
    )


def diagnose(row):
    report = {
        key: row[key]
        for key in (
            "seed",
            "variant",
            "layout",
            "directory",
            "complete",
            "success",
            "steps",
            "native_result",
            "native_sha256",
        )
    }
    if not row["complete"]:
        return dict(
            report,
            diagnostics_available=False,
            diagnostic_note="Episode incomplete; live action logs are not treated as final.",
        )
    root = Path(row["directory"])
    calls, calls_hash = read_log(root / "mcp-calls.jsonl")
    ik_path = root / "ik-continuity.jsonl"
    if ik_path.exists():
        events, ik_hash = read_log(ik_path)
        rejected = []
        for i, event in enumerate(events):
            if event["accepted"]:
                continue
            later = next((e for e in events[i + 1 :] if e["accepted"]), None)
            rejected.append(
                dict(
                    attempted_step=event["attempted_step"],
                    errors=[
                        dict(arm=c["arm"], error=c.get("error"))
                        for c in event["checks"]
                        if not c["passed"]
                    ],
                    next_accepted_step=later["attempted_step"] if later else None,
                )
            )
        report["ik"] = dict(
            attempts=len(events),
            rejections=len(rejected),
            continued_after_rejection=sum(r["next_accepted_step"] is not None for r in rejected),
            rejected_actions=rejected,
        )
    else:
        ik_hash = None
        report["ik"] = None
    actions = [action_summary(c) for c in calls if c["tool"] == "execute_chunk"]
    report["actions"] = dict(
        chunks=len(actions),
        zero_step_chunks=sum(a["step_before"] == a["step_after"] for a in actions),
        discarded_tail_chunks=sum((a["discarded_rows"] or 0) > 0 for a in actions),
        timeline=actions,
    )
    measurements, checks = [], []
    for i, call in enumerate(calls):
        if call["tool"] != "measure_planar":
            continue
        command = call["request"]["planar_measure"]["command"]
        result = call["result"]["result"]
        measurements.append(
            dict(
                command=command,
                plan_ok=result.get("plan_ok"),
                reason=result.get("plan_fail_reason"),
            )
        )
        if command == "planar_check":
            following = next((c for c in calls[i + 1 :] if c["tool"] == "execute_chunk"), None)
            checks.append(
                dict(
                    step=call["result"]["step_after"],
                    **{
                        key: result.get(key)
                        for key in (
                            "plan_ok",
                            "plan_fail_reason",
                            "alignment_verified",
                            "alignment_scope",
                            "residual_contact_m",
                            "residual_yaw_deg",
                            "released_fit_rms_m",
                        )
                    },
                    next_action=action_summary(following) if following else None,
                )
            )
    report["measurements"] = dict(
        counts=dict(Counter(m["command"] for m in measurements)),
        failures=dict(
            Counter(m["reason"] or "unspecified" for m in measurements if m["plan_ok"] is False)
        ),
        checks=checks,
    )
    report["tool_counts"] = dict(Counter(c["tool"] for c in calls))
    report["log_sha256"] = dict(mcp_calls=calls_hash, ik_continuity=ik_hash)
    report["diagnostics_available"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = candidate_rows(
        args.runs, task="push_T"
    )  # checks native hashes and rejects mixed revisions
    cases = [diagnose(row) for row in rows]
    available = [c for c in cases if c["diagnostics_available"]]
    guarded = [c for c in available if c["ik"] is not None]
    rejections = sum(c["ik"]["rejections"] for c in guarded)
    continued = sum(c["ik"]["continued_after_rejection"] for c in guarded)
    all_ik_known = bool(available) and len(guarded) == len(available)
    summary = dict(
        recorded=len(cases),
        completed=len(available),
        native_successes=sum(c["success"] for c in available),
        ik_logs_available=len(guarded),
        ik_rejections=rejections if all_ik_known else None,
        ik_rejections_known_lower_bound=rejections,
        rejected_actions_with_later_acceptance=continued if all_ik_known else None,
        rejected_actions_with_later_acceptance_known_lower_bound=continued,
        zero_step_chunks=sum(c["actions"]["zero_step_chunks"] for c in available),
    )
    write_json(
        args.output,
        dict(
            schema="roborag.push-t-diagnostics.v1",
            summary=summary,
            cases=cases,
            interpretation=[
                "Native results alone establish success; geometric alignment is a separate visual measurement.",
                "IK attempts exclude direct joint-space home trajectories and are not the native control step count.",
                "Later accepted actions show continuation, not a causal safety benefit or successful contact recovery.",
                "A check's next_action is the next recorded execution, not a forced graph transition or a claim about its intent.",
                "Missing IK logs are unavailable, not zero rejections. Incomplete episodes retain identity but no final diagnostics.",
                "No best-of selection, automatic retries, motion generation or changes to the policy occur in this report.",
            ],
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
