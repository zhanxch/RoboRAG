"""Summarize the finished diagnostic control without counting unknowns as losses.

Run the control auditor and both registered evaluation commands first. This
script reads their evidence; it never starts, retries, or changes an episode.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = "v3-original-control-20261028"


def read(path):
    return json.loads(path.read_text())


def key(row):
    return row["seed"], row["variant"], row["layout"]


def case_name(row):
    return f"seed{row['seed']}-{row['variant']}-l{row['layout']}"


def scored_pairs(pairs):
    comparable = [p for p in pairs if p["baseline"]["complete"] and p["candidate"]["complete"]]
    groups = {name: [] for name in ("joint_success", "gains", "losses", "joint_failure")}
    for pair in comparable:
        before, after = pair["baseline"]["success"], pair["candidate"]["success"]
        assert type(before) is bool and type(after) is bool
        group = ("joint_success" if before else "gains") if after else ("losses" if before else "joint_failure")
        groups[group].append(case_name(pair))
    excluded = [dict(case=case_name(p), baseline_success=p["baseline"]["success"],
                     control_success=p["candidate"]["success"])
                for p in pairs if not (p["baseline"]["complete"] and p["candidate"]["complete"])]
    return dict(count=len(comparable), excluded_unscored=excluded,
                baseline_successes=sum(p["baseline"]["success"] for p in comparable),
                control_successes=sum(p["candidate"]["success"] for p in comparable),
                **{name: dict(count=len(cases), cases=cases) for name, cases in groups.items()})


def usage_coverage(rows):
    fields = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
    counts = {field: sum(type(r.get("usage", {}).get(field)) is int for r in rows) for field in fields}
    uncached = [r["usage"]["input_tokens"] - r["usage"]["cached_input_tokens"]
                for r in rows if type(r.get("usage", {}).get("input_tokens")) is int
                and type(r.get("usage", {}).get("cached_input_tokens")) is int]
    return dict(attempts=len(rows), known_attempts=counts,
                known_uncached_attempts=len(uncached),
                uncached_input_tokens_known_lower_bound=sum(uncached),
                note="Missing usage remains unknown; known lower bounds are not full totals.")


def main():
    out = ROOT / "runs" / STEM
    status_path = out / "status.json"
    audit_path = out.with_suffix(".completed-audit.json")
    protocol_path = ROOT / "experiments/fold_clothes" / f"{STEM}.json"
    status, audit, protocol = read(status_path), read(audit_path), read(protocol_path)
    expected = {key(r) for r in read(ROOT / protocol["cases_file"])}
    assert status["complete"] and not status["active"] and not status["pending"], "Control still running"
    assert len(status["finished"]) == len(expected) == protocol["cases_count"]
    assert {key(r) for r in status["finished"]} == expected
    assert audit["batch_complete"] and audit["counts"]["attempts_closed"] == len(expected)
    assert len(audit["cases"]) == len(expected)
    assert {r["case"] for r in audit["cases"]} == {case_name(r) for r in status["finished"]}
    sources = [status_path, audit_path, protocol_path]
    comparisons = {}
    for name in ("history", "refactor"):
        path = ROOT / "results" / f"v3-original-control-vs-{name}-20261028.json"
        sources.append(path)
        document = read(path)
        rows, pairs = document["cases"], document["paired"]["pairs"]
        assert not document["pending"] and len(rows) == len(pairs) == len(expected)
        assert {key(r) for r in rows} == {key(p) for p in pairs} == expected
        assert document["overall"]["scored"] == audit["counts"]["scored"]
        assert document["overall"]["successes"] == audit["counts"]["successes"]
        assert all(p["baseline"]["complete"] for p in pairs)
        comparison = scored_pairs(pairs)
        assert comparison["count"] == audit["counts"]["scored"]
        assert comparison["joint_success"]["count"] == document["paired"]["jointly_successful"]["count"]
        comparisons[name] = dict(
            both_scored=comparison,
            by_variant={variant: scored_pairs([p for p in pairs if p["variant"] == variant])
                        for variant in sorted({p["variant"] for p in pairs})},
            all_attempts={side: document["paired"][side] for side in ("baseline", "candidate")},
            usage_coverage={side: usage_coverage([p[side] for p in pairs])
                            for side in ("baseline", "candidate")},
            jointly_successful=document["paired"]["jointly_successful"],
        )
    incident_path = out.with_suffix(".runtime-incidents.jsonl")
    sources.append(incident_path)
    incidents = [json.loads(line) for line in incident_path.read_text().splitlines()]
    starts = [r["started_epoch"] for r in status["finished"]]
    ends = [r["ended_epoch"] for r in status["finished"]]
    report = dict(
        schema="roborag.original-v3-control-final-summary.v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        counts=audit["counts"],
        attempt_span_seconds=max(ends) - min(starts),
        attempt_span_note="First worker start to last worker end; distinct from sum of episode times.",
        comparisons=comparisons,
        observed_runtime_incidents=incidents,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        candidate_acceptance=protocol["refactor_result"],
        interpretation=[
            "Control is the candidate side of both comparisons; gains and losses are relative to each named baseline.",
            "Both-scored comparisons exclude unknown outcomes explicitly; all-attempt cost summaries retain them.",
            "Recorded runtime incidents are observations, not an exhaustive proof of service stability or causal explanations.",
            "Sequential diagnostic control, not randomized concurrent treatment or statistical equivalence evidence.",
            "Historical and refactor results remain immutable; this control cannot make the failed refactor gate pass.",
        ],
    )
    destination = ROOT / "results" / f"{STEM}-summary.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(counts=report["counts"], output=str(destination)), ensure_ascii=False))


if __name__ == "__main__":
    main()
