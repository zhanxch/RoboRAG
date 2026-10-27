import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from roborag.evaluation.episode import summarize

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--policy", required=True)
parser.add_argument("--heldout", action="store_true")
args = parser.parse_args()
p = args.run.resolve()
report = summarize(p)
calls = [json.loads(line) for line in (p / "mcp-calls.jsonl").read_text().splitlines()]
image_receipts = defaultdict(list)
malformed_event_lines = 0
if (p / "agent-events.jsonl").exists():
    with (p / "agent-events.jsonl").open() as events:
        for line in events:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed_event_lines += 1
                continue
            item = event.get("item", {})
            if event.get("type") != "item.completed" or item.get("tool") != "read_action_reference":
                continue
            response = item.get("result") or {}
            content = response.get("content", []) if isinstance(response, dict) else []
            if not content or content[0].get("type") != "text":
                continue
            try:
                packet = json.loads(content[0]["text"])
            except json.JSONDecodeError:
                continue
            if packet.get("role") != "HISTORICAL_REFERENCE":
                continue
            image_receipts[(packet["source"], packet["window_section"])].append(
                dict(
                    provenance="actual completed native MCP response in agent-events.jsonl",
                    count=sum(block.get("type") == "image" for block in content),
                    labels=[
                        block["text"]
                        for block in content
                        if block.get("type") == "text"
                        and block.get("text", "").startswith("HISTORICAL ")
                    ],
                )
            )
expanded = []
plans = []
measures = []
for call in calls:
    tool = call["tool"]
    request = call["request"]
    result = call["result"]
    if tool == "read_action_reference":
        receipts = image_receipts[(request["source"], request["section"])]
        expanded.append(
            dict(
                source=request["source"],
                section=request["section"],
                operation=result.get("operation"),
                phase_interval=result.get("phase_interval"),
                window_selection=result.get("window_selection"),
                delivered_images=receipts.pop(0) if receipts else None,
                phase_context_returned=bool(result.get("phase_context")),
                phase_context=result.get("phase_context"),
            )
        )
    elif tool == "record_plan":
        plans.append(request)
    elif tool == "measure_rim":
        measurement = result["result"]
        measures.append(
            dict(request=request["rim_measure"], step=result.get("step_before"), result=measurement)
        )
adopted = sorted(
    {
        ref
        for plan in plans
        for step in plan.get("steps", [])
        for ref in step.get("evidence_refs", [])
    }
)
guards = (
    [json.loads(line) for line in (p / "ik-continuity.jsonl").read_text().splitlines()]
    if (p / "ik-continuity.jsonl").exists()
    else []
)
rejections = [
    dict(
        attempted_step=g.get("attempted_step"),
        failed_checks=[check for check in g.get("checks", []) if check.get("passed") is False],
    )
    for g in guards
    if g.get("accepted") is False
]
report.update(
    role="frozen_heldout_evaluation" if args.heldout else "development_candidate_only",
    policy_version=args.policy,
    heldout=args.heldout,
    rag_audit=dict(
        tool_counts=dict(Counter(c["tool"] for c in calls)),
        expanded_phases=expanded,
        malformed_native_event_lines=malformed_event_lines,
        plan_evidence_refs=adopted,
        recorded_plans=plans,
        semantic_limit="Tool delivery and citation are audited mechanics. Correct adaptation of the reference motion must be established from the recorded plan, generated motion and native feedback; it is not inferred from an ID or phase_context presence.",
    ),
    measurement_audit=dict(
        native_calls=measures,
        fit_rim_success_count=sum(
            m["request"]["command"] == "fit_rim" and m["result"].get("plan_ok") is True
            for m in measures
        ),
        fit_rim_failure_count=sum(
            m["request"]["command"] == "fit_rim" and m["result"].get("plan_ok") is not True
            for m in measures
        ),
        table_plane_success_count=sum(
            m["request"]["command"] == "table_plane" and m["result"].get("plan_ok") is True
            for m in measures
        ),
        note="These are native policy calls only; offline replay is excluded. Measurement acceptance is not task success.",
    ),
    ik_guard=dict(rejected_event_count=len(rejections), rejected_events=rejections),
    source_snapshot=str(p / "source-snapshot") if (p / "source-snapshot").exists() else None,
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(
    json.dumps(
        dict(
            output=str(args.output),
            native=report.get("native"),
            control_steps=report.get("native_control_steps"),
            fit_successes=report["measurement_audit"]["fit_rim_success_count"],
            fit_failures=report["measurement_audit"]["fit_rim_failure_count"],
        ),
        ensure_ascii=False,
    )
)
