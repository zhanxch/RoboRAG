#!/usr/bin/env python3
"""Summarize completed and interrupted MCP pilots without inventing usage."""
import argparse
import json
from pathlib import Path


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def summarize(path):
    configuration = load(path / "run.json", {})
    result = load(path / "result.json", {})
    calls = [json.loads(x) for x in (path / "mcp-calls.jsonl").read_text().splitlines()] if (path / "mcp-calls.jsonl").exists() else []
    agent = load(path / "agent-exit.json", {})
    usage = agent.get("accounting", {}).get("reported_usage", {})
    names = [c["tool"] for c in calls]
    chunks = [c for c in calls if c["tool"] == "execute_chunk"]
    plan = load(path / "gpt-plan.json", {})
    inputs, cached = usage.get("input_tokens"), usage.get("cached_input_tokens")
    complete = result.get("complete", False)
    native = result.get("native")
    return dict(run=str(path.resolve()), complete=complete, native=native,
        controller=configuration.get("controller"), model=configuration.get("model"),
        effort=configuration.get("args", {}).get("effort"),
        failure=load(path / "sim_failure.json") or load(path / "exit.json"),
        reported_usage=usage, actual_billed_usd=None,
        cached_input_fraction=cached/inputs if isinstance(inputs, int) and inputs and isinstance(cached, int) else None,
        uncached_input_tokens=inputs-cached if isinstance(inputs, int) and isinstance(cached, int) else None,
        completed_native_turns=agent.get("completed_turns"), model_request_count=None,
        tool_counts={name:names.count(name) for name in sorted(set(names))},
        native_control_steps=sum(c["result"]["result"]["action_steps"] for c in chunks),
        executed_delta_rows=sum(c["result"]["result"].get("executed_rows",0) for c in chunks),
        executed_waypoints=sum(c["result"]["result"]["executed_waypoints"] for c in chunks),
        retrieval_context_chars=[c["result"]["diagnostics"].get("context_chars") for c in calls if c["tool"] == "retrieve_evidence"],
        plan_steps=len(plan.get("steps", [])), plan_adaptation=plan.get("adaptation"),
        workflow_verified=result.get("retrieval_before_plan_before_execution"),
        wall_seconds=result.get("wall_seconds", load(path / "exit.json", {}).get("wall_seconds")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(schema="roborag.mcp-pilot-summary.v1", runs=[summarize(p) for p in args.runs],
        interpretation=["Development integration pilot, not a heldout success-rate study.",
            "Historical RAG data supports GPT; generated plans and live actions are separate records.",
            "MCP chunks, delta rows, waypoints and native control steps are different counts; PhysX integration substeps are not counted here.",
            "Reported token usage covers online policy calls, not offline graph construction or a per-run invoice.",
            "Cache usage is measured, but no causal improvement over a no-RAG policy is established without a controlled comparison.",
            "Interrupted attempts may consume unreported tokens; missing usage is not zero."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
