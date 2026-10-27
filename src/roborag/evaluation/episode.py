"""Episode summaries use simulator evidence even if an MCP reply is lost."""

from __future__ import annotations

from pathlib import Path

from roborag.io import read_json, read_jsonl


def control_accounting(path, calls=None):
    path = Path(path)
    actions, malformed_actions = read_jsonl(path / "actions.jsonl")
    commands, malformed_commands = read_jsonl(path / "commands.jsonl")
    if calls is None:
        calls, _ = read_jsonl(path / "mcp-calls.jsonl")
    chunks = [c for c in calls if c.get("tool") == "execute_chunk"]
    end = read_json(path / "episode_end.json")
    # Step counters survive partial chunks; sums of received chunks do not.
    counters = [a["step"] for a in actions if isinstance(a.get("step"), int)]
    counters += [c["step_after"] for c in commands if isinstance(c.get("step_after"), int)]
    counters += [c["result"]["result"]["step"] for c in chunks]
    if end is not None:
        counters.append(end["step"])
    step = max(counters) if counters else None
    finalized = end is not None and end["step"] == step
    # A native command receipt is authoritative. MCP receipts are a fallback
    # for older runs that stopped between exchange and archive publication.
    receipts = [c["result"] for c in commands if "executed_waypoints" in c.get("result", {})]
    if not receipts:
        receipts = [c["result"]["result"] for c in chunks]
    logged_rows = sum("delta_row" in a for a in actions)
    return {
        "native_control_steps": step,
        "native_control_steps_complete": finalized,
        "control_steps_status": "final_counter"
        if finalized
        else "lower_bound"
        if step is not None
        else "unknown",
        "executed_delta_rows": max(logged_rows, sum(r.get("executed_rows", 0) for r in receipts)),
        "executed_waypoints": sum(r.get("executed_waypoints", 0) for r in receipts),
        "recorded_action_chunks": len(receipts),
        "action_log_rows": len(actions),
        "malformed_action_records": malformed_actions,
        "malformed_command_records": malformed_commands,
        "control_step_note": "Native action dispatches; partial-run counters are lower bounds, not PhysX substeps.",
    }


def workflow_completed(calls):
    names = [c["tool"] for c in calls]
    if "record_graph_plan" in names:
        required = ["get_task_sop", "record_graph_plan", "enter_stage", "execute_chunk"]
        return all(n in names for n in required) and [names.index(n) for n in required] == sorted(
            names.index(n) for n in required
        )
    required = ["retrieve_evidence", "record_plan", "execute_chunk"]
    return all(n in names for n in required) and names.index(required[0]) < names.index(
        required[1]
    ) < names.index(required[2])


def summarize(path):
    path = Path(path)
    configuration = read_json(path / "run.json", {})
    result = read_json(path / "result.json", {})
    calls, malformed = read_jsonl(path / "mcp-calls.jsonl")
    agent = read_json(path / "agent-exit.json", {})
    usage = agent.get("accounting", {}).get("reported_usage", {})
    names = [c["tool"] for c in calls]
    plan = read_json(path / "gpt-plan.json", {})
    inputs, cached = usage.get("input_tokens"), usage.get("cached_input_tokens")
    arguments = configuration.get("args", {})
    return dict(
        run=str(path.resolve()),
        complete=result.get("complete", False),
        native=result.get("native"),
        task=configuration.get("task"),
        variant=arguments.get("variant", "standard"),
        layout=arguments.get("layout"),
        seed=arguments.get("seed"),
        controller=configuration.get("controller"),
        architecture=arguments.get("architecture", "legacy"),
        model=configuration.get("model"),
        effort=arguments.get("effort"),
        failure=read_json(path / "sim_failure.json") or read_json(path / "exit.json"),
        reported_usage=usage,
        actual_billed_usd=None,
        cached_input_fraction=cached / inputs
        if isinstance(inputs, int) and inputs and isinstance(cached, int)
        else None,
        uncached_input_tokens=inputs - cached
        if isinstance(inputs, int) and isinstance(cached, int)
        else None,
        completed_native_turns=agent.get("completed_turns"),
        model_request_count=None,
        tool_counts={name: names.count(name) for name in sorted(set(names))},
        **control_accounting(path, calls),
        malformed_mcp_records=malformed,
        retrieval_context_chars=[
            c["result"]["diagnostics"].get("context_chars")
            for c in calls
            if c["tool"] == "retrieve_evidence"
        ],
        plan_steps=len(plan.get("phases", plan.get("steps", []))),
        plan_adaptation=plan.get("adaptation"),
        workflow_verified=workflow_completed(calls),
        wall_seconds=result.get(
            "wall_seconds", read_json(path / "exit.json", {}).get("wall_seconds")
        ),
        agent_wall_seconds=agent.get("wall_seconds"),
    )


def main():
    import argparse
    import json

    from roborag.io import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.run)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
