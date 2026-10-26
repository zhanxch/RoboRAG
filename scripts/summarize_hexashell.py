#!/usr/bin/env python3
"""Rebuild the HexaShell public-suite results and consumption ledger from evidence."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from roborag.hexashell_accounting import USAGE_FIELDS, event_accounting, read_events, resource_accounting

PRIMARY = ["hexashell-public-calibrated-v2-20261023"]
PRE_CALIBRATION = ["hexashell-public-standard-v1-20261023", "hexashell-public-remaining-v1-20261023"]


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def relative(path):
    return str(path.relative_to(ROOT))


def category(directory):
    top = directory.relative_to(ROOT / "runs").parts[0]
    return ("primary" if top in PRIMARY else "pre_calibration" if top in PRE_CALIBRATION else
            "pilot" if top == "hexashell-native-batch-20261023" else "setup")


def collect_execution(directory):
    manifest = read(directory / "run.json", {})
    arguments = manifest.get("arguments", {})
    result = read(directory / "result.json")
    failed = read(directory / "exit.json")
    agent = read(directory / "agent-exit.json", {})
    execution = read(directory / "execution.json", {})
    end = read(directory / "episode_end.json", {})
    resources = read(directory / "resource_summary.json", {})
    if not resources and "child_cpu_seconds" in execution:
        resources = {key: execution[key] for key in ("child_cpu_seconds", "maximum_child_rss_kib") if key in execution}
    events, malformed, event_files = [], 0, []
    for path in [directory / "agent-events.jsonl", directory / "events.jsonl", *sorted((directory / "decisions").glob("*/events.jsonl"))]:
        if path.is_file():
            values, broken = read_events(path)
            events.extend(values)
            malformed += broken
            event_files.append(relative(path))
    accounting = event_accounting(events, malformed)
    native, native_path = None, None
    if result is not None:
        assert result["complete"] and result["scored"], directory
        native_path = Path(result["native_result"])
        native = read(native_path)
        assert native == result["native"], native_path
        assert hashlib.sha256(native_path.read_bytes()).hexdigest() == result["native_sha256"], native_path
        assert native["eval_time"] == 1 and not native.get("policy_error"), native_path
        assert native["details"]["0"]["layout_id"] == arguments["layout"], native_path
    actions = read_events(directory / "actions.jsonl")[0]
    if actions:
        assert [entry["step"] for entry in actions] == list(range(1, len(actions)+1)), directory
    status = ("native_completed" if native is not None else
              "infrastructure_incomplete" if failed and not failed.get("complete") else
              "unscored_check" if category(directory) == "setup" else "not_finalized")
    process = read(directory / "process.json", {})
    if status == "not_finalized" and process.get("runner_pid"):
        command_line = Path(f"/proc/{process['runner_pid']}/cmdline")
        try:
            if str(directory).encode() in command_line.read_bytes().split(b"\0"):
                status = "running"
        except FileNotFoundError:
            pass
    return dict(directory=relative(directory), category=category(directory), status=status,
        variant=arguments.get("variant"), layout=arguments.get("layout"), seed=arguments.get("seed"),
        model=manifest.get("model", "gpt-6-astra" if event_files else None),
        success=native["details"]["0"]["success"] if native else None,
        progress=native["score"] if native else None,
        native_result=relative(native_path) if native_path else None,
        native_sha256=result.get("native_sha256") if result else None,
        steps=end.get("step"), budgeted_commands=end.get("commands"), stop_reason=end.get("reason"),
        action_log_records=len(actions) if (directory / "actions.jsonl").exists() else None,
        action_log_complete=len(actions)==end["step"] if end else None,
        wall_seconds=(result or failed or execution).get("wall_seconds") if (result or failed or execution) else agent.get("wall_seconds"),
        agent_wall_seconds=agent.get("wall_seconds"), agent_returncode=agent.get("returncode"),
        runner_error=(failed or {}).get("error"), local_resources=resources,
        logical_artifact_bytes=sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()),
        event_files=event_files, accounting=accounting,
        source_sha256=manifest.get("source_sha256"),
        limitations=["Native video contains observation snapshots, not every physics step."])


def aggregate(rows):
    scored = [row for row in rows if row["status"] == "native_completed"]
    cpu = []
    for row in rows:
        local = row["local_resources"]
        if "child_cpu_user_seconds" in local and "child_cpu_system_seconds" in local:
            cpu.append(local["child_cpu_user_seconds"]+local["child_cpu_system_seconds"])
        elif "child_cpu_seconds" in local:
            cpu.append(local["child_cpu_seconds"])
    return dict(executions=len(rows), scored_episodes=len(scored),
        successful_episodes=sum(row["success"] for row in scored),
        success_rate=sum(row["success"] for row in scored)/len(scored) if scored else None,
        mean_progress=sum(row["progress"] for row in scored)/len(scored) if scored else None,
        known_token_lower_bounds={key: sum(row["accounting"]["known_token_lower_bounds"][key] for row in rows) for key in USAGE_FIELDS},
        native_cli_executions=sum(bool(row["event_files"]) for row in rows),
        actor_executions_without_terminal_usage=sum(bool(row["event_files"]) and row["status"] != "running" and not row["accounting"]["terminal_usage_available"] for row in rows),
        live_actor_executions_awaiting_usage=sum(bool(row["event_files"]) and row["status"] == "running" and not row["accounting"]["terminal_usage_available"] for row in rows),
        invocation_count_note="Counts CLI executions, including startup failures; does not imply the same number of model requests.",
        sum_episode_wall_seconds=sum(row["wall_seconds"] for row in rows if row["wall_seconds"] is not None),
        wall_measurement_missing_count=sum(row["wall_seconds"] is None for row in rows),
        known_child_cpu_seconds=sum(cpu), cpu_measurement_missing_count=len(rows)-len(cpu),
        logical_artifact_bytes=sum(row["logical_artifact_bytes"] for row in rows),
        summed_wall_note="Sum of per-execution residence times; overlapping episodes are counted separately. Use batch_wall_seconds for elapsed batch time.",
        actual_billed_usd=None)


def audit_primary(rows):
    """Verify retained inputs without changing any running evaluation."""
    protocol = read(ROOT / "experiments/hexashell-fold-public-suite-v2.json")
    manifests, prompt_hashes, manual_hashes = [], set(), set()
    source_files_checked = 0
    for row in rows:
        directory = ROOT / row["directory"]
        manifest = read(directory / "run.json")
        manifests.append(manifest)
        assert manifest["hexashell_commit"] == protocol["frozen_hexashell_commit"], directory
        arguments = manifest["arguments"]
        assert arguments["seed"] == 0 and arguments["transport"] == "native" and not arguments["probe_only"], directory
        assert arguments["variant"] in ("standard", "random") and arguments["layout"] in range(10), directory
        assert manifest["model"] == "gpt-6-astra" and manifest["effort"] == "xhigh" and manifest["native_step_limit"] == 500, directory
        ready = read(directory / "ready.json")
        if ready:
            assert ready["step_limit"] == 500, directory
        if row["status"] == "native_completed":
            assert row["steps"] <= 500 and row["budgeted_commands"] <= 60, directory
            invocation = read(directory / "agent-invocation.json")
            command = invocation["command"]
            assert invocation["persistent"] and invocation["model"] == protocol["model"], directory
            assert command[command.index("--model")+1] == protocol["model"], directory
            assert f'model_reasoning_effort="{protocol["reasoning_effort"]}"' in command, directory
        assert not manifest["hexashell_worktree"], directory
        assert hashlib.sha256(Path(manifest["layout"]).read_bytes()).hexdigest() == manifest["layout_sha256"], directory
        for name, expected in manifest["source_sha256"].items():
            source = ROOT / name if name.startswith("vendor/") else directory / "source" / Path(name).name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, source
            source_files_checked += 1
        for name, hashes in (("agent-prompt.txt", prompt_hashes), ("INTERFACE.md", manual_hashes)):
            path = directory / name
            if path.exists():
                hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    identical = all(manifest["source_sha256"] == manifests[0]["source_sha256"] for manifest in manifests)
    assert identical, "Primary source files changed within the frozen evaluation"
    dependencies_identical = all(all(manifest[key] == manifests[0][key] for key in ("dependencies", "agent_python", "hexashell_commit")) for manifest in manifests)
    assert dependencies_identical, "Primary dependency versions changed"
    assert len(prompt_hashes) <= 1 and len(manual_hashes) <= 1, "Primary prompt or interface changed"
    return dict(manifests_checked=len(manifests), source_files_checked=source_files_checked,
        source_hashes_identical=identical, layout_hashes_verified=True,
        dependency_versions_identical=dependencies_identical, model_effort_seed_transport_and_budgets_verified=True,
        prompt_sha256=sorted(prompt_hashes), manual_sha256=sorted(manual_hashes),
        completed_action_logs_complete=all(row["action_log_complete"] for row in rows if row["status"] == "native_completed"))


def main():
    directories = set()
    for top in (ROOT / "runs").glob("hexashell*"):
        for name in ("run.json", "agent-events.jsonl", "execution.json"):
            directories.update(path.parent for path in top.rglob(name))
    rows = [collect_execution(directory) for directory in sorted(directories)]
    batches = []
    for top in sorted((ROOT / "runs").glob("hexashell*")):
        manifest = read(top / "batch.json")
        if manifest is None:
            continue
        status = read(top / "status.json", {})
        samples = read_events(top / "resources.jsonl")[0]
        stopped = read(top / "stopped.json")
        elapsed = (stopped["stopped_at_epoch"] - datetime.fromisoformat(manifest["started_at"]).timestamp()
                   if stopped else status.get("wall_seconds"))
        batch_rows = [row for row in rows if (ROOT / row["directory"]).parent == top]
        finished_rows = [row for row in batch_rows if row["status"] == "native_completed"]
        batches.append(dict(directory=relative(top), category=category(top),
            complete=status.get("complete", False), batch_wall_seconds=elapsed,
            elapsed_time_basis="start-to-stop timestamp, including scheduler pause" if stopped else "scheduler monotonic clock",
            last_scheduler_status_wall_seconds=status.get("wall_seconds"),
            parallel=manifest["parallel"], scored_episodes=len(finished_rows),
            completed_throughput_episodes_per_hour=(len(finished_rows)*3600/elapsed if elapsed and status.get("complete") else None),
            concurrency_note="Throughput is observed for this batch. Without a matched serial run it is not a measured speedup.",
            stopped=stopped, planned_cases=manifest["cases"], resources=resource_accounting(samples)))
    reference = {}
    for split, filename in [("development", "dev_layout_retests.json"), ("new_layouts", "new_layout_tests.json")]:
        data = read(ROOT / "vendor/HexaShell/results" / filename)
        for entry in data["variants"]:
            if entry["task"] == "fold_clothes":
                variant = "random" if entry["variant"].endswith("_random") else "standard"
                for layout in entry["per_layout"]:
                    reference[(variant, layout["layout"])] = dict(**layout, split=split, source=f"https://github.com/HexaFuture/HexaShell/blob/main/results/{filename}")
    public = []
    primary = [row for row in rows if row["category"] == "primary"]
    for variant in ("standard", "random"):
        for layout in range(10):
            matches = [row for row in primary if row["variant"]==variant and row["layout"]==layout]
            assert len(matches) <= 1, (variant, layout, "duplicate primary attempts")
            public.append(dict(variant=variant, layout=layout,
                split="development" if layout < 5 else "new_layouts", reference=reference[(variant,layout)],
                execution=matches[0] if matches else None))
    complete = all(case["execution"] and case["execution"]["status"]=="native_completed" for case in public)
    report = dict(schema="roborag.hexashell-consumption.v1", updated_at=datetime.now(timezone.utc).isoformat(),
        protocol="experiments/hexashell-fold-public-suite-v2.json", public_suite_complete=complete,
        public_planned_episodes=20, public_scored_episodes=sum(row["status"]=="native_completed" for row in primary),
        public_cases=public, executions=rows, batches=batches,
        groups={name: aggregate([row for row in rows if row["category"]==name]) for name in ("primary","pre_calibration","pilot","setup")},
        primary_evidence_audit=audit_primary(primary),
        reference_split_summaries={f"{split}/{variant}": dict(
            episodes=5, successful_episodes=sum(reference[(variant,layout)]["success"] for layout in layouts),
            mean_progress=sum(reference[(variant,layout)]["progress"] for layout in layouts)/5,
            comparison_note=("Same public layout IDs, but author development scores predate recipes and notes; local runs use the final public prompt."
                if split=="development" else "Same public seed-0 new layouts, frozen tools and recipe/notes condition; private executor and model snapshot unavailable."))
            for split,layouts in (("development",range(5)),("new_layouts",range(5,10))) for variant in ("standard","random")},
        all_recorded_executions={**{key:value for key,value in aggregate(rows).items()
            if key not in ("successful_episodes", "success_rate", "mean_progress")},
            "performance_note":"Consumption includes all attempts. Performance is not pooled across primary, pre-calibration and pilot protocols."},
        account_and_orchestrator_snapshots=[dict(source=relative(path), **read(path)) for path in sorted(
            (ROOT / "runs/hexashell-accounting-20261023").glob("*.json")) if "orchestrator_goal_meter" in read(path, {})],
        split_summaries={f"{split}/{variant}": aggregate([row for row in primary if row["variant"]==variant and (row["layout"]<5)==(split=="development")]) for split in ("development","new_layouts") for variant in ("standard","random")},
        money=dict(actual_billed_usd=None,actual_billed_credits=None,
            reason="Codex account login, not API-key billing; per-experiment subscription charge is not exposed. Missing billing is not zero cost.",
            source="https://learn.chatgpt.com/docs/pricing", checked_date="2026-10-23"),
        boundaries=["Cached input is a subset of input, and reasoning is a breakdown of output; do not sum them twice.",
            "All nine completed v1 public cases, including successes, are retained as pre-calibration diagnostics after the independent marker experiment proved the camera-interface error. Their costs remain in the ledger.",
            "CLI usage sums model round trips, not unique context length. Native completed turns are not model-request counts.",
            "Old batches have no power samples: their energy is unknown, not inferred from GPU TDP.",
            "Power measures the entire GPU board including unrelated workloads; CPU and other machine power are excluded.",
            "Local CPU/GPU resource records do not measure the remote model provider's compute or energy.",
            "Concurrent memory is the sum of per-process RSS and can count shared pages more than once; it is not whole-machine physical memory usage.",
            "Orchestration goal-meter tokens and shared account limit snapshots are separate sources, not added to a claimed invoice.",
            "Author development scores predate recipes and notes. Local v2 applies the final public recipe and notes to all layouts; development results are not an identical-prompt comparison.",
            "Public development/new-layout suites cannot verify the official hidden-seed 50-episode result."])
    destination = ROOT / "results/hexashell-consumption-20261023.json"
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    csv_path = destination.with_suffix(".csv")
    fields = ["directory","category","status","variant","layout","success","progress","steps",
              "budgeted_commands","stop_reason","action_log_complete","wall_seconds","agent_wall_seconds",
              "agent_returncode","child_cpu_seconds","maximum_child_rss_kib","logical_artifact_bytes",
              *USAGE_FIELDS,"non_cached_input_tokens","terminal_usage_available"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            usage = row["accounting"]["reported_usage"]
            uncached = usage["input_tokens"]-usage["cached_input_tokens"] if usage["input_tokens"] is not None and usage["cached_input_tokens"] is not None else None
            local = row["local_resources"]
            cpu = (local["child_cpu_user_seconds"]+local["child_cpu_system_seconds"]
                   if "child_cpu_user_seconds" in local and "child_cpu_system_seconds" in local else local.get("child_cpu_seconds"))
            writer.writerow({**{key:row[key] for key in fields if key in row},**usage,
                "child_cpu_seconds":cpu,
                "maximum_child_rss_kib":local.get("maximum_child_peak_rss_kib",local.get("maximum_child_rss_kib")),
                "non_cached_input_tokens":uncached,"terminal_usage_available":row["accounting"]["terminal_usage_available"]})
    print(json.dumps(dict(public_scored=report["public_scored_episodes"],public_planned=20,
                         all_recorded_actor_usage=report["all_recorded_executions"]["known_token_lower_bounds"],
                         missing_terminal_usage=report["all_recorded_executions"]["actor_executions_without_terminal_usage"])))


if __name__ == "__main__":
    main()
