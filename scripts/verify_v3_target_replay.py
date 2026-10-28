"""Replay archived target compilation without model calls or robot movement."""

import hashlib
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from roborag.action_targets import compile_targets
from roborag.rag_execution import expand_delta_actions

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / "fold-v3-frozen-baseline/src/roborag/action_targets.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def events(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def main():
    spec = importlib.util.spec_from_file_location("roborag._original_action_targets", ORIGINAL)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    audit_path = ROOT / "results/v3-tool-error-audit-20261028.json"
    cohorts, total = {}, Counter()
    for name, cohort in read(audit_path)["cohorts"].items():
        cases, fixes = [], []
        root = Path(cohort["directory"])
        for directory in sorted(p for p in root.glob("seed1-*-l*") if p.is_dir()):
            counts, state_files = Counter(), []
            initial = directory / "observations/000_initial/state.json"
            if initial.exists():
                state = read(initial)
                state_files.append(dict(path=str(initial.relative_to(directory)), sha256=digest(initial)))
            commands = directory / "commands.jsonl"
            if commands.exists():
                for command in events(commands):
                    assert command["step_before"] == state["step"], directory
                    request = command["request"]
                    if "actions" in request:
                        encoding = request.get("action_encoding", {})
                        if "targets" in encoding:
                            rows = compile_targets(state["action_state"], encoding["targets"])
                            assert rows == original.compile_targets(state["action_state"], encoding["targets"])
                            counts["unchanged_target_calls"] += 1
                        else:
                            rows = expand_delta_actions(segments=encoding["segments"]) if "segments" in encoding else expand_delta_actions(actions=request["actions"])
                            counts["unchanged_explicit_delta_calls"] += 1
                        assert rows == request["actions"], (directory, counts)
                        counts["unchanged_submitted_rows"] += len(rows)
                    snapshot = directory / "observations" / Path(command["observation"]).name / "state.json"
                    state = read(snapshot)
                    state_files.append(dict(path=str(snapshot.relative_to(directory)), sha256=digest(snapshot)))
            failures = [f for f in cohort["failed_calls"] if f["case"] == directory.name and f["tool"] == "execute_chunk"]
            recovered = []
            if failures:
                state = read(initial)
                log = directory / "agent-events.jsonl"
                assert digest(log) == cohort["event_log_sha256"][str(log)]
                for event in events(log):
                    item = event.get("item") or {}
                    if event.get("type") != "item.completed":
                        continue
                    if item.get("tool") == "execute_chunk" and item.get("status") == "failed":
                        args = item["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)
                        expected = failures[len(recovered)]
                        assert args == expected["arguments"], directory
                        try:
                            original.compile_targets(state["action_state"], args["targets"])
                        except ValueError as error:
                            original_error = str(error)
                        else:
                            raise AssertionError((directory, "original failure did not reproduce"))
                        try:
                            compile_targets(state["action_state"], args["targets"])
                        except ValueError as error:
                            assert str(error) == original_error
                        else:
                            raise AssertionError("Restored v3 accepted an originally invalid target")
                        recovered.append(dict(case=directory.name, category=expected["category"],
                            original_error=original_error, observed_at_step=state["step"],
                            restored_v3_rejection=True))
                    for block in (item.get("result") or {}).get("content", []):
                        if block.get("type") != "text":
                            continue
                        try:
                            packet = json.loads(block["text"])
                        except (ValueError, TypeError):
                            continue
                        if isinstance(packet, dict) and "current_observation" in packet:
                            state = packet["current_observation"]
                assert len(recovered) == len(failures), directory
                fixes.extend(recovered)
            total.update(counts)
            cases.append(dict(case=directory.name, counts=dict(counts),
                commands_sha256=digest(commands) if commands.exists() else None,
                input_states_sha256=hashlib.sha256(json.dumps(state_files, sort_keys=True).encode()).hexdigest()))
        assert len(cases) == 50, name
        cohorts[name] = dict(cases=cases, verified_rejections=fixes,
                            rejection_categories=dict(Counter(f["category"] for f in fixes)))
    assert sum(len(c["verified_rejections"]) for c in cohorts.values()) == 18
    report = dict(schema="roborag.v3-target-replay.v1", passed=True,
        generated_at=datetime.now(timezone.utc).isoformat(), totals=dict(total), cohorts=cohorts,
        sources_sha256={str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), ROOT / "src/roborag/action_targets.py", ROOT / "src/roborag/rag_execution.py", audit_path)},
        original_source_sha256=digest(ORIGINAL),
        scope="Offline compiler replay only. Existing valid submissions are exactly unchanged; previously rejected targets are again rejected exactly as v3. No new model calls or robot movement, and no claim that native folding outcomes improve.")
    output = ROOT / "results/v3-execution-aligned-replay-20261028.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(output=str(output), passed=True, cases=150, verified_rejections=18, **total)))


if __name__ == "__main__":
    main()
