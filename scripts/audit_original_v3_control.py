"""Audit the registered original-v3 control, preserving unscored interruptions."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from roborag import episode_batch as batch

    out = ROOT / "runs/v3-original-control-20261028"
    protocol = read(ROOT / "experiments/fold_clothes/v3-original-control-20261028.json")
    freeze = read(out.with_suffix(".freeze.json"))
    assert sha(ROOT / "experiments/fold_clothes/v3-original-control-20261028.json") == freeze["protocol_sha256"]
    assert sha(ROOT / protocol["cases_file"]) == protocol["cases_sha256"]
    status = read(out / "status.json")
    batch.ROOT = Path(protocol["workspace"])
    assert batch.protocol_sources(Path(freeze["conditions"]["bank"])) == freeze["source_sha256"]
    declared = read(ROOT / protocol["cases_file"])
    keys = [(r["seed"], r["variant"], r["layout"]) for r in status["finished"]]
    expected = {(r["seed"], r["variant"], r["layout"]) for r in declared}
    assert len(declared) == len(expected) == protocol["cases_count"]
    assert len(set(keys)) == len(keys) and set(keys) <= expected
    reports = []
    for entry in status["finished"]:
        path = Path(entry["output"])
        config = read(path / "run.json")
        assert all(config["args"][key] == entry[key] for key in ("seed", "variant", "layout"))
        original = Path(protocol["baseline"]) / path.name
        before = read(original / "run.json")
        assert config["git_commit"] == freeze["candidate_commit"]
        assert config["source_sha256"] == before["source_sha256"]
        for source, digest in config["source_sha256"].items():
            assert sha(Path(source)) == digest, source
            assert freeze["source_sha256"][str(Path(source).resolve())] == digest, source
        for key in ("task", "variant", "layout", "seed", "model", "effort",
                    "max_wall_seconds", "no_rag", "controller", "bank"):
            assert config["args"][key] == before["args"][key], key
        assert config["layout_sha256"] == before["layout_sha256"]
        assert config["prompt_sha256"] == before["prompt_sha256"]
        assert sha(path / "agent-prompt.txt") == config["prompt_sha256"]
        report = dict(
            case=path.name, source_and_prompt_verified=True,
            original_v3_conditions_match=True,
            source_files_verified=len(config["source_sha256"]),
            original_success=read(original / "result.json")["native"]["success_rate"] == 1,
            attempt_wall_seconds=entry["ended_epoch"] - entry["started_epoch"],
        )
        result_path = path / "result.json"
        if result_path.exists():
            result = read(result_path)
            native = Path(result["native_result"])
            assert result["complete"] and entry["returncode"] == 0
            assert sha(native) == result["native_sha256"]
            assert read(native) == result["native"]
            assert result["native"]["eval_time"] == 1 and not result["native"].get("policy_error")
            required = ("run.json", "agent-prompt.txt", "agent-events.jsonl", "agent-exit.json",
                        "commands.jsonl", "actions.jsonl", "mcp-calls.jsonl", "gpt-plan.json",
                        "result.json", "episode_end.json")
            assert all((path / name).is_file() for name in required), path
            commands = [json.loads(line) for line in (path / "commands.jsonl").read_text().splitlines()]
            for command in commands:
                observation = Path(command["observation"])
                assert all((observation / name).is_file()
                           for name in ("state.json", "head.png", "wrist_l.png", "wrist_r.png"))
            report.update(status="scored", native_score=result["native"]["score"],
                          success=result["native"]["success_rate"] == 1,
                          control_steps=result["native_control_steps"],
                          wall_seconds=result["wall_seconds"],
                          native_sha256=result["native_sha256"], native_hash_verified=True,
                          command_records=len(commands))
        else:
            # Keep this branch strict: other interruption classes require their
            # own evidence review rather than silently accepting every crash.
            failure = read(path / "exit.json")
            agent_exit = read(path / "agent-exit.json")
            events = [json.loads(line) for line in (path / "agent-events.jsonl").read_text().splitlines()]
            assert entry["returncode"] != 0 and not failure["complete"]
            assert not list(path.rglob("_result.json")), "Unaccounted native score"
            assert agent_exit["returncode"] == 1 and agent_exit["completed_turns"] == 0
            assert any(event.get("type") == "turn.failed"
                       and event.get("error", {}).get("message")
                       == "Selected model is at capacity. Please try a different model."
                       for event in events)
            assert (path / "actions.jsonl").stat().st_size == 0
            assert read(path / "ready.json")["step"] == 0
            report.update(status="unscored", native_score=None, success=None,
                          classification="model_capacity_before_execution",
                          control_steps_lower_bound=0, wall_seconds=failure["wall_seconds"],
                          model_usage=agent_exit["usage"], failure=failure,
                          evidence_sha256={name: sha(path / name) for name in
                                           ("exit.json", "agent-exit.json", "agent-events.jsonl",
                                            "agent-stderr.log", "actions.jsonl", "ready.json")})
        inventory = [dict(path=str(p.relative_to(path)), bytes=p.stat().st_size)
                     for p in sorted(path.rglob("*")) if p.is_file()]
        report["artifacts"] = dict(
            passed=True, files=len(inventory), bytes=sum(p["bytes"] for p in inventory),
            path_size_sha256=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
            scope="Required scored-trajectory or unscored-interruption evidence exists; inventory covers paths and sizes, not physical semantics.",
        )
        reports.append(report)
    counts = dict(attempts_closed=len(reports), scored=sum(r["status"] == "scored" for r in reports),
                  unscored=sum(r["status"] == "unscored" for r in reports),
                  successes=sum(r["success"] is True for r in reports),
                  scored_failures=sum(r["success"] is False for r in reports))
    audit = dict(schema="roborag.original-v3-control-audit.v1", batch_complete=status["complete"],
                 counts=counts, cases=reports)
    destination = out.with_suffix(".completed-audit.json")
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(dict(**counts, latest=reports[-1] if reports else None), ensure_ascii=False))


if __name__ == "__main__":
    main()
