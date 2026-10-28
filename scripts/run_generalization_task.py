#!/usr/bin/env python3
"""Launch one frozen 50-scene Generalization task exactly once.

Uses the unchanged episode_batch scheduler and v3 action/observation tools.
Adds complete source coverage to the scheduler's existing drift check.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from roborag import episode_batch
from roborag.episode_files import write_json
from roborag.rag_tasks import prompt_for


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag:
        parser.error("tag must be a directory name")
    protocol_path = ROOT / "experiments/generalization/remaining-v3-20261029.json"
    protocol = json.loads(protocol_path.read_text())
    config = next(row for row in protocol["tasks"] if row["task"] == args.task)
    out = ROOT / "runs" / args.tag
    launch = out.with_name(out.name + "-launch.json")
    freeze = out.with_name(out.name + "-freeze.json")
    if any(path.exists() for path in (out, launch, freeze)):
        raise RuntimeError("This attempt already exists; inspect its process and evidence, never restart it")
    dirty = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("Commit candidate source changes before freezing native tests")
    bank = ROOT / config["bank"]
    preflight_path = ROOT / "results" / f"generalization-{args.task}-bank-preflight.json"
    preflight = json.loads(preflight_path.read_text())
    assert preflight["passed"] and preflight["task"] == args.task
    assert Path(preflight["bank"]).resolve() == bank.resolve()
    for path, sha in {**preflight["metadata_sha256"], **preflight["source_sha256"]}.items():
        assert digest(path) == sha, path
    cases_path = ROOT / config["cases_file"]
    cases = episode_batch.normalize_cases(json.loads(cases_path.read_text()), 1)
    assert len(cases) == 50
    native = ROOT / "vendor/RoboICL/third_party/RoboDojo"
    # Static native source/configuration is included; outputs and model secrets
    # are never part of the source inventory or policy input.
    paths = [protocol_path, cases_path, preflight_path, ROOT / "local.runtime.json"]
    paths += list((ROOT / "scripts").glob("*.py"))
    paths += list((native / "task/RoboDojo").rglob("*.py"))
    paths += list((native / "task/RoboDojo/config").glob("*.yml"))
    paths += list((native / "env_cfg").rglob("*.yml"))
    paths += list((native / "src/eval_client").rglob("*.py"))
    paths += [Path(os.environ["COMPARE_CODEX_BIN"])]
    paths += [ROOT / "data/Assets/Eval_Layout/RoboDojo/arx_x5" / str(c["seed"]) /
              f"{args.task}{'_random' if c['variant'] == 'random' else ''}_{c['layout']}.json" for c in cases]
    for graph_path in (bank / "graphs").glob("*.json"):
        graph = json.loads(graph_path.read_text())
        source = (ROOT / "data" / graph["source"]["path"]).resolve()
        assert digest(source) == graph["source"]["sha256"]
        paths.append(source)
    extra = sorted({p.resolve() for p in paths})
    original_sources = episode_batch.protocol_sources

    def sources(directory):
        return {**original_sources(directory), **{str(path):digest(path) for path in extra}}

    frozen = sources(bank)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    launch.parent.mkdir(parents=True, exist_ok=True)
    with launch.open("x") as handle:
        json.dump(dict(pid=os.getpid(), status="running", task=args.task, candidate_commit=commit,
                       reported_count=0, created_at=datetime.now(timezone.utc).isoformat()), handle, indent=2)
    write_json(freeze, dict(candidate_commit=commit, protocol=str(protocol_path),
        source_sha256=frozen, conditions=protocol["conditions"], task=config,
        prompt_sha256=hashlib.sha256(prompt_for(ROOT,args.task,"delta").encode()).hexdigest()))
    episode_batch.protocol_sources = sources
    conditions = protocol["conditions"]
    sys.argv = ["episode_batch", "--task", args.task, "--output", str(out),
        "--cases-file", str(cases_path), "--bank", str(bank), "--controller", "delta",
        "--parallel", str(conditions["parallel"]), "--model", conditions["model"],
        "--effort", conditions["effort"], "--max-wall-seconds", str(conditions["max_wall_seconds"])]
    episode_batch.main()


if __name__ == "__main__":
    main()
