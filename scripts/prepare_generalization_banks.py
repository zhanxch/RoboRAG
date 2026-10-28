#!/usr/bin/env python3
"""Build complete per-task TRAIN banks after download and pending fold retests.

No evaluations are launched here. Existing annotation and source records remain
available for audit; a failed task is reported, never silently replaced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from roborag.episode_files import write_json
from prepare_generalization_sources import TASKS


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--after-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                  complete=False, tasks={t: {"status": "pending"} for t in TASKS})

    def save():
        write_json(args.output / "status.json", status)

    status["phase"] = "waiting_for_fold_retests"
    save()
    while True:
        parent = json.loads((args.after_batch / "status.json").read_text())
        if parent["complete"]:
            break
        launch = json.loads(args.after_batch.with_name(args.after_batch.name + "-launch.json").read_text())
        try:
            os.kill(launch["pid"], 0)
        except ProcessLookupError:
            raise RuntimeError("Fold retest driver exited without completing; inspect its evidence")
        time.sleep(15)

    status["phase"] = "building_banks"
    for task in TASKS:
        while True:
            source_status = json.loads((args.sources / "status.json").read_text())
            row = source_status["tasks"][task]
            if row["status"] in ("verified", "error"):
                break
            os.kill(source_status["pid"], 0)
            time.sleep(15)
        if row["status"] != "verified":
            status["tasks"][task] = dict(status="source_error")
            save()
            continue
        manifest = ROOT / "data/runtime-data" / task / "train/sources-0-1-2.json"
        bank = ROOT / "artifacts/generalization-v3-full" / task
        command = [sys.executable, "-u", str(ROOT / "scripts/build_graph_bank.py"),
                   "--manifest", str(manifest), "--output", str(bank),
                   "--model", "gpt-6-astra", "--effort", "xhigh", "--require-full-coverage"]
        status["tasks"][task] = dict(status="running", command=command,
                                      manifest_sha256=digest(manifest))
        save()
        with (args.output / f"{task}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status["tasks"][task].update(status="built" if result.returncode == 0 else "error",
                                      returncode=result.returncode)
        if result.returncode == 0:
            sources = json.loads(manifest.read_text())
            index = json.loads((bank / "execution-index.json").read_text())
            inventory = dict(schema="roborag.full-train-evidence.v1", task=task,
                source_manifest=str(manifest), source_manifest_sha256=digest(manifest),
                train_source_sha256={str((ROOT / "data" / r["path"]).resolve()): r["sha256"]
                                     for r in sources["records"]},
                metadata_sha256={str(p.relative_to(bank)): digest(p) for p in
                                 [*bank.glob("*.json"), *(bank / "graphs").glob("*.json")]},
                counts=index["counts"],
                scope="All preselected TRAIN episodes, complete phase coverage, original actions/states/RGB accessible; no test-usage pruning")
            write_json(bank / "source-inventory.json", inventory)
        save()
    status.update(complete=True, ended_at=datetime.now(timezone.utc).isoformat())
    save()
    if any(row["status"] != "built" for row in status["tasks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
