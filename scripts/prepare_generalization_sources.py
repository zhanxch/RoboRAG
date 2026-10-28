#!/usr/bin/env python3
"""Fetch the preselected TRAIN episodes for every missing Generalization task.

This only prepares demonstration inputs. It never launches an evaluation or
selects episodes using test outcomes. Per-task download logs are preserved.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "stack_bowls", "pack_objects_into_box", "hang_mugs", "sweep_blocks",
    "pour_liquid_into_cup", "make_toast", "arrange_largest_number",
    "sort_nesting_dolls_by_size", "store_laptop_and_headphones",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                  episodes=[0, 1, 2], tasks={task: {"status": "pending"} for task in TASKS},
                  complete=False)
    lock = threading.Lock()

    def update(task=None, **fields):
        with lock:
            if task:
                status["tasks"][task].update(fields)
            else:
                status.update(fields)
            temporary = output / "status.tmp"
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(output / "status.json")

    def fetch(task):
        command = [sys.executable, str(ROOT / "scripts/fetch_demonstrations.py"),
                   "--task", task, "--episodes", "0,1,2"]
        update(task, status="running", command=command)
        try:
            with (output / f"{task}.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=6000)
            update(task, status="verified" if result.returncode == 0 else "error",
                   returncode=result.returncode)
        except Exception as exc:
            update(task, status="error", error=repr(exc))

    update()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, TASKS))
    update(complete=True, ended_at=datetime.now(timezone.utc).isoformat())
    if any(row["status"] != "verified" for row in status["tasks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
