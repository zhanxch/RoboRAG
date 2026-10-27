"""Run the preregistered pouring suite with a shared native resource gate."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from roborag.evaluation.comparison import candidate_rows
from roborag.evaluation.episode import summarize
from roborag.evaluation.suite import suite
from roborag.runtime.provenance import protocol_sources

parser = argparse.ArgumentParser()
parser.add_argument("group", type=Path)
args = parser.parse_args()
group = args.group.resolve()
root = Path(__file__).resolve().parents[2]
protocol = json.loads((group / "protocol.json").read_text())
sources = json.loads((group / "frozen-sources.json").read_text())
cases = json.loads((root / "experiments/pour_balls_into_vase/heldout.json").read_text())
assert protocol["max_wall_seconds"] == 3600
assert [case["layout"] for case in cases] == list(range(3, 25))
state_path = group / "supervisor.json"
if state_path.exists():
    raise RuntimeError("Existing supervisor state must be reviewed before any continuation")
state = dict(
    role="frozen_heldout_evaluation",
    policy_version="rim-delta-v3",
    planned_layouts=list(range(3, 25)),
    status="running",
    results=[],
    active=[],
    supervisor_pid=os.getpid(),
    supervisor_pgid=os.getpgrp(),
    supervisor_sid=os.getsid(0),
)
active = {}
pending = list(cases)


def save():
    state["active"] = [
        dict(layout=item["case"]["layout"], pid=process.pid, pgid=os.getpgid(process.pid))
        for process, item in active.items()
        if process.poll() is None
    ]
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(state_path)


def shutdown(sig, frame):
    state.update(status="interrupted", signal=sig)
    save()
    for process in active:
        if process.poll() is None:
            os.killpg(process.pid, sig)
    for process in active:
        try:
            process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    raise SystemExit(128 + sig)


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)
(group / "supervisor-script.py").write_bytes(Path(__file__).read_bytes())
(group / "audit-script.py").write_bytes(
    (root / "experiments/pour_balls_into_vase/audit_episode.py").read_bytes()
)
(group / "driver-manifest.json").write_text(
    json.dumps(
        {
            name: hashlib.sha256((group / name).read_bytes()).hexdigest()
            for name in ("launcher-script.py", "supervisor-script.py", "audit-script.py")
        },
        indent=2,
    )
    + "\n"
)
save()
try:
    while pending or active:
        for process, item in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            item["log"].close()
            output = item["output"]
            summary = summarize(output)
            run = json.loads((output / "run.json").read_text())
            if run["source_sha256"] != sources:
                raise RuntimeError(f"Run source provenance differs: {output}")
            if run["args"]["max_wall_seconds"] != 3600:
                raise RuntimeError(f"Run wall-time protocol differs: {output}")
            audit = group / "audits" / f"layout{item['case']['layout']:02d}.json"
            audit_result = subprocess.run(
                [
                    sys.executable,
                    str(group / "audit-script.py"),
                    str(output),
                    str(audit),
                    "--policy",
                    "rim-delta-v3",
                    "--heldout",
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )
            failure = (
                json.loads((output / "exit.json").read_text())
                if (output / "exit.json").exists()
                else {}
            )
            status = (
                "native_complete"
                if summary["complete"]
                else "wall_time_budget_exhausted"
                if failure.get("error") == "TimeoutExpired"
                else "infrastructure_failure"
            )
            entry = dict(
                **item["case"],
                directory=str(output),
                launcher_pid=process.pid,
                launcher_returncode=returncode,
                complete=summary["complete"],
                native=summary["native"],
                steps=summary["native_control_steps"],
                status=status,
                audit=str(audit),
                audit_returncode=audit_result.returncode,
            )
            state["results"].append(entry)
            del active[process]
            save()
            report = dict(
                schema="roborag.task-evaluation.v1",
                task="pour_balls_into_vase",
                role="frozen_heldout_evaluation",
                protocol=str(group / "protocol.json"),
                **suite(
                    candidate_rows(
                        [row["directory"] for row in state["results"]],
                        task="pour_balls_into_vase",
                    ),
                    cases,
                ),
            )
            report_path = group / "evaluation.json"
            temporary = report_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(report_path)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            if audit_result.returncode:
                raise RuntimeError(f"Automatic audit failed: {audit_result.stderr}")
            if status == "infrastructure_failure":
                raise RuntimeError(f"Native infrastructure failure: {output}")
            time.sleep(15)
        limit = json.loads((group / "worker-limit.json").read_text())["workers"]
        if limit not in (1, 2):
            raise ValueError("Worker limit must be one or two")
        while pending and len(active) < limit:
            if protocol_sources(root / "data/graph-bank/current", root) != sources:
                raise RuntimeError("Frozen source drift; no next case launched")
            case = pending.pop(0)
            output = group / f"seed1-standard-l{case['layout']}"
            log = (group / f"layout{case['layout']:02d}-launcher.log").open("w")
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(group / "launcher-script.py"),
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                    "--layout",
                    str(case["layout"]),
                ],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[process] = dict(case=case, output=output, log=log)
            save()
        if active:
            time.sleep(2)
    state["status"] = "all_planned_cases_attempted"
    state["supervisor_returncode"] = 0
    save()
except Exception as exc:
    state.update(status="stopped_for_review", error=repr(exc), supervisor_returncode=1)
    save()
    for process in active:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    raise
