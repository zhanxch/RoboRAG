#!/usr/bin/env python3
"""One native robot episode: MCP evidence -> GPT plan -> GPT action chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from roborag.config import PROJECT_ROOT as ROOT
from roborag.config import RuntimeConfig
from roborag.evaluation.episode import control_accounting, workflow_completed
from roborag.io import read_jsonl
from roborag.io import write_json as save
from roborag.runtime.agent import run_agent
from roborag.runtime.provenance import protocol_sources
from roborag.tasks import TASKS, prompt_for, simulation_task, task_spec


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_file(path, process, seconds=300):
    started = time.monotonic()
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f"Process exited {process.returncode} before {path}")
        if time.monotonic() - started > seconds:
            raise TimeoutError(str(path))
        time.sleep(0.2)


def stop(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Unscored initialization check; no model or policy actions",
    )
    parser.add_argument("--variant", choices=["standard", "random"], default="standard")
    parser.add_argument("--bank", type=Path, default=ROOT / "data/graph-bank/current")
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--max-wall-seconds", type=float, default=5400)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--controller", choices=["delta"], default="delta")
    parser.add_argument("--architecture", choices=["legacy", "staged-v1"], default="legacy")
    args = parser.parse_args()
    task = simulation_task(args.task, args.variant)
    spec = task_spec(args.task)
    prompt = prompt_for(
        ROOT, args.task, args.controller, no_rag=args.no_rag, architecture=args.architecture
    )
    runtime = RuntimeConfig.load(ROOT).as_dict()
    sys.path.insert(0, runtime["upstream"])
    from roboicl.paths import python_paths

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = Path(tempfile.mkdtemp(prefix="roborag-mcp-"))
    for name in ("bin", "obs", "inbox", "outbox"):
        (workspace / name).mkdir()
    (workspace / "bin/python").symlink_to("/usr/bin/python3")
    save(output / "agent-workspace.json", {"path": str(workspace)})
    hexa = ROOT / "vendor/HexaShell"
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
        OMNI_KIT_ACCEPT_EULA="YES",
        OMNI_KIT_ALLOW_ROOT="1",
        CUDA_VISIBLE_DEVICES="0",
        ROBOICL_DATA_ROOT=str(ROOT / "data"),
        ROBOICL_RESULTS_ROOT=str(output / "native"),
        ROBODOJO_RUN_ID=output.name,
        OMP_NUM_THREADS="4",
        PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(hexa), *python_paths("sim")]),
        LD_PRELOAD=str(Path(runtime["sim_library"]) / "libstdc++.so.6"),
        LD_LIBRARY_PATH=runtime["sim_library"],
    )
    sim_command = [
        runtime["sim_python"],
        "-u",
        "-m",
        "roborag.backends.simulation",
        "--task",
        task,
        "--output",
        str(output),
        "--agent-workspace",
        str(workspace),
        "--hexashell",
        str(hexa),
        "--layout",
        str(args.layout),
        "--seed",
        str(args.seed),
        "--max-wall-seconds",
        str(args.max_wall_seconds),
        "--enable_cameras",
        "--headless",
        "--kit_args=--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera --/app/hangDetector/enabled=false --/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true",
    ]
    if args.probe_only:
        sim_command.append("--probe-only")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mcp_env = dict(
        os.environ, PYTHONPATH=str(ROOT / ".runtime/mcp") + os.pathsep + str(ROOT / "src")
    )
    mcp_command = [
        runtime["policy_python"],
        "-m",
        "roborag.mcp.server",
        "--bank",
        str(args.bank.resolve()),
        "--data",
        str(ROOT / "data"),
        "--workspace",
        str(workspace),
        "--output",
        str(output),
        "--port",
        str(port),
        "--controller",
        args.controller,
        "--task",
        args.task,
        "--architecture",
        args.architecture,
    ] + (["--no-rag"] if args.no_rag else [])
    save(
        output / "run.json",
        dict(
            started_at=datetime.now(timezone.utc).isoformat(),
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            schema="roborag.mcp-pilot.v2",
            task=task,
            model=args.model,
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            source_sha256=protocol_sources(args.bank.resolve()),
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            layout_sha256=digest(
                ROOT
                / f"data/Assets/Eval_Layout/RoboDojo/arx_x5/{args.seed}/{task}_{args.layout}.json"
            ),
            observation="current RGB + camera calibration + robot proprioception; no object ground truth",
            execution_guard=(
                {
                    "kind": "native_ik_continuity",
                    "max_joint_increment_rad": spec.ik_joint_increment_rad,
                }
                if spec.ik_joint_increment_rad is not None
                else None
            ),
            controller="RoboICL world link6 delta grammar; native RoboDojo EE control",
            workflow=(
                "Stage SOP + history -> persistent GPT graph plan -> adaptive stage execution -> offline native evaluator"
                if args.architecture == "staged-v1"
                else "MCP retrieval -> GPT new plan -> GPT chunks -> native evaluator"
            ),
        ),
    )
    processes, started = [], time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with (output / "simulator.log").open("w") as log:
            sim = subprocess.Popen(
                sim_command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(sim)
        save(output / "process.json", dict(simulator_pid=sim.pid, runner_pid=os.getpid()))
        wait_file(output / "ready.json", sim)
        if args.probe_only:
            if sim.wait(timeout=120):
                raise RuntimeError("Initialization probe finalization failed")
            probe = json.loads((output / "initialization-probe.json").read_text())
            if probe["policy_action_steps"] != 0:
                raise RuntimeError("Initialization probe unexpectedly dispatched policy actions")
            save(
                output / "probe-result.json",
                dict(
                    complete=True,
                    scored=False,
                    role="infrastructure_probe",
                    probe=probe,
                    wall_seconds=time.monotonic() - started,
                ),
            )
            print(
                json.dumps(dict(initialization_complete=True, scored=False, task=task)), flush=True
            )
            return
        with (output / "mcp-server.log").open("w") as log:
            mcp = subprocess.Popen(
                mcp_command,
                cwd=ROOT,
                env=mcp_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(mcp)
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if mcp.poll() is not None:
                    raise RuntimeError("MCP server exited")
                time.sleep(0.1)
        else:
            raise TimeoutError("MCP server startup")
        agent = run_agent(
            workspace,
            output,
            prompt,
            model=args.model,
            effort=args.effort,
            timeout=args.max_wall_seconds - (time.monotonic() - started),
            python_packages=ROOT / "data/hexashell-agent-python",
            mcp_url=f"http://127.0.0.1:{port}/mcp",
        )
        if not json.loads((workspace / "obs/state.json").read_text())["episode_over"]:
            save(workspace / "inbox/99999999999999999999-agent-exit.json", {"finish": "agent_exit"})
        if sim.wait(timeout=120):
            raise RuntimeError("Simulator finalization failed")
        completed = json.loads((output / "sim_complete.json").read_text())
        native = json.loads(Path(completed["native_result"]).read_text())
        if native.get("eval_time") != 1 or native.get("policy_error"):
            raise RuntimeError("Incomplete native evaluation")
        calls, malformed_calls = read_jsonl(output / "mcp-calls.jsonl")
        names = [c["tool"] for c in calls]
        result = dict(
            complete=True,
            native=native,
            native_result=completed["native_result"],
            native_sha256=digest(completed["native_result"]),
            native_agent=agent,
            wall_seconds=time.monotonic() - started,
            actual_billed_usd=None,
            retrieval_before_plan_before_execution=workflow_completed(calls),
            **control_accounting(output, calls),
            malformed_mcp_records=malformed_calls,
            tool_counts={name: names.count(name) for name in sorted(set(names))},
            caveat="Single native episode; evaluation role is defined by the declared suite, not inferred from success.",
        )
        save(output / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as exc:
        save(
            output / "exit.json",
            dict(
                complete=False,
                error=type(exc).__name__,
                detail=str(exc),
                wall_seconds=time.monotonic() - started,
            ),
        )
        raise
    finally:
        for process in reversed(processes):
            stop(process)


if __name__ == "__main__":
    main()
