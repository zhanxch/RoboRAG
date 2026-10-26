#!/usr/bin/env python3
"""Run frozen HexaShell cloth tools using one native persistent Codex agent."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import signal
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from roborag.transports.codex_cli import infer, write_json
from roborag.hexashell_agent import prepare_workspace, run_agent

SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string"},
    "command": {"type": "string", "description": "Exactly one robo command, shell quoting allowed, no pipelines or shell operators"},
}, "required": ["reason", "command"], "additionalProperties": False}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_file(path, process, seconds):
    start = time.monotonic()
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f"Simulator exited ({process.returncode}) before {path.name}")
        if time.monotonic() - start > seconds:
            raise TimeoutError(f"Waiting for {path.name}")
        time.sleep(.2)


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", choices=["standard", "random"], default="standard")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--transport", choices=["native", "per-command"], default="native")
    parser.add_argument("--agent-python-packages", type=Path, default=ROOT / "data/hexashell-agent-python",
                        help="Python 3.10 scientific packages mounted read-only inside the native actor")
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument("--max-wall-seconds", type=float, default=7200)
    args = parser.parse_args()
    runtime = json.loads((ROOT / "local.runtime.json").read_text())
    upstream = Path(runtime["upstream"])
    hexa = ROOT / "vendor/HexaShell"
    agent_packages = args.agent_python_packages.resolve()
    package_manifest = {}
    if args.transport == "native" and not args.probe_only:
        if not agent_packages.is_dir():
            raise FileNotFoundError(f"Missing actor packages: {agent_packages}; see docs/hexashell-reproduction.md")
        for dist in importlib.metadata.distributions(path=[str(agent_packages)]):
            record = dist.read_text("RECORD")
            package_manifest[dist.metadata["Name"]] = dict(version=dist.version,
                record_sha256=hashlib.sha256(record.encode()).hexdigest() if record else None)
    sys.path.insert(0, str(upstream))
    from roboicl.paths import python_paths
    from roboicl.upstream import verify_submodules
    dependencies = verify_submodules()
    task = "fold_clothes" + ("_random" if args.variant == "random" else "")
    layout = ROOT / f"data/Assets/Eval_Layout/RoboDojo/arx_x5/{args.seed}/{task}_{args.layout}.json"
    if not layout.is_file():
        raise FileNotFoundError(layout)
    free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    if free < 11000:
        raise RuntimeError(f"Simulator needs free GPU memory; available {free} MiB")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("inbox", "outbox"):
        (output / name).mkdir()
    workspace = output
    if not args.probe_only and args.transport == "native":
        workspace = Path(tempfile.mkdtemp(prefix="hexashell-episode-"))
        prepare_workspace(workspace, hexa)
        write_json(output / "agent-workspace.json", dict(path=str(workspace)))
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", OMNI_KIT_ACCEPT_EULA="YES",
               OMNI_KIT_ALLOW_ROOT="1", CUDA_VISIBLE_DEVICES="0", ROBOICL_DATA_ROOT=str(ROOT / "data"),
               ROBOICL_RESULTS_ROOT=str(output / "native"), ROBODOJO_RUN_ID=output.name, OMP_NUM_THREADS="4",
               PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(hexa), *python_paths("sim")]),
               LD_PRELOAD=str(Path(runtime["sim_library"]) / "libstdc++.so.6"), LD_LIBRARY_PATH=runtime["sim_library"])
    command = [runtime["sim_python"], "-u", "-m", "roborag.hexashell_sim", "--output", str(output),
               "--hexashell", str(hexa), "--layout", str(args.layout), "--seed", str(args.seed),
               "--variant", args.variant, "--max-wall-seconds", str(args.max_wall_seconds),
               "--enable_cameras", "--headless",
               "--kit_args=--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera --/app/hangDetector/enabled=false --/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true"]
    if args.probe_only:
        command.append("--probe-only")
    elif args.transport == "native":
        command += ["--agent-workspace", str(workspace)]
    sources = [ROOT / "src/roborag/hexashell_adapter.py", ROOT / "src/roborag/hexashell_sim.py", Path(__file__).resolve()]
    sources += [ROOT / "src/roborag/hexashell_agent.py", ROOT / "src/roborag/transports/codex_cli.py"]
    sources += [ROOT / "src/roborag/hexashell_accounting.py"]
    sources += list((hexa / "hexashell").rglob("*.py")) + list((hexa / "tasks/fold_clothes").rglob("*.py")) + list((hexa / "tools/table_plane").rglob("*.py"))
    manifest = dict(schema="roborag.hexashell-reproduction.v1", started_at=datetime.now(timezone.utc).isoformat(),
                    arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    hexashell_commit=subprocess.check_output(["git", "-C", str(hexa), "rev-parse", "HEAD"], text=True).strip(),
                    hexashell_worktree=subprocess.check_output(["git", "-C", str(hexa), "status", "--porcelain"], text=True),
                    dependencies=dependencies, layout=str(layout), layout_sha256=digest(layout),
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                    command=command, model=args.model, effort=args.effort, native_step_limit=500,
                    agent_python=dict(executable="/usr/bin/python3", packages=package_manifest,
                                      read_only_mount="/opt/hexashell-python"),
                    observation="RGB + calibrated cameras + robot proprioception; no depth or object truth",
                    camera_intrinsics="RTX square-pixel pinhole projection from live horizontal focal/aperture and image resolution; validated by independent RGB markers",
                    transport=("single persistent native Codex CLI; native shell, view_image, history and compaction"
                               if args.transport == "native" else "diagnostic: one tool-free inference per robo command; serialized history"),
                    limitations=["Unpublished executor reconstructed from EpisodeAPI; motion helpers and task tools unmodified.",
                                 "Reasoning effort xhigh is a local choice; paper does not specify it.",
                                 "Author CLI version, private server and model provider snapshot are unavailable.",
                                 "Public seed and local assets do not reproduce hidden-seed official success rate."],
                    gpu_processes=subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).splitlines())
    write_json(output / "run.json", manifest)
    for source in sources[:6]:
        snapshot = output / "source" / source.name
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    started = time.monotonic()
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    process = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (output / "simulator.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(output / "process.json", dict(simulator_pid=process.pid, runner_pid=os.getpid()))
            wait_file(output / "ready.json", process, 300)
            if args.probe_only:
                if process.wait(timeout=180):
                    raise RuntimeError("Probe failed")
                write_json(output / "exit.json", dict(complete=True, scored=False, wall_seconds=time.monotonic()-started))
                return
            manual = (output / "INTERFACE.md").read_text()
            prompt = ("There is a piece of clothing. Fold it neatly.\n\nTASK RECIPE:\n" +
                      (hexa / "recipes/fold_clothes.md").read_text() + "\n\nTASK NOTES (from earlier runs of the same tools in other scenes):\n" +
                      (hexa / "tasks/fold_clothes/notes.md").read_text())
            history = []
            agent_result = None
            if args.transport == "native":
                agent_result = run_agent(workspace, output,
                    prompt + "\n\nRead INTERFACE.md for the robot manual and extra tools. "
                    "Use robo commands in the shell to operate the robot and view obs/*.png with your image tool. "
                    "Work only with the supplied interface, observations and notes.md. "
                    "Keep operating until the episode is over, following the manual's stopping rules.",
                    model=args.model, effort=args.effort, timeout=args.max_wall_seconds - (time.monotonic()-started),
                    python_packages=agent_packages)
                state = json.loads((output / "obs/state.json").read_text())
                if not state["episode_over"]:
                    # The published retests count a normal early agent exit as
                    # failure. Finalize that state without pretending success.
                    write_json(workspace / "inbox/99999999999999999999-agent-exit.json", dict(finish="agent_exit"))
            for decision in range(1, args.max_decisions + 1) if args.transport == "per-command" else []:
                if time.monotonic()-started > args.max_wall_seconds:
                    raise TimeoutError("Model wall-time budget exhausted; incomplete")
                state = json.loads((output / "obs/state.json").read_text())
                if state["episode_over"]:
                    break
                parts = []
                for camera in ("head", "wrist_l", "wrist_r"):
                    parts += [{"type": "input_text", "text": f"Current {camera} RGB, step {state['step']}"},
                              {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode((output / f"obs/{camera}.png").read_bytes()).decode()}]
                payload = dict(task=prompt, manual=manual, history=history, state=state, current_images=parts,
                               response_instruction="Choose exactly one documented robo command. Use measurements, inspect feedback and images, and continue until native episode_over. The command is executed by the harness. Do not invent commands or report success from your intention.")
                result, usage = infer(payload, SCHEMA, output / f"decisions/{decision:03d}", model=args.model,
                                      effort=args.effort, timeout=600)
                argv = shlex.split(result["command"])
                request = dict(argv=argv, decision=decision, reason=result["reason"])
                request_path = output / f"inbox/{decision:04d}.json"
                temporary = request_path.with_suffix(".tmp")
                write_json(temporary, request)
                temporary.replace(request_path)
                response_path = output / "outbox" / request_path.name
                wait_file(response_path, process, 600)
                receipt = json.loads(response_path.read_text())
                history.append(dict(decision=decision, **result, receipt=receipt["result"]))
                write_json(output / "progress.json", dict(decision=decision, command=result["command"],
                           step=receipt["step_after"], result=receipt["result"], usage=usage))
                print(json.dumps(dict(decision=decision, command=result["command"], step=receipt["step_after"],
                                      plan_ok=receipt["result"].get("plan_ok"), episode_over=receipt["result"].get("episode_over"))), flush=True)
                if receipt["result"].get("episode_over"):
                    break
            else:
                if args.transport == "per-command":
                    raise RuntimeError("Decision budget exhausted; episode incomplete")
            if process.wait(timeout=120):
                raise RuntimeError("Simulator failed during finalization")
            completed = json.loads((output / "sim_complete.json").read_text())
            native = json.loads(Path(completed["native_result"]).read_text())
            if native.get("eval_time") != 1 or native.get("policy_error"):
                raise RuntimeError("No complete native evaluation")
            write_json(output / "result.json", dict(complete=True, scored=True, native=native,
                       native_result=completed["native_result"], native_sha256=digest(completed["native_result"]),
                       wall_seconds=time.monotonic()-started, decisions=len(history) if agent_result is None else None,
                       native_agent=agent_result))
            print(json.dumps(native), flush=True)
    except BaseException as exc:
        write_json(output / "exit.json", dict(complete=False, scored=False, error=type(exc).__name__,
                                             detail=str(exc), wall_seconds=time.monotonic()-started))
        if process and process.poll() is None:
            write_json(workspace / "inbox/99999999999999999999-abort.json", dict(abort=True))
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        raise
    finally:
        if process:
            stop(process)
        cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        own = resource.getrusage(resource.RUSAGE_SELF)
        write_json(output / "resource_summary.json", dict(
            wall_seconds=time.monotonic()-started,
            child_cpu_user_seconds=cpu_after.ru_utime-cpu_before.ru_utime,
            child_cpu_system_seconds=cpu_after.ru_stime-cpu_before.ru_stime,
            maximum_child_peak_rss_kib=cpu_after.ru_maxrss,
            runner_cpu_seconds=own.ru_utime+own.ru_stime,
            memory_note="Linux wait/getrusage high-water mark, not the sum of simultaneous child RSS; batch sampler records concurrent RSS.",
            actual_billed_usd=None, billing_note="Existing Codex login; no per-experiment invoice available. Token counts are usage, not a dollar bill."))


if __name__ == "__main__":
    main()
