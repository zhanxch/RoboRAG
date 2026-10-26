#!/usr/bin/env python3
"""Launch one recorded RoboDojo pilot using explicitly configured local dependencies."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def ready(port, process, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.2):
                return
        except OSError:
            time.sleep(.2)
    raise TimeoutError(f"Service at {port} did not become ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="deposit_coin")
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--variant", choices=("standard", "random"), default="standard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--role", choices=("development", "validation"), default="development")
    parser.add_argument("--mode", choices=("zero", "fixed", "dual", "long_only"), default=None,
                        help="Defaults to zero for queued, dual for existing profiles")
    parser.add_argument("--profile", choices=("standard", "continuous_fast", "hybrid", "queued"), default="standard",
                        help="queued retains whole bottle-disposal segments across bounded dispatch windows")
    parser.add_argument("--effort", default=None,
                        help="Defaults to xhigh for queued, medium for existing profiles")
    parser.add_argument("--model", default="gpt-6-astra", help="Model passed to the policy and inference transport")
    parser.add_argument("--long-k", type=int, default=6)
    parser.add_argument("--short-k", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=0, help="0 uses the native task limit; otherwise 1..80 decision rounds")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true", help="Record overlap with another pilot; wall times are not comparable")
    args = parser.parse_args()
    if args.mode is None:
        args.mode = "zero" if args.profile == "queued" else "dual"
    if args.effort is None:
        args.effort = "xhigh" if args.profile == "queued" else "medium"
    if args.profile in ("continuous_fast", "hybrid", "queued") and args.task != "put_bottles_into_dustbin":
        parser.error(f"{args.profile} currently supports only --task put_bottles_into_dustbin")
    runtime = json.loads((ROOT / "local.runtime.json").read_text())
    code = Path(runtime["upstream"])
    sys.path.insert(0, str(code))
    from roboicl.paths import python_paths
    from roboicl.upstream import verify_submodules
    from roboicl.config import load_task, render_official_task_prompt

    pinned = verify_submodules()
    task = args.task + ("_random" if args.variant == "random" else "")
    layout = ROOT / "data/Assets/Eval_Layout/RoboDojo/arx_x5" / str(args.seed) / f"{task}_{args.layout}.json"
    if not layout.is_file():
        raise FileNotFoundError(layout)
    corpus = (args.corpus or ROOT / "data/corpora" / args.task / "index.json").resolve()
    if args.mode != "zero" and not args.capture_only and not corpus.is_file():
        raise FileNotFoundError(corpus)
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if busy and not args.allow_shared_gpu:
        raise RuntimeError(f"GPU already in use by compute PIDs: {busy}")
    if busy and args.allow_shared_gpu:
        free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        if free < 11000:
            raise RuntimeError(f"Insufficient free GPU memory for a second simulator: {free} MiB")
    out = args.output.resolve()
    config = json.loads((code / "configs/shared_harness.json").read_text())
    task_config = load_task(args.task)
    config.update(task_config.get("harness_overrides", {}))
    document_version, document_source, task_prompt = render_official_task_prompt(task_config)
    horizon = json.loads((code / "configs/adaptive_horizons.json").read_text())["task_horizons"][args.task]
    config.update(predict_horizon=horizon, execute_horizon=horizon,
                  official_task_prompt=task_prompt, live_anchor_count=args.short_k,
                  official_task_documentation_version=document_version,
                  official_task_documentation_source=document_source,
                  reasoning_effort=args.effort, preserve_reasoning=False, rolling_history_cache=False,
                  cache_mode="omit", max_turns=args.max_calls or None, max_seconds=7200,
                  request_timeout_s=600, max_transport_attempts=2)
    from roborag.control_profiles import apply_control_profile
    reference_horizon = (
        int(json.loads(corpus.read_text(encoding="utf-8"))["horizon"])
        if args.mode != "zero" and corpus.is_file() else horizon
    )
    profile_metadata = apply_control_profile(
        config, args.profile, task=args.task, reference_horizon=reference_horizon)
    profile_metadata["reasoning_effort"] = args.effort
    config["reasoning_effort"] = args.effort
    out.mkdir(parents=True, exist_ok=False)
    transport_port, policy_port = free_port(), free_port()
    write(out / "harness.json", config)
    from roboicl.policy.dialogue_policy import load_config
    load_config(out / "harness.json")
    base_env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
                    OMNI_KIT_ACCEPT_EULA="YES", OMNI_KIT_ALLOW_ROOT="1", CUDA_VISIBLE_DEVICES="0",
                    ROBOICL_DATA_ROOT=str(ROOT / "data"), ROBOICL_RESULTS_ROOT=str(out / "native"),
                    ROBODOJO_RUN_ID=out.name, ASTRA_HARNESS_CONFIG=str(out / "harness.json"),
                    ASTRA_CAMERA_PREFLIGHT=str(out / "camera_preflight"),
                    ASTRA_LOG_DIR=str(out / "policy"), ASTRA_REASONING_EFFORT=args.effort,
                    ASTRA_API_MODE="responses", ASTRA_ENDPOINT=f"http://127.0.0.1:{transport_port}/v1/responses",
                    ASTRA_API_KEY="local-loopback-no-api-credential", ASTRA_TRACKING_GUARD="1",
                    ASTRA_VALIDATE_RGB="1", ASTRA_NETWORK_MODE="direct", ASTRA_STREAM="0",
                    ASTRA_RETRIEVABLE_HISTORY="0", ROBORAG_MODE=args.mode,
                    ROBORAG_CONTROL_PROFILE=args.profile,
                    ROBORAG_CORPUS=str(corpus), ROBORAG_LONG_K=str(args.long_k),
                    ROBORAG_SHORT_K=str(args.short_k), OMP_NUM_THREADS="4")
    base_env.pop("ASTRA_DEMO_MANIFEST", None)
    policy_env = dict(base_env, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), *python_paths("policy")]))
    sim_env = dict(base_env, PYTHONPATH=os.pathsep.join(python_paths("sim")),
                   LD_PRELOAD=str(Path(runtime["sim_library"]) / "libstdc++.so.6"),
                   LD_LIBRARY_PATH=runtime["sim_library"])
    if args.profile == "queued":
        # Upstream rollout counts get_action windows; the policy counts actual
        # model turns. A queued segment can need several windows for one turn.
        # Keep the model budget in the policy, and the native step limit in sim.
        sim_config = dict(config, max_turns=None)
        write(out / "sim-harness.json", sim_config)
        load_config(out / "sim-harness.json")
        sim_env["ASTRA_HARNESS_CONFIG"] = str(out / "sim-harness.json")
        profile_metadata["model_turn_budget_owner"] = "policy"
    transport = [runtime["policy_python"], "-m", "roborag.transports.responses_bridge",
                 "--port", str(transport_port), "--output", str(out / "decisions")]
    policy = [runtime["policy_python"], "-u", "-m", "roborag.robodojo_policy", "--port", str(policy_port),
              "--task", task, "--seed", str(args.seed), "--reasoning-effort", args.effort,
              "--model", args.model]
    simulator = [runtime["sim_python"], "-u", "-m", "roboicl.robodojo", "--task", task,
                 "--seed", str(args.seed), "--layout", str(args.layout), "--data-root", str(ROOT / "data"),
                 "--results-root", str(out / "native"), "--port", str(policy_port), "--enable_cameras", "--headless",
                 "--kit_args=--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera --/app/hangDetector/enabled=false --/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true"]
    if args.capture_only:
        simulator.append("--capture-only")
    write(out / "run.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "roborag_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "roborag_worktree_changes": subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).splitlines(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "control_profile": profile_metadata,
        "harness_files": {"policy": str(out / "harness.json"),
                          "simulator": sim_env["ASTRA_HARNESS_CONFIG"]},
        "dependencies": pinned, "roboicl_commit": subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip(),
        "layout": str(layout), "layout_sha256": hashlib.sha256(layout.read_bytes()).hexdigest(),
        "assets": runtime["assets"], "asset_audit": "reused local assets; not a new complete official-revision audit",
        "commands": {"transport": transport, "policy": policy, "simulator": simulator},
        "model": args.model, "transport": "Codex-account adaptation, not original paper API",
        "evaluation_role": args.role, "capture_only": args.capture_only,
        "overlapping_compute_pids_at_launch": busy.splitlines(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in (ROOT / "src", ROOT / "scripts") for p in folder.rglob("*.py")},
    })
    processes = []
    logs = []
    started = time.monotonic()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        commands = [("simulator", simulator, sim_env)] if args.capture_only else [
            ("transport", transport, policy_env), ("policy", policy, policy_env), ("simulator", simulator, sim_env)]
        for name, command, environment in commands:
            log = (out / f"{name}.log").open("w")
            logs.append(log)
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            write(out / "processes.json", [{"pid": p.pid, "returncode": p.poll()} for p in processes])
            if name == "transport":
                ready(transport_port, process)
            elif name == "policy":
                ready(policy_port, process, timeout=300)
        rc = processes[-1].wait()
        if rc == 0:
            if args.capture_only:
                if not list((out / "camera_preflight").rglob("capture_probe.json")):
                    raise RuntimeError("Simulator returned zero without camera capture evidence")
            else:
                results = list((out / "native").rglob("_result.json"))
                result = json.loads(results[0].read_text()) if len(results) == 1 else None
                if (not result or result.get("eval_time") != 1 or result.get("policy_error")
                        or list((out / "native").rglob("_infrastructure_failure.json"))):
                    raise RuntimeError("Simulator returned zero without one complete native evaluation")
        write(out / "exit.json", {"returncode": rc, "wall_seconds": time.monotonic() - started})
        return rc
    except BaseException as error:
        write(out / "exit.json", {"returncode": None, "wall_seconds": time.monotonic() - started,
                                 "failure_kind": "infrastructure", "error": str(error)})
        raise
    finally:
        for process in reversed(processes):
            stop(process)
        for log in logs:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
