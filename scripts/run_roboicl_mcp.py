#!/usr/bin/env python3
"""Run a frozen RoboICL policy over native RoboDojo MCP, with historical controls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-plan", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--proxy", help="Optional HTTP proxy for the logged-in model CLI")
    args = parser.parse_args()
    history = json.loads(args.historical_plan.read_text())
    runtime = json.loads(args.runtime.read_text())
    baseline = Path(history["baseline_run"])
    upstream = Path(history["upstream"])
    changed = [p for p, digest in history["source_sha256"].items() if not Path(p).is_file() or sha(Path(p)) != digest]
    if changed:
        raise ValueError(f"Historical source changed; review comparability before running: {changed}")
    native_command = json.loads((baseline / "run.json").read_text())["command"]
    data = Path(native_command[native_command.index("--data-root") + 1])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source"
    shutil.copytree(ROOT / "src/roborag", snapshot / "roborag", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(upstream / "roboicl", snapshot / "roboicl", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(upstream / "configs", snapshot / "configs")
    (snapshot / "third_party").symlink_to(upstream / "third_party", target_is_directory=True)
    sys.path.insert(0, str(snapshot))
    from roboicl.paths import python_paths
    sys.path[:] = [str(snapshot), *python_paths("policy"), *sys.path]
    from roborag.cap.mcp_client import MCPClient
    from roborag.cap.roboicl import run_episode
    from roborag.transports.codex_cli import materialize_images
    harness = dict(history["harness"], reasoning_effort=args.effort)
    save(out / "harness.json", harness)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    policy_env = dict(PYTHONPATH=str(snapshot), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
        ASTRA_API_KEY="local-loopback-transport-no-api-credential", ASTRA_NETWORK_MODE="direct", ASTRA_STREAM="1",
        ASTRA_HARNESS_CONFIG=str(out / "harness.json"), ASTRA_DEMO_MANIFEST=history["reference"],
        ASTRA_LOG_DIR=str(out / "policy"), ASTRA_REASONING_EFFORT=args.effort, ASTRA_RETRIEVABLE_HISTORY="0",
        ASTRA_API_MODE="responses", ROBOICL_DATA_ROOT=str(data), NO_PROXY="127.0.0.1,localhost")
    if args.proxy:
        policy_env.update(HTTP_PROXY=args.proxy, HTTPS_PROXY=args.proxy, ALL_PROXY=args.proxy)
    os.environ.pop("ASTRA_ENDPOINT", None)
    os.environ.update(policy_env)
    worker_env = dict(policy_env, PYTHONPATH=os.pathsep.join([str(snapshot), *python_paths("sim")]),
        ASTRA_VALIDATE_RGB="1", OMNI_KIT_ACCEPT_EULA="YES", OMNI_KIT_ALLOW_ROOT="1", OMP_NUM_THREADS="4",
        LD_PRELOAD=str(Path(runtime["sim_library"]) / "libstdc++.so.6"), LD_LIBRARY_PATH=runtime["sim_library"])
    if Path("/etc/vulkan/icd.d/nvidia_icd.json").exists():
        worker_env.update(VK_DRIVER_FILES="/etc/vulkan/icd.d/nvidia_icd.json", VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json")
    worker = [runtime["sim_python"], "-u", "-m", "roborag.devices.roboicl_worker", "--task", history["task"],
        "--layout", str(history["layout"]), "--seed", str(history["seed"]), "--data-root", str(data),
        "--output", str(out / "native"), "--enable_cameras", "--headless",
        "--kit_args=--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera --/app/hangDetector/enabled=false --/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true"]
    save(out / "worker.json", {"command": worker, "env": worker_env, "timeout": 240})
    server = ["-m", "roborag.devices.robodojo_server", "--worker-config", str(out / "worker.json")]
    save(out / "mcp-server.json", {"command": sys.executable, "args": server, "env": {"PYTHONPATH": str(snapshot)}})
    layout = data / "Assets/Eval_Layout/RoboDojo/arx_x5" / str(history["seed"]) / f"{history['task']}_{history['layout']}.json"
    save(out / "frozen-plan.json", {"schema": "roborag.roboicl-mcp-trial.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "effort": args.effort, "task": history["task"], "layout": history["layout"], "seed": history["seed"],
        "mode": "preflight" if args.preflight_only else "model_evaluation", "baseline_run": str(baseline),
        "upstream_commit": history["upstream_commit"], "historical_source_matches": True,
        "layout_sha256": sha(layout), "reference": history["reference"], "reference_sha256": sha(Path(history["reference"])),
        "harness": harness, "transport": "standard MCP stdio for robot I/O; logged-in Codex CLI for inference",
        "policy": "Frozen RoboICL dialogue policy; no CaP program generation, no RAG",
        "source_sha256": {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob("*.py") if "third_party" not in p.parts},
        "launcher_sha256": sha(Path(__file__)), "launch_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "limitations": history["limitations"] + ["Historical comparison, not a simultaneous randomized trial."]})

    def check_initial(payload):
        folder = out / "initial-context"
        folder.mkdir()
        rendered, images = materialize_images({k: payload[k] for k in ("instructions", "tools", "input")}, folder)
        save(folder / "input.json", rendered)
        original = baseline / "decisions/call-0001"
        old = json.loads((original / "input.json").read_text())
        old_images = json.loads((original / "invocation.json").read_text())["images"]
        checks = {"instructions_identical": old["instructions"] == rendered["instructions"],
                  "tools_identical": old["tools"] == rendered["tools"], "serialized_input_identical": old == rendered,
                  "image_count": len(images), "historical_image_count": len(old_images),
                  "train_images_identical": len(images) == len(old_images) == 25 and
                      all(a["sha256"] == b["sha256"] for a, b in zip(images[:24], old_images[:24])),
                  "images": images}
        save(out / "initial-context-comparison.json", checks)
        if not all(checks[k] for k in ("instructions_identical", "tools_identical", "train_images_identical")):
            raise ValueError("Frozen TRAIN/policy context mismatch; no model request sent")

    bridge = None
    summary = {"status": "failed", "native_score": None}
    started = time.monotonic()
    def interrupted(signum, frame): raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (out / "transport.log").open("w") as transport_log, (out / "simulator.log").open("w") as sim_log:
            if not args.preflight_only:
                bridge = subprocess.Popen([sys.executable, "-m", "roborag.transports.responses_bridge", "--port", str(port),
                    "--output", str(out / "decisions")], env=dict(os.environ), stdout=transport_log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                deadline = time.monotonic() + 30
                while True:
                    if bridge.poll() is not None: raise RuntimeError("model bridge exited")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=.2): break
                    except OSError:
                        if time.monotonic() > deadline: raise TimeoutError("model bridge startup")
                        time.sleep(.1)
            from roboicl.policy.dialogue_policy import Model, load_config
            policy = Model({"model": args.model, "api_mode": "responses", "endpoint": f"http://127.0.0.1:{port}/v1/responses"})
            with MCPClient(sys.executable, server, env={"PYTHONPATH": str(snapshot)}, timeout=300, errlog=sim_log) as mcp:
                save(out / "capabilities.json", mcp.catalog)
                summary = run_episode(mcp, policy, load_config(out / "harness.json"), out,
                    check_initial=check_initial, preflight_only=args.preflight_only)
                mcp.call_tool("stop_motion")
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop(bridge)
        summary["wall_seconds"] = time.monotonic() - started
        save(out / "exit.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
