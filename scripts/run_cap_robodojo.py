#!/usr/bin/env python3
"""Run the native RoboDojo MCP server and the same generic CaP harness."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SmokeBrain:
    metadata = {"kind": "scripted-native-smoke", "real_model": False}
    def __call__(self, context):
        if context["history"]:
            return {"kind": "finish", "code": "", "note": "Native interface smoke finished; inspect receipts, not a task score."}
        return {"kind": "program", "note": "Read native reference/RGB, move 1 cm and return, hold current joints and gripper.",
                "code": '''reference = mcp.read_resource("robodojo://robot")
images = mcp.call_tool("camera_observe", {})
state = mcp.call_tool("robot_state", {})["structuredContent"]
p = state["arms"]["left"]["position"]
mcp.call_tool("move_ee", {"arm": "left", "position": [p[0], p[1], p[2] + 0.01]})
raised = mcp.call_tool("robot_state", {})["structuredContent"]
mcp.call_tool("move_ee", {"arm": "left", "position": p})
state = mcp.call_tool("robot_state", {})["structuredContent"]
mcp.call_tool("move_joints", {"arm": "left", "positions": state["joints"]["left"]})
mcp.call_tool("set_gripper", {"arm": "left", "opening": state["arms"]["left"]["gripper_commanded"]})
receipt = mcp.call_tool("execution_status", {})
'''}


def smoke_evidence(events, device):
    """Check observed motion as well as successful calls; never a task score."""
    feedback = [e for e in events if e["type"] == "feedback"]
    calls = [e for e in events if e["type"] == "device_call" and e["status"] == "returned"]
    states = [e["result"]["structuredContent"] for e in calls if e["args"][0] == "robot_state"]
    images = [c for e in calls if e["args"][0] == "camera_observe" for c in e["result"]["content"]
              if c["type"] == "image" and c.get("data")]
    checks = {"program_executed": bool(feedback) and all(e["status"] == "executed" for e in feedback),
              "rgb_received": bool(images), "stopped": device["stopped"], "observed_motion": False}
    if len(states) >= 3:
        initial, raised, returned = [s["arms"]["left"]["position"] for s in states[:3]]
        checks["observed_motion"] = raised[2] - initial[2] > 0.002 and math.dist(initial, returned) < 0.01
    return checks


def main():
    from roborag.cap.mcp_client import MCPClient
    from roborag.cap.brains import CodexBrain
    from roborag.cap.harness import run_episode
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "local.runtime.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--task", default="stack_blocks")
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Native API smoke with a scripted brain; no model calls")
    parser.add_argument("--prepare-only", action="store_true", help="Write standard MCP launch config for another client, without starting simulation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = json.loads(args.runtime.read_text())
    sys.path.insert(0, runtime["upstream"])
    from roboicl.paths import python_paths
    from roboicl.upstream import verify_submodules
    pinned = verify_submodules()
    data = (args.data_root or Path(runtime["assets"]).parent).resolve()
    layout = data / "Assets/Eval_Layout/RoboDojo/arx_x5" / str(args.seed) / f"{args.task}_{args.layout}.json"
    if not layout.is_file():
        raise FileNotFoundError(layout)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [str(ROOT / "src"), *python_paths("sim")]
    env = dict(PYTHONPATH=os.pathsep.join(paths), PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
               OMNI_KIT_ACCEPT_EULA="YES", OMNI_KIT_ALLOW_ROOT="1", OMP_NUM_THREADS="4",
               LD_PRELOAD=str(Path(runtime["sim_library"]) / "libstdc++.so.6"), LD_LIBRARY_PATH=runtime["sim_library"])
    worker = [runtime["sim_python"], "-u", "-m", "roborag.devices.robodojo_worker", "--task", args.task, "--layout", str(args.layout),
               "--seed", str(args.seed), "--data-root", str(data), "--output", str(out / "native"),
               "--enable_cameras", "--headless",
               "--kit_args=--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera --/app/hangDetector/enabled=false --/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true"]
    worker_config = out / "worker.json"
    worker_config.write_text(json.dumps({"command": worker, "env": env, "timeout": 240}, indent=2) + "\n")
    command = ["-m", "roborag.devices.robodojo_server", "--worker-config", str(worker_config)]
    server_config = {"command": sys.executable, "args": command, "env": {"PYTHONPATH": str(ROOT / "src")}}
    (out / "mcp-server.json").write_text(json.dumps(server_config, indent=2) + "\n")
    manifest = {"backend": "native RoboDojo", "protocol": "MCP stdio", "smoke": args.smoke,
                "prepare_only": args.prepare_only,
                "model": None if args.smoke else args.model, "dependencies": pinned,
                "layout": str(layout), "layout_sha256": hashlib.sha256(layout.read_bytes()).hexdigest(),
                "command": [sys.executable, *command], "native_worker": worker,
                "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for folder in (ROOT / "src", ROOT / "scripts") for p in folder.rglob("*.py")}}
    (out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(f"MCP launch configuration: {out / 'mcp-server.json'}")
        return 0
    brain = SmokeBrain() if args.smoke else CodexBrain(out / "model", model=args.model, effort=args.effort)
    summary = {"status": "failed", "task_success": None}
    try:
        with (out / "simulator.log").open("w") as log, MCPClient(sys.executable, command,
                env={"PYTHONPATH": str(ROOT / "src")}, timeout=300, errlog=log) as mcp:
            before = [{"tool": "robot_state"}]
            if not args.smoke:
                before.append({"tool": "camera_observe"})
            summary = run_episode(mcp, brain, "Follow the task instruction from robot_state and verify the outcome from observations.",
                                  output=out / "cap", max_rounds=args.max_rounds,
                                  before_round=before, on_exit=[{"tool": "stop_motion"}])
            summary["device"] = mcp.call_tool("execution_status")["structuredContent"]
            if args.smoke:
                events = [json.loads(line) for line in (out / "cap/events.jsonl").read_text().splitlines()]
                summary["smoke_checks"] = smoke_evidence(events, summary["device"])
                summary["native_smoke_passed"] = all(summary["smoke_checks"].values())
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "model_finished" and summary.get("native_smoke_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
