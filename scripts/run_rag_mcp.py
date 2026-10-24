#!/usr/bin/env python3
"""One native robot episode: MCP evidence -> GPT plan -> GPT action chunks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from roborag.hexashell_agent import run_agent


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_file(path, process, seconds=300):
    started = time.monotonic()
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f"Process exited {process.returncode} before {path}")
        if time.monotonic()-started > seconds:
            raise TimeoutError(str(path))
        time.sleep(.2)


def stop(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


PROMPT = """You are the robot policy for this episode. Task: stack the three blocks with different textures into one stable tower. Any ordering is allowed.

Use the roborag MCP tools. Start by observe(head, calibration=true), retrieve_evidence(mode=plan) describing the CURRENT scene, then record_plan with YOUR steps, object bindings, completion checks, cited sources and adaptations. The retrieved graph is a historical example, NOT the plan to execute. You can skip, reorder and combine its ideas. Read 1-2 relevant phase action references if they clarify grasping/placement; the default measured summaries and images usually suffice (include_actions=false). For a release/withdrawal example use section=end. Avoid repeatedly retrieving the same evidence.

Then execute_chunk with short batches of YOUR world-frame fingertip TCP waypoints. Combine known free-space approach/motion operations into one call, but pause for fresh images after an uncertain grasp/lift and after release. The tool returns new head images automatically; request the active wrist when checking grasp/alignment. Do not issue a separate observe after every chunk. Simulated time is stopped during thinking.

Coordinate/controller manual:
- world xyz in meters, +x right as seen by robot/head image, +y forward/upper image, +z up; table z=0.74. Arm bases x=-0.30/+0.30,y=-0.45.
- TCP is fingertip center, link6 +0.145 m along local approach X. Historical action reference uses link6 deltas, so never copy its numbers into live absolute TCP waypoints.
- point down/forward/down45 selects approach; open_axis x/y/z is finger separation axis. gripper=1 fully open (8.8 cm),0 closed; applied after reaching xyz. Commanded closure does NOT prove a grasp. Check lifted object in RGB.
- Joint interpolation uses the existing kinematic planner and knows only the arm and table, not objects. An infeasible/unreached/clipped waypoint stops the rest of that chunk. Use receipts to replan.
- Camera K and T map camera coordinates (+x right,+y down,+z forward) to world. A pixel ray d=R@[(u-cx)/fx,(v-cy)/fy,1]; intersection with z=h is c+(h-c.z)/d.z*d. Estimate object dimensions and coordinates from current RGB + calibration. Fixed head K/T can be reused. Wrist T changes as the arm moves.
- Current images are also available at obs/head.png, obs/wrist_l.png, obs/wrist_r.png; obs/state.json has calibration and proprioception. Use these existing files for calculations/crops with numpy/scipy/Pillow/OpenCV and view_image. All robot actions must use MCP execute_chunk. No source code, object poses, hidden state or evaluation files are available.
- The robot has a finite simulation step budget in state; keep chunks economical. Public task completion requires all three blocks stacked AND BOTH ARMS RETURNED TO THEIR STARTING POSES. Once the pile is stable, withdraw fingers clear and issue home=true (no xyz/point) for the arm(s) that moved, grippers open. A tower alone is insufficient for completion.
- Verify the blocks are centered in WORLD XY after placement. Perspective makes stacked top faces shift in the head image; compare calibrated top-face centers at their respective estimated heights, not just 2D overlap. Aim for sub-centimeter alignment, accounting for the current object offset inside the gripper. If visibly off-center, regrasp/recenter rather than waiting for time to expire. Reserve time for returning home.
- Episode stops automatically on native success or when the step budget ends. episode_over alone does NOT distinguish these cases; do not infer success from it at the time limit. If the tower is complete and arms are home but the episode remains active, investigate alignment and correct it rather than repeatedly waiting.

Keep comments brief, spend effort on geometry and feedback, and finish with what the returned observations establish.
"""

DELTA_PROMPT = """Control the dual ARX X5 robot to stack the three blocks with different textures into one stable pile, then RETURN BOTH ARMS TO THEIR INITIAL POSES. Any block ordering is allowed. Use only RGB, camera calibration and measured robot state; no object truth or evaluation files are available.

Workflow: observe current head and wrists, retrieve_evidence(mode=plan) describing the INITIAL current scene, then generate and record YOUR new plan with current object bindings, completion observations and adopted evidence IDs. Historical episode traces are examples, not executable plans. You may skip/reorder their steps. Before movement, expand 2-3 relevant read_action_reference sources with include_actions=true: learn approach/grasp and placement action scale, orientation and gripper timing. Use section=end for a release/withdrawal window. Keep initial evidence in history; only retrieve again if a specific gap prevents action.

Use execute_chunk for all robot control. The LIVE action_space is IDENTICAL to the recorded 14D world link6 delta format. Rows are [left dx,dy,dz,rx,ry,rz,gripper, right dx,dy,dz,rx,ry,rz,gripper]. Rotations are world rotation vectors radians, not Euler angles. Start from current action_state EE poses, then accumulate each row from the PREVIOUS REQUESTED target. Per-arm translation norm<=0.035m and rotation norm<=0.35rad per row. Predict 1-24 steps at 25Hz; use short chunks near grasp/contact/release and longer chunks for known free travel. Keep the inactive arm still by zero deltas while preserving its gripper target. Output <=6 decimal places.

EE means link6, NOT fingertip/TCP. Poses use xyz + quaternion [w,x,y,z]. The fingers extend roughly 0.145m along link6 local +X. World +X points right in head image, +Y away from robot, +Z up; table z=0.74m. Each arm is best on its own side (left for negative X, right for positive X). Cross-center grasps can rotate elbow/wrist into poor posture. Right gripper upside-down over a tabletop object can push instead of lift. Preserve useful demonstrated downward/forward wrist orientation when closing. Gripper 0=closed,1=open; if the pose is suitable, hold the same EE pose while closing rather than translating and closing simultaneously. After closure, verify a small lift from CURRENT RGB; a target gripper value or small tracking residual does not prove a grasp.

Observe before planning. Current RGB files also exist at obs/head.png, obs/wrist_l.png, obs/wrist_r.png, with current calibration/state in obs/state.json. Native Python numpy/scipy/Pillow/OpenCV and image viewing are available for calculations. Camera K,T are calibrated; T maps OpenCV camera coordinates to world. Native action feedback already includes new RGB: do not redundantly call observe after every chunk. Simulation pauses during reasoning.

Adapt source motions to live object positions. Historical success text is not current success. After release, inspect relative alignment and object motion; regrasp if needed. Do not spend many calls on speculative geometry fitting when a short observation/action correction suffices. Return both arms using execute_chunk(actions=[],return_home=true) once the pile is stable and fingers are clear. If it remains active after returning home, correct the pile; episode_over at the time limit alone does not mean success. Continue until the episode ends. Keep comments and action notes brief.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-wall-seconds", type=float, default=5400)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--controller", choices=["tcp", "delta"], default="delta")
    args = parser.parse_args()
    runtime = json.loads((ROOT / "local.runtime.json").read_text())
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
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", OMNI_KIT_ACCEPT_EULA="YES",
        OMNI_KIT_ALLOW_ROOT="1", CUDA_VISIBLE_DEVICES="0", ROBOICL_DATA_ROOT=str(ROOT / "data"),
        ROBOICL_RESULTS_ROOT=str(output / "native"), ROBODOJO_RUN_ID=output.name, OMP_NUM_THREADS="4",
        PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(hexa), *python_paths("sim")]),
        LD_PRELOAD=str(Path(runtime["sim_library"]) / "libstdc++.so.6"), LD_LIBRARY_PATH=runtime["sim_library"])
    sim_command = [runtime["sim_python"], "-u", "-m", "roborag.rag_sim", "--task", "stack_blocks",
        "--output", str(output), "--agent-workspace", str(workspace), "--hexashell", str(hexa),
        "--layout", str(args.layout), "--seed", str(args.seed), "--max-wall-seconds", str(args.max_wall_seconds),
        "--enable_cameras", "--headless",
        "--kit_args=--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera --/app/hangDetector/enabled=false --/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true"]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mcp_env = dict(os.environ, PYTHONPATH=str(ROOT / ".runtime/mcp") + os.pathsep + str(ROOT / "src"))
    mcp_command = [runtime["policy_python"], "-m", "roborag.rag_mcp", "--bank", str(ROOT / "data/graph-bank/v1"),
        "--data", str(ROOT / "data"), "--workspace", str(workspace), "--output", str(output),
        "--port", str(port), "--controller", args.controller] + (["--no-rag"] if args.no_rag else [])
    paths = [ROOT / "src/roborag" / name for name in ("rag_mcp.py", "rag_execution.py", "rag_sim.py",
        "hexashell_adapter.py", "hexashell_agent.py", "graph_retrieval.py", "graph_chunks.py")]
    paths += [Path(__file__).resolve(), ROOT / "data/graph-bank/v1/execution-index.json"]
    save(output / "run.json", dict(started_at=datetime.now(timezone.utc).isoformat(),
        args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        schema="roborag.mcp-pilot.v1", task="stack_blocks", model=args.model,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
        layout_sha256=digest(ROOT / f"data/Assets/Eval_Layout/RoboDojo/arx_x5/{args.seed}/stack_blocks_{args.layout}.json"),
        observation="current RGB + camera calibration + robot proprioception; no object ground truth",
        controller=("RoboICL world link6 delta grammar; native RoboDojo EE control" if args.controller == "delta"
                    else "HexaShell kinematic adapter; GPT authored TCP waypoint chunks"),
        workflow="MCP retrieval -> GPT new plan -> GPT chunks -> native evaluator"))
    processes, started = [], time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with (output / "simulator.log").open("w") as log:
            sim = subprocess.Popen(sim_command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(sim)
        save(output / "process.json", dict(simulator_pid=sim.pid, runner_pid=os.getpid()))
        wait_file(output / "ready.json", sim)
        with (output / "mcp-server.log").open("w") as log:
            mcp = subprocess.Popen(mcp_command, cwd=ROOT, env=mcp_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(mcp)
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.1):
                    break
            except OSError:
                if mcp.poll() is not None:
                    raise RuntimeError("MCP server exited")
                time.sleep(.1)
        else:
            raise TimeoutError("MCP server startup")
        agent = run_agent(workspace, output, DELTA_PROMPT if args.controller == "delta" else PROMPT, model=args.model, effort=args.effort,
            timeout=args.max_wall_seconds-(time.monotonic()-started),
            python_packages=ROOT / "data/hexashell-agent-python", mcp_url=f"http://127.0.0.1:{port}/mcp")
        if not json.loads((workspace / "obs/state.json").read_text())["episode_over"]:
            save(workspace / "inbox/99999999999999999999-agent-exit.json", {"finish": "agent_exit"})
        if sim.wait(timeout=120):
            raise RuntimeError("Simulator finalization failed")
        completed = json.loads((output / "sim_complete.json").read_text())
        native = json.loads(Path(completed["native_result"]).read_text())
        if native.get("eval_time") != 1 or native.get("policy_error"):
            raise RuntimeError("Incomplete native evaluation")
        calls = [json.loads(x) for x in (output / "mcp-calls.jsonl").read_text().splitlines()]
        names = [c["tool"] for c in calls]
        chunks = [c for c in calls if c["tool"] == "execute_chunk"]
        result = dict(complete=True, native=native, native_result=completed["native_result"],
            native_sha256=digest(completed["native_result"]), native_agent=agent,
            wall_seconds=time.monotonic()-started, actual_billed_usd=None,
            retrieval_before_plan_before_execution=names.index("retrieve_evidence") < names.index("record_plan") < names.index("execute_chunk"),
            tool_counts={name:names.count(name) for name in sorted(set(names))},
            executed_waypoints=sum(c["result"]["result"]["executed_waypoints"] for c in chunks),
            executed_delta_rows=sum(c["result"]["result"].get("executed_rows", 0) for c in chunks),
            physics_action_steps=sum(c["result"]["result"]["action_steps"] for c in chunks),
            control_step_note="physics_action_steps counts native action dispatches, not PhysX integration substeps",
            caveat="One development episode; no causal retrieval benefit or heldout success rate established.")
        save(output / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as exc:
        save(output / "exit.json", dict(complete=False, error=type(exc).__name__, detail=str(exc), wall_seconds=time.monotonic()-started))
        raise
    finally:
        for process in reversed(processes):
            stop(process)


if __name__ == "__main__":
    main()
