#!/usr/bin/env python3
"""Prepare, train, and inspect the local one-demonstration cloth-grasp experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "probe", "train", "evaluate", "serve"])
    p.add_argument("--run", type=Path, default=ROOT / "runs/cloth-grasp-v1")
    p.add_argument("--demo", type=Path)
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=1042)
    p.add_argument("--port", type=int, default=8768)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--bootstrap", type=Path, help="Reuse only TRAIN physics episodes from an earlier run")
    p.add_argument("--warmup-episodes", type=int, default=8)
    p.add_argument("--bc-steps", type=int, default=400)
    p.add_argument("--bc-weight", type=float, default=3.)
    p.add_argument("--policy-updates", type=int, default=20)
    p.add_argument("--binary-gripper", action="store_true")
    p.add_argument("--evaluation-episodes", type=int, default=8)
    p.add_argument("--evaluate-initial", action="store_true")
    p.add_argument("--protocol", type=Path)
    p.add_argument("--scale-plan", type=Path)
    p.add_argument("--mimic-plan", type=Path)
    p.add_argument("--mimic-phase", choices=["reference","collect","probe","confirm","check"])
    p.add_argument("--focus-plan", type=Path)
    p.add_argument("--focus-phase", choices=['collect','baseline','probe','confirm'])
    p.add_argument("--focus-round", type=int, default=0)
    p.add_argument("--case-start", type=int, default=0)
    p.add_argument("--case-count", type=int, default=6)
    args = p.parse_args()
    runtime = json.loads((ROOT / "local.runtime.json").read_text())
    out = args.run.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, runtime["upstream"])
    from roboicl.paths import python_paths
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), *python_paths("sim")]))
    if args.command == "prepare":
        if not args.demo:
            p.error("--demo must identify exactly one TRAIN HDF5 episode")
        code = "from roborag.cloth_grasp.demo import prepare; import sys; print(prepare(sys.argv[1],sys.argv[2]))"
        return subprocess.call([runtime["policy_python"], "-c", code, str(args.demo), str(out)], env=env)
    if args.command == "serve":
        return subprocess.call([sys.executable, "-m", "roborag.cloth_grasp.dashboard", "--run", str(out), "--port", str(args.port)], env=env)
    if not (out / "demo.json").is_file():
        p.error("Run prepare first")
    env.update(LD_PRELOAD=str(Path(runtime["sim_library"]) / "libstdc++.so.6"),
               LD_LIBRARY_PATH=runtime["sim_library"], ROBOICL_DATA_ROOT=str(ROOT / "data"),
               ROBOICL_RESULTS_ROOT=str(out / "native"), ROBODOJO_RUN_ID=out.name,
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", PYTHONNOUSERSITE="1",
               OMNI_KIT_ACCEPT_EULA="YES", OMNI_KIT_ALLOW_ROOT="1", CUDA_VISIBLE_DEVICES="0")
    cmd = [runtime["sim_python"], "-u", "-m", "roborag.cloth_grasp.sim_worker",
           "--mode", args.command, "--run", str(out), "--episodes", str(args.episodes),
           "--seed", str(args.seed), "--headless", "--enable_cameras",
           "--kit_args=--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera --/app/hangDetector/enabled=false --/app/updateOrder/checkForHydraRenderComplete=1000 --/app/renderer/waitIdle=true --/app/hydraEngine/waitIdle=true"]
    if args.checkpoint:
        cmd += ["--checkpoint", str(args.checkpoint.resolve())]
    if args.bootstrap:
        cmd += ["--bootstrap", str(args.bootstrap.resolve())]
    if not args.mimic_plan:
        cmd += ["--warmup-episodes",str(args.warmup_episodes),"--bc-steps",str(args.bc_steps),
                "--bc-weight",str(args.bc_weight),"--policy-updates",str(args.policy_updates),
                "--evaluation-episodes",str(args.evaluation_episodes)]
    if args.binary_gripper:cmd += ["--binary-gripper"]
    if args.evaluate_initial:cmd += ["--evaluate-initial"]
    if args.mimic_plan:
        cmd += ["--mimic-plan",str(args.mimic_plan.resolve()),"--mimic-phase",args.mimic_phase,
                "--focus-round",str(args.focus_round),"--case-start",str(args.case_start),"--case-count",str(args.case_count)]
    elif args.focus_plan:
        cmd += ["--focus-plan",str(args.focus_plan.resolve()),"--focus-phase",args.focus_phase,
                "--focus-round",str(args.focus_round),"--case-start",str(args.case_start),"--case-count",str(args.case_count)]
    elif args.protocol:
        cmd += ["--protocol",str(args.protocol.resolve()),"--case-start",str(args.case_start),"--case-count",str(args.case_count)]
    elif args.scale_plan:
        cmd += ["--scale-plan",str(args.scale_plan.resolve()),"--case-start",str(args.case_start),"--case-count",str(args.case_count)]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
