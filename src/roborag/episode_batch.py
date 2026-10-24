#!/usr/bin/env python3
"""Run a fixed set of independent MCP episodes with bounded concurrency."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import psutil

from .episode_files import write_json as write

ROOT = Path(__file__).resolve().parents[2]


def protocol_sources(bank):
    paths = list((ROOT / "src/roborag").rglob("*.py")) + list((ROOT / "prompts").glob("*.txt"))
    paths += list(bank.glob("*.json")) + list((bank / "graphs").glob("*.json"))
    paths += list((ROOT / "vendor/HexaShell/tasks/fold_clothes/tools/cloth_view").glob("*.py"))
    paths += [ROOT / "vendor/HexaShell/tools/table_plane/tool.py"]
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def resource_sample(active):
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"], text=True).strip()
    values = [value.strip() for value in gpu.split(",")]
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True).strip()
    owned = {}
    for key, (process, _, _) in active.items():
        samples = []
        try:
            parent = psutil.Process(process.pid)
            children = [parent, *parent.children(recursive=True)]
        except psutil.NoSuchProcess:
            children = []
        for child in children:
            try:
                with child.oneshot():
                    cpu = child.cpu_times()
                    samples.append(dict(pid=child.pid, started_epoch=child.create_time(),
                        rss_bytes=child.memory_info().rss, cpu_seconds=cpu.user + cpu.system))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        owned[key] = samples
    try:
        watts = float(values[3])
    except ValueError:
        watts = None
    return dict(epoch=time.time(), gpu=", ".join(values[:3]), gpu_power_w=watts,
                compute_processes=processes, active=list(active), owned_processes=owned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["stack_blocks", "fold_clothes"], required=True)
    parser.add_argument("--controller", choices=["delta", "tcp"], default="delta")
    parser.add_argument("--max-wall-seconds", type=float, default=1800)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--bank", type=Path, default=ROOT / "data/graph-bank/v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts", default="0,1")
    parser.add_argument("--variants", default="standard")
    parser.add_argument("--cases-file", type=Path, help="JSON list of {variant, layout} cases, in fixed launch order")
    parser.add_argument("--parallel", type=int, choices=[1, 2], default=2)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="xhigh")
    args = parser.parse_args()
    runtime = json.loads((ROOT / "local.runtime.json").read_text())
    cases = (json.loads(args.cases_file.read_text()) if args.cases_file else
             [dict(variant=variant, layout=int(layout)) for variant in args.variants.split(",") for layout in args.layouts.split(",")])
    if not cases or any(c["variant"] not in ("standard", "random") or not isinstance(c["layout"], int) or c["layout"] < 0 for c in cases):
        parser.error("nonempty cases must have standard/random variant and nonnegative integer layout")
    if len({(c["variant"], c["layout"]) for c in cases}) != len(cases):
        parser.error("duplicate cases would overwrite experiment evidence")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen_sources = protocol_sources(args.bank.resolve())
    write(out / "batch.json", dict(started_at=datetime.now(timezone.utc).isoformat(), cases=cases,
          parallel=args.parallel, model=args.model, effort=args.effort, task=args.task, controller=args.controller,
          bank=str(args.bank.resolve()), no_rag=args.no_rag, max_wall_seconds=args.max_wall_seconds,
          source_sha256=frozen_sources,
          resources="5-second GPU board power/memory/utilization and owned-process RSS/CPU samples; board energy includes other GPU workloads",
          interpretation="Independent episode throughput; no matched serial speedup claim"))
    active, records, seen = {}, [], {}
    started = time.monotonic()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (out / "resources.jsonl").open("w") as resources:
            while cases or active:
                sample = resource_sample(active)
                gpu = sample["gpu"]
                resources.write(json.dumps(sample)+"\n")
                resources.flush()
                for key, (process, log, record) in list(active.items()):
                    directory = Path(record["output"])
                    progress = directory / "progress.json"
                    if progress.exists():
                        p = json.loads(progress.read_text())
                        if p.get("counter") != seen.get(key):
                            seen[key] = p.get("counter")
                            print(json.dumps(dict(case=key, subgoal=p.get("subgoal"), step=p.get("step_after"), plan_ok=p["result"].get("plan_ok"))), flush=True)
                    if process.poll() is not None:
                        log.close()
                        record.update(returncode=process.returncode, ended_epoch=time.time())
                        result = directory / "result.json"
                        record["result"] = json.loads(result.read_text()) if result.exists() else None
                        records.append(record)
                        del active[key]
                        print(json.dumps(dict(case=key, finished=True, returncode=process.returncode,
                                              score=(record["result"] or {}).get("native", {}).get("score"))), flush=True)
                # Allocate the next simulator only after the previous launch
                # has finished GPU initialization. This avoids racing the free-
                # memory checks in two cold starts on the same device.
                ready = all((Path(record["output"]) / "ready.json").exists() for _, _, record in active.values())
                if cases and len(active) < args.parallel and ready and int(gpu.split(",")[0]) >= 11000:
                    if protocol_sources(args.bank.resolve()) != frozen_sources:
                        raise RuntimeError("Source or bank changed during frozen suite; refusing to mix revisions")
                    case = cases.pop(0)
                    key = f"{case['variant']}-l{case['layout']}"
                    directory = out / key
                    command = [runtime["policy_python"], str(ROOT / "scripts/run_rag_mcp.py"),
                               "--output", str(directory), "--layout", str(case["layout"]), "--variant", case["variant"],
                               "--model", args.model, "--effort", args.effort, "--task", args.task,
                               "--controller", args.controller, "--max-wall-seconds", str(args.max_wall_seconds)]
                    command += ["--bank", str(args.bank.resolve())]
                    if args.no_rag:
                        command.append("--no-rag")
                    log = (out / f"{key}.log").open("w")
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    record = dict(**case, output=str(directory), pid=process.pid, started_epoch=time.time(), command=command)
                    active[key] = (process, log, record)
                    print(json.dumps(dict(case=key, started=True, pid=process.pid)), flush=True)
                write(out / "status.json", dict(complete=not cases and not active, pending=cases,
                    active={key: record for key, (_, _, record) in active.items()}, finished=records,
                    wall_seconds=time.monotonic()-started))
                if cases or active:
                    time.sleep(5)
    except BaseException as exc:
        write(out / "exit.json", dict(complete=False,error=type(exc).__name__,detail=str(exc),
                                      wall_seconds=time.monotonic()-started))
        raise
    finally:
        for process, log, record in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            log.close()
        for process, _, _ in active.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
