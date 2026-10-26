#!/usr/bin/env python3
"""Run a frozen suite and compare completed native episodes with public layouts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from summarize_runs import summarize

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def public_layout(baseline, task, seed, variant, layout):
    task_row = next(t for t in baseline["tasks"] if t["task"] == task)
    row = next(r for r in task_row["layouts"]
               if (r["seed"], r["variant"], r["layout"]) == (seed, variant, layout))
    if not row["included_in_official_evaluation"]:
        raise ValueError("Partial public rollout cannot serve as a completed baseline")
    return row


def verify_plan(plan):
    for name, expected in plan["frozen_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen file changed: {name}; create a separate suite")
    baseline = json.loads((ROOT / plan["baseline"]).read_text())
    for task in plan["tasks"]:
        for layout in plan["layouts"]:
            public_layout(baseline, task["task"], plan["seed"], plan["variant"], layout)
    return baseline


def aggregate(rows):
    valid = [r for r in rows if r["status"] == "native_completed"]
    if not valid:
        return None
    return {
        "completed_episodes": len(valid),
        "roborag_progress_mean": statistics.mean(r["roborag_progress"] for r in valid),
        "public_same_layout_progress_mean": statistics.mean(r["public_progress"] for r in valid),
        "paired_difference_mean": statistics.mean(r["difference"] for r in valid),
        "roborag_success_rate": statistics.mean(r["roborag_success"] for r in valid),
        "public_same_layout_success_rate": statistics.mean(r["public_success"] for r in valid),
        "roborag_progress_range": [min(r["roborag_progress"] for r in valid),
                                  max(r["roborag_progress"] for r in valid)],
        "model_calls": sum(r["evidence"]["completed_model_calls"] for r in valid),
        "known_total_tokens": sum(r["evidence"]["known_total_tokens"] for r in valid),
    }


def report(plan, baseline, output_root):
    rows = []
    for task in plan["tasks"]:
        for layout in plan["layouts"]:
            public = public_layout(baseline, task["task"], plan["seed"], plan["variant"], layout)
            folder = output_root / f"{task['task']}-l{layout}"
            local = summarize(folder) if folder.exists() else None
            valid = local is not None and local["valid_native_episode"]
            if valid:
                manifest = json.loads((folder / "run.json").read_text())
                arguments = manifest["arguments"]
                expected = {"task": task["task"], "layout": layout, "seed": plan["seed"],
                            "variant": plan["variant"], "mode": plan["mode"],
                            "long_k": plan["long_k"], "short_k": plan["short_k"],
                            "effort": plan["effort"], "max_calls": 0}
                if any(arguments.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Run settings differ from suite: {folder}")
                if local["manual_stop"]:
                    valid = False
            row = {
                "task": task["task"], "category": task["category"], "layout": layout,
                "status": "native_completed" if valid else (local["status"] if local else "not_run"),
                "run_directory": str(folder.relative_to(ROOT)),
                "public_progress": public["official_score_0_100"],
                "public_success": public["official_success"],
                "public_trace_url": public["trace_url"], "evidence": local,
            }
            if valid:
                row.update(roborag_progress=local["score_0_100"],
                           roborag_success=bool(local["success_rate"]),
                           difference=local["score_0_100"] - public["official_score_0_100"])
            elif local and local["manual_stop"]:
                row["status"] = "interrupted_by_experimenter"
            rows.append(row)
    task_summaries = {task["task"]: aggregate([r for r in rows if r["task"] == task["task"]])
                      for task in plan["tasks"]}
    complete = all(r["status"] == "native_completed" for r in rows)
    overall = None
    if complete:
        overall = {
            "roborag_task_macro_mean": statistics.mean(s["roborag_progress_mean"] for s in task_summaries.values()),
            "public_same_layout_task_macro_mean": statistics.mean(s["public_same_layout_progress_mean"] for s in task_summaries.values()),
            "paired_task_macro_difference": statistics.mean(s["paired_difference_mean"] for s in task_summaries.values()),
        }
    result = {
        "schema": "roborag.public-comparison.v1", "suite_id": plan["id"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete, "planned_episodes": len(rows),
        "completed_episodes": sum(r["status"] == "native_completed" for r in rows),
        "comparison_kind": "external public baseline; not a controlled causal ablation",
        "excluded_infrastructure_attempts": plan.get("excluded_infrastructure_attempts", []),
        "public_task_references": [
            {"task": t["task"], "published_all_layouts_progress_mean": t["published_score_0_100"],
             "published_full_episode_count": t["official_episode_count"],
             "selected_layout_progress_mean": statistics.mean(
                 r["public_progress"] for r in rows if r["task"] == t["task"])}
            for t in baseline["tasks"] if t["task"] in task_summaries
        ],
        "limitations": plan["limitations"], "task_summaries": task_summaries,
        "overall_complete_suite_only": overall, "episodes": rows,
    }
    write(ROOT / plan["report"], result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Otherwise verify and report only")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    baseline = verify_plan(plan)
    output_root = ROOT / plan["output_root"]
    initial = report(plan, baseline, output_root)
    if not args.execute:
        print(json.dumps({k: v for k, v in initial.items() if k != "episodes"}, indent=2))
        return 0
    runtime = json.loads((ROOT / "local.runtime.json").read_text())
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "suite-status.json"
    active = []
    workers = plan.get("workers", 1)
    if workers not in (1, 2):
        raise ValueError("This single-GPU launcher supports one or two simulator workers")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Suite interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        # Cycle tasks within each layout; use the first completed attempt, not best-of-N.
        pending = []
        for layout in plan["layouts"]:
            for task in plan["tasks"]:
                folder = output_root / f"{task['task']}-l{layout}"
                if folder.exists():
                    existing = summarize(folder)
                    if existing and existing["valid_native_episode"]:
                        continue
                    raise RuntimeError(f"Preserve incomplete attempt for review: {folder}")
                pending.append((task, layout, folder))
        while pending or active:
            while pending and len(active) < workers:
                verify_plan(plan)
                task, layout, folder = pending.pop(0)
                command = [runtime["policy_python"], str(ROOT / "scripts/run_episode.py"),
                           "--task", task["task"], "--layout", str(layout),
                           "--seed", str(plan["seed"]), "--variant", plan["variant"],
                           "--mode", plan["mode"], "--role", "validation",
                           "--effort", plan["effort"], "--long-k", str(plan["long_k"]),
                           "--short-k", str(plan["short_k"]), "--corpus", str(ROOT / task["corpus"]),
                           "--output", str(folder), "--allow-shared-gpu"]
                print(f"Starting {task['task']} layout {layout}", flush=True)
                log = (output_root / f"{folder.name}-launcher.log").open("x")
                environment = dict(os.environ, **plan.get("transport_environment", {}))
                try:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                             stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                active.append((child, log, folder))
                write(status_path, {"status": "running",
                                    "active": [{"pid": p.pid, "run": f.name} for p, _, f in active],
                                    "pending_episodes": len(pending), "plan": str(args.plan)})
            for child, log, folder in active[:]:
                rc = child.poll()
                if rc is None:
                    continue
                log.close()
                active.remove((child, log, folder))
                current = report(plan, baseline, output_root)
                local = summarize(folder)
                if rc or local is None or not local["valid_native_episode"]:
                    raise RuntimeError(f"Episode infrastructure failure: {folder} (rc={rc})")
                print(f"Completed {current['completed_episodes']}/{current['planned_episodes']}", flush=True)
            if active:
                time.sleep(1)
        write(status_path, {"status": "completed", "report": plan["report"]})
    except BaseException as error:
        for child, _, _ in active:
            if child.poll() is None:
                child.terminate()
        for child, log, _ in active:
            child.wait(timeout=40)
            log.close()
        write(status_path, {"status": "stopped", "error": str(error), "report": plan["report"]})
        report(plan, baseline, output_root)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
