#!/usr/bin/env python3
"""Audit native MCP RoboICL outcome and recorded model consumption."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


def read(path): return json.loads(path.read_text()) if path.is_file() else None


def rows(path):
    if path.is_file():
        for line in path.read_text().splitlines():
            try: yield json.loads(line)
            except json.JSONDecodeError: pass  # A partial live append is not a completed event.


def count(value): return value if type(value) is int and value >= 0 else None


def source_audit(folder, plan):
    expected = plan.get("source_sha256")
    if not expected:
        return None
    changed = [name for name, digest in expected.items()
               if not (folder / "source" / name).is_file()
               or hashlib.sha256((folder / "source" / name).read_bytes()).hexdigest() != digest]
    return {"checked_files": len(expected), "all_match": not changed, "changed_or_missing": changed}


def summarize(folder, audit_videos=False):
    plan = read(folder / "frozen-plan.json")
    exit_info = read(folder / "exit.json")
    marker = read(folder / "native/evaluation.json")
    native = read(Path(marker["result"])) if marker else None
    native_calls = list(rows(folder / "mcp-calls.jsonl"))
    by_tool = {}
    for call in native_calls:
        item = by_tool.setdefault(call["tool"], {"calls": 0, "errors": 0, "wall_seconds": 0, "response_json_bytes": 0})
        item["calls"] += 1
        item["errors"] += call["status"] == "error"
        item["wall_seconds"] += call["seconds"]
        item["response_json_bytes"] += call.get("result_bytes", 0)
    controls = sum(c["tool"] == "control_step" and c["status"] == "returned" for c in native_calls)
    valid = bool(plan["mode"] == "model_evaluation" and exit_info and exit_info.get("status") == "native_completed"
        and marker and marker["status"] == "scored" and marker["steps"] == controls == exit_info["steps"]
        and native and native.get("eval_time") == 1 and not native.get("policy_error")
        and type(native.get("score")) in (int, float) and math.isfinite(native["score"]) and 0 <= native["score"] <= 100
        and native.get("success_rate") in (0, 1)
        and not list((folder / "native").rglob("_infrastructure_failure.json")))
    calls = []
    for invocation_path in sorted((folder / "decisions").glob("call-*/invocation.json")):
        call = invocation_path.parent
        invocation = read(invocation_path)
        execution = read(call / "execution.json") or {}
        response = read(call / "response.json") or {}
        events = list(rows(call / "events.jsonl"))
        usages = [e.get("usage", {}) for e in events if e.get("type") == "turn.completed"]
        usage = usages[0] if len(usages) == 1 else {}
        command = invocation["command"]
        requested_model = command[command.index("--model") + 1]
        efforts = [s.split("=", 1)[1].strip('"') for s in command if s.startswith("model_reasoning_effort=")]
        calls.append({"call": call.name, "completed": response.get("status") == "completed",
            "model_matches": requested_model == invocation["model"] == plan["model"],
            "effort_matches": efforts == [plan["effort"]] and invocation["effort"] == plan["effort"],
            "input_tokens": count(usage.get("input_tokens")), "output_tokens": count(usage.get("output_tokens")),
            "reasoning_output_tokens": count(usage.get("reasoning_output_tokens")),
            "cache_write_input_tokens": count(usage.get("cache_write_input_tokens")),
            "cached_input_tokens": count(usage.get("cached_input_tokens")), "wall_seconds": execution.get("wall_seconds"),
            "transport_reconnects": sum(e.get("type") == "error" and e.get("message", "").startswith("Reconnecting...") for e in events)})
    totals = {}
    for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens", "cache_write_input_tokens"):
        totals[f"known_{key}"] = sum(c[key] for c in calls if c[key] is not None)
        totals[key] = totals[f"known_{key}"] if calls and all(c[key] is not None for c in calls) else None
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"] if all(
        totals[k] is not None for k in ("input_tokens", "output_tokens")) else None
    totals["calls_without_complete_usage"] = sum(c["input_tokens"] is None or c["output_tokens"] is None for c in calls)
    durations = [c["wall_seconds"] for c in calls if c["wall_seconds"] is not None]
    videos = []
    if valid:
        details = list(native.get("details", {}).values())
        valid = len(details) == 1 and details[0]["layout_id"] == plan["layout"] and not details[0].get("interrupted")
        for camera in ("head", "left_wrist", "right_wrist"):
            paths = list(Path(marker["result"]).parent.glob(f"*cam_{camera}_*.mp4"))
            if len(paths) != 1: raise ValueError(f"Expected one native {camera} video")
            path = paths[0]
            info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(path)], text=True))["streams"][0]
            if int(info["nb_frames"]) != controls + 1:
                raise ValueError(f"Native video/control count mismatch: {camera}: {info['nb_frames']} vs {controls + 1}")
            if audit_videos:
                decoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
                if decoded.returncode or decoded.stderr.strip(): raise ValueError(f"Invalid video: {path}")
            videos.append({"path": str(path), **info, "decoded": True if audit_videos else None})
    decisions = [e for p in (folder / "policy").glob("*/decisions.jsonl") for e in rows(p)]
    rejections = [e for p in (folder / "policy").glob("*/action_rejections.jsonl") for e in rows(p)]
    return {"valid_native_episode": valid, "status": (exit_info or {}).get("status", "running"),
        "native_score_0_100": native.get("score") if valid else None,
        "native_success": bool(native["success_rate"]) if valid else None,
        "executed_control_steps": controls, "attempted_model_calls": len(calls),
        "completed_model_calls": sum(c["completed"] for c in calls), "accepted_decisions": len(decisions),
        "rejected_action_tensors": sum(e.get("error") == "invalid_action_tensor" for e in rejections),
        "mcp_tool_errors": sum(c["status"] == "error" for c in native_calls),
        "total_mcp_calls": len(native_calls), "mcp_call_wall_seconds": sum(c["seconds"] for c in native_calls),
        "mcp_by_tool": by_tool,
        "wall_seconds": (exit_info or {}).get("wall_seconds"), "model_call_wall_seconds": sum(durations),
        "model_call_mean_seconds": statistics.mean(durations) if durations else None,
        "model_call_median_seconds": statistics.median(durations) if durations else None,
        "model_configuration_matches": bool(calls) and all(c["model_matches"] and c["effort_matches"] for c in calls),
        **totals, "native_result": marker["result"] if marker else None, "videos": videos, "calls": calls,
        "source_audit": source_audit(folder, plan),
        "overlapping_gpu_processes": read(folder / "overlapping-gpu-processes.json"),
        "initial_context_comparison": {k:v for k,v in (read(folder / "initial-context-comparison.json") or {}).items() if k != "images"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-videos", action="store_true")
    args = parser.parse_args()
    folder = args.run.resolve()
    plan = read(folder / "frozen-plan.json")
    baseline = read(args.baseline_report)["baseline"]
    baseline = {k:baseline[k] for k in ("run_directory", "valid_native_episode", "native_score_0_100", "native_success",
        "executed_control_steps", "completed_model_calls", "known_input_tokens", "known_output_tokens", "total_tokens",
        "cached_input_tokens", "all_attempt_wall_seconds", "completed_call_mean_seconds", "exit")}
    result = {"schema": "roborag.roboicl-mcp-result.v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task": plan["task"], "layout": plan["layout"], "seed": plan["seed"], "model": plan["model"], "effort": plan["effort"],
        "baseline": baseline, "trial": summarize(folder, args.audit_videos),
        "baseline_report_sha256": hashlib.sha256(args.baseline_report.read_bytes()).hexdigest(),
        "limitations": plan["limitations"] + ["Usage is CLI-reported tokens, not an invoice or actual dollar charge.",
            "Input tokens include cached input; cache is not added again to total tokens.",
            "MCP call wall time includes native simulation and image encoding/transfer; it is not isolated protocol overhead.",
            "Observed concurrent GPU work includes cloth-grasp training; timing is not a controlled speed comparison."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result["trial"].items() if k not in ("calls", "videos")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
