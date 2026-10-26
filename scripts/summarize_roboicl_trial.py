#!/usr/bin/env python3
"""Audit a frozen RoboICL model trial and its historical native baseline."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def rows(path):
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # A live append can be incomplete; it is not evidence.


def token_count(value):
    return value if type(value) is int and value >= 0 else None


def command_effort(command):
    values = [argument.split("=", 1)[1] for previous, argument in zip(command, command[1:])
              if previous in ("-c", "--config") and argument.startswith("model_reasoning_effort=")]
    if len(values) != 1:
        return None
    try:
        return json.loads(values[0])
    except json.JSONDecodeError:
        return None


def summarize(folder, *, model, effort, layout, audit_videos=False):
    native_paths = list(folder.glob("upstream-results/eval_result/**/_result.json"))
    native = read(native_paths[0]) if len(native_paths) == 1 else None
    statuses = [read(path) for path in folder.glob("upstream-results/20*/status.json")]
    failures = list(folder.glob("upstream-results/**/_infrastructure_failure.json"))
    failures += list(folder.glob("upstream-results/**/_policy_error.json"))
    exit_info = read(folder / "exit.json")
    details = list((native or {}).get("details", {}).values())
    score = (native or {}).get("score")
    valid = bool(native and native.get("eval_time") == 1 and not native.get("policy_error")
                 and not failures and len(statuses) == 1 and statuses[0].get("status") == "scored"
                 and exit_info and exit_info.get("returncode") == 0
                 and len(details) == 1 and details[0].get("layout_id") == layout
                 and not details[0].get("interrupted") and not details[0].get("policy_error")
                 and type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 100
                 and native.get("success_rate") in (0, 1))
    calls, tokens, cached, output, wait, unknown_usage, unknown_cache = [], 0, 0, 0, 0., 0, 0
    for invocation_path in sorted(folder.glob("decisions/call-*/invocation.json")):
        call = invocation_path.parent
        invocation = read(invocation_path)
        execution = read(call / "execution.json") or {}
        response = read(call / "response.json") or {}
        events = list(rows(call / "events.jsonl"))
        usages = [event["usage"] for event in events if isinstance(event.get("usage"), dict)]
        known_inputs = [token_count(u.get("input_tokens")) for u in usages]
        known_outputs = [token_count(u.get("output_tokens")) for u in usages]
        known_cache = [token_count(u.get("cached_input_tokens")) for u in usages]
        usage_complete = bool(usages) and None not in known_inputs + known_outputs
        cache_complete = bool(usages) and None not in known_cache
        if not usage_complete:
            unknown_usage += 1
        if not cache_complete:
            unknown_cache += 1
        input_tokens = sum(v for v in known_inputs if v is not None)
        output_tokens = sum(v for v in known_outputs if v is not None)
        tokens += input_tokens + output_tokens
        output += output_tokens
        cached += sum(v for v in known_cache if v is not None)
        wait += execution.get("wall_seconds", 0)
        command = invocation.get("command", [])
        configured_model = command[command.index("--model") + 1] if "--model" in command else None
        calls.append({"call": call.name, "completed": response.get("status") == "completed",
                      "action_call_ids": [item["call_id"] for item in response.get("output", [])
                                          if item.get("type") == "function_call" and "call_id" in item],
                      "model": invocation.get("model"), "effort": invocation.get("effort"),
                      "command_model": configured_model, "command_effort": command_effort(command),
                      "usage_recorded": bool(usages), "usage_complete": usage_complete,
                      "cached_input_tokens": sum(known_cache) if cache_complete else None,
                      "known_input_tokens": input_tokens, "known_output_tokens": output_tokens,
                      "known_total_tokens": input_tokens + output_tokens,
                      "wall_seconds": execution.get("wall_seconds"),
                      "transport_reconnects": sum(e.get("type") == "error" and
                                                  e.get("message", "").startswith("Reconnecting...")
                                                  for e in events)})
    controls = 0
    decisions = [row for path in folder.glob("upstream-results/20*/astra/*/decisions.jsonl")
                 for row in rows(path)]
    rejections = [row for path in folder.glob("upstream-results/20*/astra/*/action_rejections.jsonl")
                  for row in rows(path)]
    guard_paths = list(folder.glob("upstream-results/eval_result/**/_ik_continuity.jsonl"))
    guard_rows = [row for path in guard_paths for row in rows(path)]
    rejection_costs = [{"rejection": rejection, "rejected_model_call": call}
                      for rejection in rejections for call in calls
                      if rejection.get("call_id") in call["action_call_ids"]]
    for path in folder.glob("upstream-results/20*/astra/*/tool_results.jsonl"):
        for row in rows(path):
            interval = row.get("executed_step_interval", [])
            if interval:
                controls = max(controls, interval[-1])
    videos = []
    if valid:
        for camera in ("head", "left_wrist", "right_wrist"):
            paths = list(native_paths[0].parent.glob(f"*cam_{camera}_*.mp4"))
            if len(paths) != 1:
                raise ValueError(f"Expected exactly one {camera} video")
            path = paths[0]
            stream = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(path)
            ], text=True))["streams"][0]
            decoded = None
            if audit_videos:
                result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                                         "-f", "null", "-"], capture_output=True, text=True)
                decoded = result.returncode == 0 and not result.stderr.strip()
                if not decoded:
                    raise ValueError(f"Video failed complete decoding: {path}")
            videos.append({"path": str(path), **stream, "full_decode_passed": decoded})
        frame_counts = {int(video["nb_frames"]) for video in videos}
        if len(frame_counts) != 1 or next(iter(frame_counts)) - 1 != controls:
            raise ValueError("Video frame counts and executed control receipt disagree")
    completed_durations = [call["wall_seconds"] for call in calls
                           if call["completed"] and call["wall_seconds"] is not None]
    return {"run_directory": str(folder), "status": "native_completed" if valid else
            "incomplete_or_infrastructure_failure" if exit_info or failures else "running",
            "valid_native_episode": valid, "native_score_0_100": score if valid else None,
            "native_success": details[0]["success"] if valid else None,
            "native_result": str(native_paths[0]) if len(native_paths) == 1 else None,
            "native_result_sha256": hashlib.sha256(native_paths[0].read_bytes()).hexdigest() if len(native_paths) == 1 else None,
            "executed_control_steps": controls, "attempted_model_calls": len(calls),
            "completed_model_calls": sum(call["completed"] for call in calls),
            "accepted_decisions": len(decisions),
            "rejected_action_tensors": sum(row.get("error") == "invalid_action_tensor" for row in rejections),
            "controller_rejections": sum(row.get("accepted") is False for row in guard_rows) if guard_paths else None,
            "controller_guard_attempts": len(guard_rows) if guard_paths else None,
            "controller_rejection_scope": "Recorded IK continuity guards: accepted=false; missing log means unknown.",
            "action_rejections": rejections,
            "action_rejection_costs": rejection_costs,
            "transport_reconnects": sum(call["transport_reconnects"] for call in calls),
            "latest_execution_note_not_verified_outcome": decisions[-1].get("execution_note") if decisions else None,
            "known_total_tokens": tokens, "known_input_tokens": tokens - output,
            "known_output_tokens": output,
            "total_tokens": tokens if calls and not unknown_usage else None,
            "cached_input_tokens": cached if calls and not unknown_cache else None,
            "known_cached_input_tokens": cached, "calls_without_complete_cache_usage": unknown_cache,
            "calls_without_usage": unknown_usage,
            "all_attempt_wall_seconds": wait,
            "completed_call_wall_seconds": sum(completed_durations),
            "completed_call_mean_seconds": statistics.mean(completed_durations) if completed_durations else None,
            "completed_call_median_seconds": statistics.median(completed_durations) if completed_durations else None,
            "model_configuration_matches": bool(calls) and all(
                call["model"] == call["command_model"] == model and
                call["effort"] == call["command_effort"] == effort for call in calls),
            "calls": calls, "videos": videos, "exit": exit_info}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-videos", action="store_true")
    args = parser.parse_args()
    folder = args.run.resolve()
    plan = read(folder / "frozen-plan.json")
    trial = summarize(folder, model=plan["model"], effort=plan["effort"], layout=plan["layout"],
                      audit_videos=args.audit_videos)
    baseline = summarize(Path(plan["baseline_run"]), model="gpt-6-astra", effort="xhigh",
                         layout=plan["layout"], audit_videos=args.audit_videos)
    source_changes = [name for name, digest in plan["source_sha256"].items()
                      if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
    output = {"schema": "roborag.model-comparison-result.v1",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "task": plan["task"], "layout": plan["layout"], "seed": plan["seed"],
              "protocol": plan["protocol"], "baseline": baseline, "trial": trial,
              "complete": trial["valid_native_episode"],
              "initial_context_comparison": read(folder / "initial-context-comparison.json"),
              "offline_audit": read(folder / "offline-audit.json"),
              "frozen_source_changes": source_changes, "limitations": plan["limitations"]}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": output["complete"], "trial": {key: value for key, value in trial.items()
                     if key not in ("calls", "videos")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
