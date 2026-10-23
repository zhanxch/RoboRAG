"""Preserve the RoboICL dialogue policy and rollout, using standard MCP for I/O.

This is a transport comparison, not the CaP program-generating experiment.
"""
import base64
from io import BytesIO
import json
import time

from .mcp_client import ToolError


def observation(result):
    import numpy as np
    from PIL import Image
    meta = result["structuredContent"]
    images = [item for item in result["content"] if item["type"] == "image"]
    if len(images) != len(meta["cameras"]) or any(item["mimeType"] != "image/png" for item in images):
        raise ValueError("RoboICL comparison requires one lossless image per camera")
    value = {"state": meta["state"], "instruction": meta["instruction"],
             "vision": {name: {"color": np.asarray(Image.open(BytesIO(base64.b64decode(item["data"], validate=True))))}
                        for name, item in zip(meta["cameras"], images)}}
    return value, meta.get("head_calibration_context")


def run_episode(mcp, policy, config, output, *, check_initial=None, preflight_only=False):
    from roboicl.policy.action_feedback import TrackingGuard, tracking_receipt
    from roboicl.policy.astra_policy import plain
    started = time.monotonic()

    def call(name, arguments=None):
        before = time.monotonic()
        try:
            result = mcp.call_tool(name, arguments)
        except Exception as exc:
            with (output / "mcp-calls.jsonl").open("a") as stream:
                stream.write(json.dumps({"tool": name, "status": "error", "error": str(exc),
                                         "seconds": time.monotonic() - before}) + "\n")
            raise
        # Image payloads are preserved in the policy observation log; avoid duplicating them here.
        with (output / "mcp-calls.jsonl").open("a") as stream:
            stream.write(json.dumps({"tool": name, "status": "returned", "arguments": arguments,
                "seconds": time.monotonic() - before, "result_bytes": len(json.dumps(result)),
                "metadata": result.get("structuredContent")}) + "\n")
        return result

    def status(): return call("episode_status")["structuredContent"]
    policy.reset()
    initial = status()
    obs, calibration = observation(call("read_observation"))
    step, turns = 0, 0
    obs["astra_meta"] = {"step": step, "max_episode_steps": initial["max_episode_steps"]}
    if config.include_head_calibration:
        obs["astra_meta"]["head_calibration_context"] = calibration
    policy.update_obs(obs)
    if check_initial:
        check_initial(policy.build_payload())
    if preflight_only:
        hold = {f"{s}_{key}": obs["state"][f"{s}_{key}"] for s in ("left", "right")
                for key in ("arm_joint_state", "ee_joint_state")}
        call("control_step", {"action": hold})
        observation(call("read_observation"))
        return {"status": "preflight_completed", "steps": status()["step"], "model_calls": 0,
                "native_score": None, "note": "Intentional partial interface check; not a scored episode."}
    while not status()["ended"]:
        if config.max_turns is not None and turns >= config.max_turns:
            raise RuntimeError("RoboICL reached max_turns")
        actions = policy.get_action()
        turns += 1
        tracking = TrackingGuard()
        for index, action in enumerate(actions):
            if status()["ended"]:
                break
            try:
                call("control_step", {"action": plain(action)})
            except ToolError as exc:
                event = exc.result.get("structuredContent", {}).get("controller_event")
                if not event or event.get("type") != "action_rejected_ik_continuity":
                    raise
                event.update(action_index=index, discarded_commands=len(actions) - index)
                policy.notify_action_rejection({"step": step, "event": event})
                break
            step += 1
            obs, _ = observation(call("read_observation"))
            obs["astra_meta"] = {"step": step, "max_episode_steps": initial["max_episode_steps"],
                                 "executed_action": plain(action)}
            interrupted = tracking.observe(tracking_receipt(action, obs["state"]))
            if interrupted and index + 1 < len(actions):
                obs["astra_meta"]["controller_event"] = {"type": "chunk_interrupted_tracking_error",
                    "discarded_commands": len(actions) - index - 1,
                    "reason": "Two consecutive large arm target residuals; reobserve and replan."}
            policy.update_obs(obs)
            if interrupted:
                break
        (output / "progress.json").write_text(json.dumps({"steps": step, "turns": turns,
            "seconds": time.monotonic() - started, "status": "running"}, indent=2) + "\n")
    policy.finalize()
    call("finish_evaluation")
    final = {"steps": step, "turns": turns, "seconds": time.monotonic() - started, "status": "native_completed"}
    (output / "progress.json").write_text(json.dumps(final, indent=2) + "\n")
    return final
