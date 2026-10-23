"""Native RoboICL evaluation controlled exclusively through the MCP worker pipe.

The native run_eval owns task registration, termination, scoring and videos.
No score or hidden object state is returned through the device interface.
"""
import base64
from io import BytesIO
import json
import os
import signal
import sys
import types

from .robodojo import RoboDojoDevice
from .arm import DeviceError


class RoboICLDevice(RoboDojoDevice):
    supports_control_step = True

    def native_observation(self):
        import numpy as np
        from PIL import Image
        from roboicl.rollout import _ready_observation
        from roboicl.policy.astra_policy import CAMERAS, STATE_KEYS, plain
        from roboicl.policy.head_calibration_capture import capture_head_calibration
        raw = _ready_observation(self.env)
        meta = {"state": {k: plain(raw["state"][k]) for k in STATE_KEYS},
                "instruction": str(raw.get("instruction", "")), "cameras": list(CAMERAS)}
        if self.seq == 0:
            meta["head_calibration_context"] = capture_head_calibration(self.env, raw)
        content = []
        for name in CAMERAS:
            buffer = BytesIO()
            Image.fromarray(np.asarray(raw["vision"][name]["color"])[:, :, :3].astype("uint8")).save(buffer, format="PNG")
            content.extend([{"type": "text", "text": name}, {"type": "image", "mimeType": "image/png",
                            "data": base64.b64encode(buffer.getvalue()).decode("ascii")}])
        return {"content": content, "structuredContent": meta}

    def control_step(self, action):
        import numpy as np
        from src.eval_client.ik_continuity import IKContinuityError
        if self.stopped or self.env.is_episode_end():
            raise DeviceError("stopped or native episode ended")
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in action.items()}
        self.last_receipt = {"skill": "control_step", "start_seq": self.seq, "status": "running"}
        try:
            self.env.take_action(arrays)
        except IKContinuityError as exc:
            self.last_receipt.update(status="rejected", executed_steps=0, end_seq=self.seq)
            error = DeviceError(str(exc))
            error.controller_event = exc.event
            raise error from exc
        except Exception:
            self.last_receipt.update(status="faulted", in_flight_step_unknown=True)
            raise
        self.seq += 1
        self.last_receipt.update(status="completed", end_seq=self.seq, executed_steps=1)
        return self.last_receipt.copy()

    def episode_status(self):
        return {"step": self.seq, "max_episode_steps": int(self.env.step_lim),
                "ended": bool(self.env.is_episode_end()), "stopped": self.stopped}


def main():
    pipe = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    if sys.platform == "linux":
        import ctypes
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != int(os.environ["ROBORAG_SERVER_PID"]):
            raise SystemExit("MCP owner exited")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    from .robodojo_runtime import open_environment
    with open_environment(register_task=False) as (env, output):
        from roboicl.policy.eval_failure import PolicyServiceError, preserve_failure
        limits = {r.arm_name.split("_")[0]: env.robot_manager.get_joint_limits(r, env_idx_list=[0])[0].tolist()
                  for r in env.robot_manager.robot_list if r.type == "target"}
        device = RoboICLDevice(env, joint_limits=limits)
        methods = {name: getattr(device, name) for name in
                   ("describe", "observe", "camera_observe", "move", "move_joints", "gripper", "execution_status",
                    "stop", "native_observation", "control_step", "episode_status")}

        def reply(value):
            pipe.write(json.dumps(value, allow_nan=False) + "\n")

        def serve(*, final=False):
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    if request["method"] == "finish_evaluation":
                        if not device.episode_status()["ended"]:
                            raise ValueError("native episode is still running; cannot finalize a partial run")
                        if not final:
                            return  # Respond only after native run_eval writes its evidence.
                        value = {"native_completed": True}
                    else:
                        value = methods[request["method"]](**request["arguments"])
                    reply({"ok": True, "value": value})
                except Exception as exc:
                    reply({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "controller_event": getattr(exc, "controller_event", None)})
            if not final:
                env._policy_failure = {"classification": "mcp_controller_disconnected", "official_score": None}
                raise PolicyServiceError("controller disconnected before native finalization")

        def episode(instance):
            reply({"ok": True, "value": {"joint_limits": limits, "supports_control_step": True}})
            serve()
        env.eval_one_episode = types.MethodType(episode, env)
        try:
            env.run_eval()
            # Only completion crosses MCP. Offline audit reads the native result file.
            (output / "evaluation.json").write_text(json.dumps({"status": "scored", "steps": device.seq,
                "result": str(__import__("pathlib").Path(env.save_dir) / "_result.json")}, indent=2) + "\n")
            reply({"ok": True, "value": {"native_completed": True}})
            serve(final=True)
        except BaseException:
            if env.video_writers:
                env._policy_failure = getattr(env, "_policy_failure", {
                    "classification": "mcp_evaluation_interrupted", "official_score": None})
                preserve_failure(env)
            raise
        finally:
            device.stop()
            (output / "device-final.json").write_text(json.dumps(device.execution_status(), indent=2) + "\n")
    pipe.close()


if __name__ == "__main__":
    main()
