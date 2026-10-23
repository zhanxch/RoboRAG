"""Private simulator worker. Public clients use robodojo_server's standard MCP.

Isaac owns its Python imports and event loop; all device calls stay on this thread.
The local pipe is an implementation detail, not a second public robot protocol.
"""
import json
import os
import signal
import sys


def main():
    output_pipe = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    # Do not leave the simulator alive if its owning MCP server is killed.
    if sys.platform == "linux":
        import ctypes
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != int(os.environ["ROBORAG_SERVER_PID"]):
            raise SystemExit("MCP owner exited during startup")

    def interrupted(signum, frame):
        raise SystemExit(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    from .robodojo_runtime import open_environment
    from .robodojo import RoboDojoDevice
    with open_environment() as (env, output):
        limits = {}
        for robot in env.robot_manager.robot_list:
            if robot.type == "target":
                side = robot.arm_name.split("_")[0]
                limits[side] = env.robot_manager.get_joint_limits(robot, env_idx_list=[0])[0].tolist()
        device = RoboDojoDevice(env, joint_limits=limits)
        methods = {name: getattr(device, name) for name in
                   ("describe", "observe", "camera_observe", "move", "move_joints", "gripper", "execution_status", "stop")}
        try:
            output_pipe.write(json.dumps({"ok": True, "value": {"joint_limits": limits}}) + "\n")
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    value = methods[request["method"]](**request["arguments"])
                    result = {"ok": True, "value": value}
                    encoded = json.dumps(result, allow_nan=False)
                except Exception as exc:
                    encoded = json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                output_pipe.write(encoded + "\n")
        finally:
            device.stop()
            (output / "device-final.json").write_text(json.dumps(device.execution_status(), indent=2) + "\n")
    output_pipe.close()


if __name__ == "__main__":
    main()
