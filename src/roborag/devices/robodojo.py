"""Synchronous adapter for an already initialized single RoboDojo environment.

The caller owns simulator setup/teardown and must run this on its simulator thread.
No task internals, object truth or evaluation score enter observations.
"""
import math
from uuid import uuid4

from .arm import DeviceError, description, opening01, position3


class RoboDojoDevice:
    def __init__(self, env, *, step_m=0.02, max_move_m=0.5, max_steps=64,
                 gripper_steps=12, tolerance_m=0.03, encode_images=None, joint_limits=None):
        if getattr(env, "num_envs", 1) != 1:
            raise ValueError("this RoboDojo adapter supports one environment per session")
        for value in (step_m, max_move_m, tolerance_m):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("motion limits must be positive finite numbers")
        if any(type(x) is not int or x < 1 for x in (max_steps, gripper_steps)):
            raise ValueError("step limits must be positive integers")
        if gripper_steps > max_steps:
            raise ValueError("gripper_steps exceeds max_steps")
        self.env, self.encode_images = env, encode_images
        self.joint_limits = joint_limits
        self.last_receipt = None
        self.step_m, self.max_move_m = step_m, max_move_m
        self.max_steps, self.gripper_steps, self.tolerance_m = max_steps, gripper_steps, tolerance_m
        self.session_id, self.seq, self.stopped = uuid4().hex, 0, False

    def describe(self):
        return description("robodojo-arx-x5", ["left", "right"], "environment_origin_world_axes",
                           backend="robodojo", limits={
                               "controlled_link": "link6 (not fingertip/TCP)",
                               "orientation": "quaternion wxyz, shortest-arc interpolation; omitted means hold",
                               "step_translation_m": self.step_m, "max_move_m": self.max_move_m,
                               "max_steps_per_skill": self.max_steps, "gripper_steps": self.gripper_steps,
                               "arrival_tolerance_m": self.tolerance_m,
                               "collision_detection": "not provided by this adapter",
                               "stop": "latched no-further-dispatch; no asynchronous emergency stop"})

    def _raw(self):
        return self.env.get_obs()

    def observe(self):
        raw = self._raw()
        state = raw["state"]
        return {"session_id": self.session_id, "state_seq": self.seq,
                "frame": "environment_origin_world_axes", "source": "simulator_proprioception",
                "instruction": str(raw.get("instruction", "")),
                "arms": {side: {"position": list(map(float, state[f"{side}_ee_pose"][:3])),
                                "quaternion_wxyz": list(map(float, state[f"{side}_ee_pose"][3:])),
                                "gripper_commanded": float(state[f"{side}_ee_joint_state"][0]),
                                "gripper_measured": None} for side in ("left", "right")},
                "joints": {side: list(map(float, state.get(f"{side}_arm_joint_state", [])))
                           for side in ("left", "right")},
                "images": self.encode_images(raw) if self.encode_images else [],
                "stopped": self.stopped}

    def _check(self, arm):
        if self.stopped:
            raise DeviceError("session stopped")
        if arm not in ("left", "right"):
            raise ValueError("unknown arm")
        if self.env.is_episode_end():
            raise DeviceError("environment episode ended")

    def _execute(self, arm, target, opening, steps, skill, orientation=None, joints=None):
        import numpy as np
        initial = self._raw()["state"]
        start = list(map(float, initial[f"{arm}_ee_pose"][:3]))
        before = self.seq
        self.last_receipt = {"skill": skill, "status": "running", "start_seq": before}
        dispatch_in_flight = False
        try:
            for index in range(steps):
                self._check(arm)
                action = {}
                for side in ("left", "right"):
                    key = f"{side}_arm_joint_state" if joints is not None else f"{side}_ee_pose"
                    action[key] = np.asarray(initial[key], dtype=float).copy()
                    action[f"{side}_ee_joint_state"] = np.asarray(initial[f"{side}_ee_joint_state"], dtype=float).copy()
                if target is not None:
                    action[f"{arm}_ee_pose"][:3] = [a + (b - a) * (index + 1) / steps for a, b in zip(start, target)]
                if orientation is not None:
                    action[f"{arm}_ee_pose"][3:] = slerp(initial[f"{arm}_ee_pose"][3:], orientation, (index + 1) / steps)
                if joints is not None:
                    source = initial[f"{arm}_arm_joint_state"]
                    action[f"{arm}_arm_joint_state"] = np.asarray(source) + (np.asarray(joints) - source) * (index + 1) / steps
                if opening is not None:
                    action[f"{arm}_ee_joint_state"][:] = opening
                dispatch_in_flight = True
                self.env.take_action(action)
                dispatch_in_flight = False
                self.seq += 1
                state = self._raw()["state"]
                if joints is None:
                    if any(math.dist(action[f"{side}_ee_pose"][:3], state[f"{side}_ee_pose"][:3]) > self.tolerance_m
                           or quaternion_angle(action[f"{side}_ee_pose"][3:], state[f"{side}_ee_pose"][3:]) > 0.2
                           for side in ("left", "right")):
                        raise DeviceError("EE tracking residual exceeds tolerance")
                elif any(max(abs(a - b) for a, b in zip(action[f"{s}_arm_joint_state"], state[f"{s}_arm_joint_state"])) > 0.2
                         for s in ("left", "right")):
                    raise DeviceError("joint tracking residual exceeds tolerance")
        except Exception as exc:
            self.last_receipt.update(status="faulted", end_seq=self.seq, executed_steps=self.seq - before,
                                     in_flight_step_unknown=dispatch_in_flight, error=str(exc))
            raise DeviceError(f"{skill} aborted: executed_steps={self.seq - before}, state_seq={self.seq}; {exc}") from exc
        self.last_receipt = {"session_id": self.session_id, "skill": skill, "status": "completed",
                "start_seq": before, "end_seq": self.seq, "executed_steps": self.seq - before,
                "task_success": None}
        return self.last_receipt.copy()

    def move(self, arm, position, quaternion_wxyz=None):
        self._check(arm)
        target = position3(position)
        pose = self._raw()["state"][f"{arm}_ee_pose"]
        start = pose[:3]
        distance = math.dist(start, target)
        steps = max(1, math.ceil(distance / self.step_m))
        if quaternion_wxyz is not None:
            quaternion_wxyz = quaternion(quaternion_wxyz)
            steps = max(steps, math.ceil(quaternion_angle(pose[3:], quaternion_wxyz) / 0.1))
        if distance > self.max_move_m or steps > self.max_steps:
            raise ValueError("move exceeds configured distance/step budget")
        return self._execute(arm, target, None, steps, "move_ee", orientation=quaternion_wxyz)

    def move_joints(self, arm, positions):
        self._check(arm)
        if not self.joint_limits or arm not in self.joint_limits:
            raise ValueError("joint limits unavailable; joint control is not exposed")
        limits = self.joint_limits[arm]
        if (not isinstance(positions, list) or len(positions) != len(limits)
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in positions)):
            raise ValueError("positions must match the finite joint vector in radians")
        if any(not lo <= q <= hi for q, (lo, hi) in zip(positions, limits)):
            raise ValueError("joint target outside native limits")
        current = self._raw()["state"][f"{arm}_arm_joint_state"]
        steps = max(1, math.ceil(max(abs(a - b) for a, b in zip(current, positions)) / 0.05))
        if steps > self.max_steps:
            raise ValueError("joint trajectory exceeds step budget")
        return self._execute(arm, None, None, steps, "move_joints", joints=positions)

    def execution_status(self):
        return {"session_id": self.session_id, "state_seq": self.seq,
                "stopped": self.stopped, "last_receipt": self.last_receipt}

    def camera_observe(self, cameras=None):
        import base64
        from io import BytesIO
        import numpy as np
        from PIL import Image
        raw = self._raw()
        available = raw.get("vision", {})
        selected = list(available) if cameras is None else cameras
        if not isinstance(selected, list) or not selected or any(c not in available for c in selected):
            raise ValueError(f"select available cameras: {list(available)}")
        content = []
        for camera in selected:
            rgb = np.asarray(available[camera]["color"])
            buffer = BytesIO()
            image = Image.fromarray(rgb[:, :, :3].astype("uint8"))
            image.thumbnail((640, 480))
            image.save(buffer, format="JPEG", quality=85)
            content.extend([{"type": "text", "text": f"{camera}; state_seq={self.seq}"},
                            {"type": "image", "mimeType": "image/jpeg",
                             "data": base64.b64encode(buffer.getvalue()).decode("ascii")}])
        return content

    def gripper(self, arm, opening):
        self._check(arm)
        return self._execute(arm, None, opening01(opening), self.gripper_steps, "gripper")

    def stop(self):
        self.stopped = True
        return {"session_id": self.session_id, "status": "stopped", "state_seq": self.seq}


def quaternion(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError("quaternion must be four finite numbers in wxyz order")
    norm = math.sqrt(sum(x * x for x in value))
    if abs(norm - 1) > 0.05:
        raise ValueError("quaternion must be unit length")
    return [x / norm for x in value]


def quaternion_angle(a, b):
    return 2 * math.acos(min(1, abs(sum(x * y for x, y in zip(a, b))) /
                             (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)))))


def slerp(a, b, t):
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b, dot = [-x for x in b], -dot
    if dot > 0.9995:
        value = [(1 - t) * x + t * y for x, y in zip(a, b)]
    else:
        angle = math.acos(min(1, dot))
        value = [(math.sin((1 - t) * angle) * x + math.sin(t * angle) * y) / math.sin(angle) for x, y in zip(a, b)]
    norm = math.sqrt(sum(x*x for x in value))
    return [x / norm for x in value]
