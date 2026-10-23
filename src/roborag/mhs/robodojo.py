"""Synchronous MHS driver for an already initialized single RoboDojo environment.

The caller owns simulator setup/teardown and must run this on its simulator thread.
No task internals, object truth or evaluation score enter observations.
"""
import math
from uuid import uuid4

from .device import DeviceError, description, opening01, position3


class RoboDojoDevice:
    def __init__(self, env, *, step_m=0.02, max_move_m=0.5, max_steps=64,
                 gripper_steps=12, tolerance_m=0.03, encode_images=None):
        if getattr(env, "num_envs", 1) != 1:
            raise ValueError("MHS v0.1 supports one environment per session")
        for value in (step_m, max_move_m, tolerance_m):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("motion limits must be positive finite numbers")
        if any(type(x) is not int or x < 1 for x in (max_steps, gripper_steps)):
            raise ValueError("step limits must be positive integers")
        if gripper_steps > max_steps:
            raise ValueError("gripper_steps exceeds max_steps")
        self.env, self.encode_images = env, encode_images
        self.step_m, self.max_move_m = step_m, max_move_m
        self.max_steps, self.gripper_steps, self.tolerance_m = max_steps, gripper_steps, tolerance_m
        self.session_id, self.seq, self.stopped = uuid4().hex, 0, False

    def describe(self):
        return description("robodojo-arx-x5", ["left", "right"], "environment_origin_world_axes",
                           backend="robodojo", limits={
                               "controlled_link": "link6 (not fingertip/TCP)",
                               "orientation": "hold observed quaternion wxyz; rotation not exposed in v0.1",
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
                "arms": {side: {"position": list(map(float, state[f"{side}_ee_pose"][:3])),
                                "quaternion_wxyz": list(map(float, state[f"{side}_ee_pose"][3:])),
                                "gripper_commanded": float(state[f"{side}_ee_joint_state"][0]),
                                "gripper_measured": None} for side in ("left", "right")},
                "images": self.encode_images(raw) if self.encode_images else [],
                "stopped": self.stopped}

    def _check(self, arm):
        if self.stopped:
            raise DeviceError("session stopped")
        if arm not in ("left", "right"):
            raise ValueError("unknown arm")
        if self.env.is_episode_end():
            raise DeviceError("environment episode ended")

    def _execute(self, arm, target, opening, steps, skill):
        import numpy as np
        initial = self._raw()["state"]
        start = list(map(float, initial[f"{arm}_ee_pose"][:3]))
        before = self.seq
        try:
            for index in range(steps):
                self._check(arm)
                action = {}
                for side in ("left", "right"):
                    action[f"{side}_ee_pose"] = np.asarray(initial[f"{side}_ee_pose"], dtype=float).copy()
                    action[f"{side}_ee_joint_state"] = np.asarray(initial[f"{side}_ee_joint_state"], dtype=float).copy()
                if target is not None:
                    action[f"{arm}_ee_pose"][:3] = [a + (b - a) * (index + 1) / steps for a, b in zip(start, target)]
                if opening is not None:
                    action[f"{arm}_ee_joint_state"][:] = opening
                self.env.take_action(action)
                self.seq += 1
                state = self._raw()["state"]
                if any(math.dist(action[f"{side}_ee_pose"][:3], state[f"{side}_ee_pose"][:3]) > self.tolerance_m
                       for side in ("left", "right")):
                    raise DeviceError("EE tracking residual exceeds tolerance")
        except Exception as exc:
            raise DeviceError(f"{skill} aborted: executed_steps={self.seq - before}, state_seq={self.seq}; {exc}") from exc
        return {"session_id": self.session_id, "skill": skill, "status": "completed",
                "start_seq": before, "end_seq": self.seq, "executed_steps": self.seq - before,
                "task_success": None}

    def move(self, arm, position):
        self._check(arm)
        target = position3(position)
        start = self._raw()["state"][f"{arm}_ee_pose"][:3]
        distance = math.dist(start, target)
        steps = max(1, math.ceil(distance / self.step_m))
        if distance > self.max_move_m or steps > self.max_steps:
            raise ValueError("move exceeds configured distance/step budget")
        return self._execute(arm, target, None, steps, "move")

    def gripper(self, arm, opening):
        self._check(arm)
        return self._execute(arm, None, opening01(opening), self.gripper_steps, "gripper")

    def stop(self):
        self.stopped = True
        return {"session_id": self.session_id, "status": "stopped", "state_seq": self.seq}
