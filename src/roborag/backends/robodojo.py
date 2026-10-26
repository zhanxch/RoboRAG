"""Reconstruct the public HexaShell EpisodeAPI over native RoboDojo.

The released perception and motion tools are imported without modification.
Only RGB, camera calibration and robot proprioception cross the tool boundary.
This is an independently implemented executor, not the unpublished server.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from hexashell.server import geometry, motion
from hexashell.server.core import (
    GRIPPER_STEPS,
    SETTLE_MAX_STEPS,
    SETTLE_TOL_RAD,
    TCP_OFFSET_M,
    WORKSPACE,
    BadRequest,
    NoDepth,
)
from PIL import Image

CAMERAS = {"cam_head": "head", "cam_left_wrist": "wrist_l", "cam_right_wrist": "wrist_r"}


def renderer_intrinsics(camera):
    """Pinhole calibration for the actual square-pixel RTX render product.

    RoboDojo config sets independent USD apertures, but RTX uses image aspect
    ratio. Camera.get_intrinsics_matrix() reads the inconsistent vertical USD
    aperture and yields fy=336 for the head image whose rendered fy is 288.
    Match RoboDojo CameraManager.get_camera_intrinsics using live horizontal
    parameters; synthetic RGB markers verify sub-pixel projection accuracy.
    """
    width, height = camera.get_resolution()
    focal = width * camera.get_focal_length() / camera.get_horizontal_aperture()
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("Invalid pinhole camera calibration")
    return np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class Arm:
    def __init__(self, manager, robot, tag):
        self._manager, self._robot, self.tag = manager, robot, tag
        self.home_joints = self.joints().copy()
        self.joint_target = self.home_joints.copy()
        self.gripper_target = 1.0
        # The upstream end link has x along the fingers, same as HexaShell.
        self.tcp_to_ee = np.eye(4)
        self.tcp_to_ee[0, 3] = -TCP_OFFSET_M

    def ee(self):
        return geometry.pose_to_matrix(self._manager.get_real_endpose(self._robot)[0])

    def tcp(self):
        return self.ee() @ np.linalg.inv(self.tcp_to_ee)

    def joints(self):
        return np.asarray(self._manager.get_joint(self._robot)[0], float)

    def gripper(self):
        return self.gripper_target


class EpisodeAPI:
    geometry = geometry
    motion = motion

    def __init__(self, env, output, *, task_instruction=False, support_robots=False):
        self._env = env
        self._output = Path(output)
        self._include_task_instruction = task_instruction
        robots = env.robot_manager.robot_list
        if support_robots:
            # The native task controls its demonstration arm. Policy commands
            # must address only the two target arms, never that support arm.
            robots = [robot for robot in robots if robot.type == "target"]
        self.arms = {
            tag: Arm(env.robot_manager, robot, tag) for tag, robot in zip(("left", "right"), robots)
        }
        if len(self.arms) != 2 or len(robots) != 2:
            raise ValueError("Expected exactly two X5 arms")
        self.commands_used = 0
        self.stop_reason = None
        self._cached_obs = None
        self._action_log = (self._output / "actions.jsonl").open("w")

    def close(self):
        self._action_log.close()

    def gripper_command_key(self, side):
        return self._env.robot_manager.process_name(self.arm(side)._robot.gripper_name)

    def dispatch_action(self, command):
        """Native dispatch owns IK rejection and updates the evaluator's counter."""
        self._env.take_action(command)

    def invalidate_observation(self):
        self._cached_obs = None

    def update_arm_targets(self, side, gripper):
        arm = self.arm(side)
        arm.joint_target = arm.joints().copy()
        arm.gripper_target = gripper

    def record_action(self, record):
        self._action_log.write(json.dumps(record) + "\n")
        self._action_log.flush()

    @property
    def step(self):
        return int(self._env.take_action_cnt[0])

    @property
    def over(self):
        return self.stop_reason is not None or self._env.is_episode_end()

    def arm(self, tag):
        if tag not in self.arms:
            raise BadRequest(f"Unknown arm {tag}")
        return self.arms[tag]

    def planner(self, tag):
        return self._env.robot_manager.ik_solver[self.arm(tag)._robot.robot_name]

    def robot_base_pose(self, tag):
        return list(self.arm(tag)._robot.entity_origin_pose)

    def sim_time_left(self):
        return max(0, int(self._env.step_lim) - self.step) / motion.CONTROL_HZ

    def stop(self, reason):
        # Native evaluator owns both success and partial credit. Never set a
        # success flag from tool status, model text or a visual impression.
        if not self._env.is_episode_end():
            self._env.success[0] = False
            self._env.is_episode_end()
        self.stop_reason = reason

    def observe(self):
        if self._cached_obs is not None:
            return self._cached_obs
        from roboicl.policy.camera_readiness import blank_cameras
        from scipy.spatial.transform import Rotation

        for _ in range(31):
            obs = self._env.get_obs()
            if not blank_cameras({0: obs}):
                break
        else:
            raise RuntimeError("RGB remained blank during render-only retries")
        result = {"png": {}, "cameras": {}, "depth": NoDepth()}
        if self._include_task_instruction:
            from roborag.tasks import public_task_instruction

            result["task_instruction"] = public_task_instruction(obs)
        manager = self._env.camera_manager
        origin = self._env.robot_manager.scene.env_origins[0].detach().cpu().numpy()
        for name, short in CAMERAS.items():
            camera = manager.cameras[0][manager.camera_names[0].index(name)]
            pos, quat = camera.get_world_pose(camera_axes="usd")
            T = np.eye(4)
            # USD camera: +X right, +Y up, -Z forward. Tools expect OpenCV.
            T[:3, :3] = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix() @ np.diag(
                [1.0, -1.0, -1.0]
            )
            T[:3, 3] = np.asarray(pos) - origin
            rgb = np.asarray(obs["vision"][name]["color"], dtype=np.uint8)
            K = renderer_intrinsics(camera)
            if list(camera.get_resolution()) != [rgb.shape[1], rgb.shape[0]]:
                raise RuntimeError("Camera calibration / image resolution mismatch")
            buffer = io.BytesIO()
            Image.fromarray(rgb[:, :, :3]).save(buffer, format="PNG")
            calibration = {
                "intrinsics": K.tolist(),
                "extrinsics_world": T.tolist(),
                "size": [rgb.shape[1], rgb.shape[0]],
            }
            # Public contract uses short names; frozen tools use native names.
            for key in (name, short):
                result["png"][key] = buffer.getvalue()
                result["cameras"][key] = calibration
        self._cached_obs = result
        return result

    def state(self):
        value = dict(
            step=self.step,
            sim_time_s=self.step / motion.CONTROL_HZ,
            sim_time_left_s=self.sim_time_left(),
            budget_left=60 - self.commands_used,
            episode_over=self.over,
            cameras=self.observe()["cameras"],
        )
        if self._include_task_instruction:
            value["task_instruction"] = self.observe()["task_instruction"]
        for tag, arm in self.arms.items():
            pose = arm.tcp()
            value[tag] = dict(
                tcp_pos=pose[:3, 3].tolist(),
                tcp_quat=geometry.matrix_to_pose(pose)[3:],
                tcp_rpy=geometry.rpy_deg(pose[:3, :3]),
                approach_dir=pose[:3, 0].tolist(),
                open_dir=pose[:3, 1].tolist(),
                gripper=arm.gripper(),
                joints=arm.joints().tolist(),
            )
        return value

    def save_observation(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, short in CAMERAS.items():
            (directory / f"{short}.png").write_bytes(self.observe()["png"][name])
        write_json(directory / "state.json", self.state())

    def run(self, sequences):
        sequences = {tag: np.asarray(seq, float) for tag, seq in sequences.items()}
        for tag, seq in sequences.items():
            if (
                seq.ndim != 2
                or seq.shape[1] != len(self.arm(tag).joints())
                or not len(seq)
                or not np.isfinite(seq).all()
            ):
                raise BadRequest("Invalid joint sequence")
        count = max((len(seq) for seq in sequences.values()), default=0)
        for index in range(count):
            if self.over:
                return False
            action = {}
            for tag, arm in self.arms.items():
                if tag in sequences:
                    arm.joint_target = sequences[tag][min(index, len(sequences[tag]) - 1)].copy()
                manager = self._env.robot_manager
                action[manager.process_name(arm._robot.arm_name)] = arm.joint_target.tolist()
                action[manager.process_name(arm._robot.gripper_name)] = [float(arm.gripper_target)]
            self._env.take_action(action)
            self._cached_obs = None
            self._action_log.write(json.dumps(dict(step=self.step, action=action)) + "\n")
            # Native termination can return from the next iteration before
            # this trajectory finishes. Persist the last dispatched step too.
            self._action_log.flush()
        return not self.over

    def hold(self, steps):
        return (
            self.run(
                {tag: np.tile(arm.joint_target, (int(steps), 1)) for tag, arm in self.arms.items()}
            )
            if steps
            else not self.over
        )

    def set_gripper(self, arm, value):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise BadRequest("Gripper opening must be in [0,1]")
        arm.gripper_target = float(value)
        return self.hold(GRIPPER_STEPS)

    def _plan(self, arm, target, joints=None, start_ee=None):
        limits = np.asarray(self._env.robot_manager.get_joint_limits(arm._robot)[0]).T
        return motion.plan_line(
            self.planner(arm.tag),
            arm._robot,
            arm.joints() if joints is None else joints,
            arm.ee() if start_ee is None else start_ee,
            target @ arm.tcp_to_ee,
            limits,
        )

    def move_tcp(self, arm, target, feedback):
        target = np.asarray(target, float).copy()
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise BadRequest("Expected finite 4x4 TCP pose")
        original = target.copy()
        for i, axis in enumerate("xyz"):
            target[i, 3] = np.clip(target[i, 3], *WORKSPACE[axis])
        feedback.update(
            plan_ok=False, workspace_limited=not np.allclose(target, original), clipped=False
        )
        if self.over:
            feedback["plan_fail_reason"] = "episode_over"
            return 3
        try:
            sequence = self._plan(arm, target)
        except motion.PlanFailure as exc:
            feedback.update(plan_fail_reason=exc.reason, plan_detail=exc.detail)
            return 2
        alive = self.run({arm.tag: sequence})
        settle = 0
        while (
            alive
            and settle < SETTLE_MAX_STEPS
            and np.max(np.abs(arm.joints() - arm.joint_target)) > SETTLE_TOL_RAD
        ):
            alive = self.hold(1)
            settle += 1
        reached = arm.tcp()
        feedback.update(
            plan_ok=True,
            plan_fail_reason=None,
            settled=bool(np.max(np.abs(arm.joints() - arm.joint_target)) <= SETTLE_TOL_RAD),
            settle_steps=settle,
            error_m=float(np.linalg.norm(reached[:3, 3] - target[:3, 3])),
            error_deg=geometry.angle_between_deg(reached[:3, :3], target[:3, :3]),
            reached_tcp=dict(
                pos=reached[:3, 3].tolist(),
                quat=geometry.matrix_to_pose(reached)[3:],
                rpy=geometry.rpy_deg(reached[:3, :3]),
            ),
        )
        return 0 if alive else 3

    def estimate_tcp_chain(self, arm, stages):
        joints, ee = arm.joints(), arm.ee()
        costs = []
        try:
            for label, target in stages:
                seq = self._plan(arm, target, joints, ee)
                costs.append(dict(stage=label, action_steps=len(seq)))
                joints, ee = seq[-1], target @ arm.tcp_to_ee
            return dict(
                estimate_ok=True,
                stage_costs=costs,
                remaining_action_steps=int(self._env.step_lim) - self.step,
            )
        except motion.PlanFailure as exc:
            return dict(
                estimate_ok=False,
                stage_costs=costs,
                plan_fail_reason=exc.reason,
                remaining_action_steps=int(self._env.step_lim) - self.step,
            )
