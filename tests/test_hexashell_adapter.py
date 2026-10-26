"""Contract tests for the reconstructed executor, independent of Isaac Sim."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HEXASHELL = ROOT / "vendor/HexaShell"
AVAILABLE = (HEXASHELL / "hexashell/server/core.py").is_file() and all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "transforms3d", "cv2", "PIL")
)
if AVAILABLE:
    sys.path[:0] = [str(ROOT / "src"), str(HEXASHELL)]
    import numpy as np

    from roborag.backends.robodojo import EpisodeAPI, renderer_intrinsics


@unittest.skipUnless(AVAILABLE, "Optional HexaShell checkout and policy dependencies required")
class ExecutorContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.positions = {"left": np.zeros(6), "right": np.zeros(6)}
        self.actions = []
        self.robots = [
            SimpleNamespace(arm_name=f"{tag}_arm", gripper_name=f"{tag}_gripper", tag=tag)
            for tag in ("left", "right")
        ]
        manager = SimpleNamespace(
            robot_list=self.robots,
            get_joint=lambda robot: {0: self.positions[robot.tag].copy()},
            get_real_endpose=lambda robot: {0: [0, 0, 1, 1, 0, 0, 0]},
            process_name=lambda value: value + "_joint_state",
        )
        self.env = SimpleNamespace(
            robot_manager=manager, take_action_cnt=[0], step_lim=5, success=[True], end_flag=[False]
        )

        def ended():
            if self.env.take_action_cnt[0] >= self.env.step_lim or not self.env.success[0]:
                self.env.end_flag[0] = True
                self.env.success[0] = False
            return self.env.end_flag[0]

        self.env.is_episode_end = ended

        def action(value):
            self.actions.append(value)
            for tag in self.positions:
                self.positions[tag] = np.array(value[f"{tag}_arm_joint_state"])
            self.env.take_action_cnt[0] += 1

        self.env.take_action = action
        self.api = EpisodeAPI(self.env, self.directory.name)

    def tearDown(self):
        self.api.close()
        self.directory.cleanup()

    def test_joint_dispatch_locks_arms_and_keeps_grippers(self):
        self.api.arm("left").gripper_target = 0
        self.assertTrue(
            self.api.run({"left": np.array([[0.1] * 6, [0.2] * 6]), "right": np.array([[0.3] * 6])})
        )
        self.assertEqual(len(self.actions), 2)
        self.assertEqual(self.actions[1]["right_arm_joint_state"], [0.3] * 6)
        self.assertEqual(self.actions[1]["left_gripper_joint_state"], [0.0])

    def test_native_step_limit_interrupts_long_sequence(self):
        self.assertFalse(self.api.run({"left": np.zeros((20, 6))}))
        self.assertEqual(len(self.actions), 5)
        self.assertEqual(self.api.sim_time_left(), 0)
        self.assertFalse(self.env.success[0])
        # Read without closing the API: an auto-terminated trajectory must not
        # leave its final dispatched actions in a userspace buffer.
        records = [
            json.loads(line)
            for line in (Path(self.directory.name) / "actions.jsonl").read_text().splitlines()
        ]
        self.assertEqual([r["step"] for r in records], [1, 2, 3, 4, 5])

    def test_tcp_offset_is_in_tool_frame(self):
        self.assertTrue(np.allclose(self.api.arm("left").tcp()[:3, 3], [0.145, 0, 1]))
        self.assertTrue(
            np.allclose(
                self.api.arm("left").tcp() @ self.api.arm("left").tcp_to_ee,
                self.api.arm("left").ee(),
            )
        )

    def test_tcp_transform_still_uses_each_live_pose(self):
        arm = self.api.arm("left")
        for pose in ([0, 0, 1, 1, 0, 0, 0], [0.2, -0.1, 0.8, 0.5, 0.5, 0.5, 0.5]):
            self.env.robot_manager.get_real_endpose = lambda robot: {0: pose}
            expected = arm.ee() @ np.linalg.inv(arm.tcp_to_ee)
            np.testing.assert_array_equal(arm.tcp(), expected)

    def test_support_robot_is_never_addressed_by_policy_commands(self):
        for robot in self.robots:
            robot.type = "target"
        support = SimpleNamespace(
            type="support", arm_name="support_arm0", gripper_name="support_ee0"
        )
        self.env.robot_manager.robot_list = [self.robots[0], support, self.robots[1]]
        directory = Path(self.directory.name) / "support"
        directory.mkdir()
        api = EpisodeAPI(self.env, directory, support_robots=True)
        try:
            api.hold(1)
            self.assertEqual(
                set(self.actions[-1]),
                {
                    "left_arm_joint_state",
                    "left_gripper_joint_state",
                    "right_arm_joint_state",
                    "right_gripper_joint_state",
                },
            )
        finally:
            api.close()

    def test_current_task_language_is_exposed_only_when_enabled(self):
        self.api._cached_obs = {
            "cameras": {},
            "task_instruction": "Pick up the red cup.",
            "target_label": "secret",
        }
        self.assertNotIn("task_instruction", self.api.state())
        self.api._include_task_instruction = True
        state = self.api.state()
        self.assertEqual(state["task_instruction"], "Pick up the red cup.")
        self.assertNotIn("target_label", state)

    def test_intrinsics_project_independently_rendered_markers(self):
        camera = SimpleNamespace(
            get_resolution=lambda: (640, 480),
            get_focal_length=lambda: 10.0,
            get_horizontal_aperture=lambda: 22.212,
        )
        K = renderer_intrinsics(camera)
        # RGB centroid measurements from the independent, unscored renderer
        # probe (2026-10-23 r4), not predictions copied from the implementation.
        points = np.array([[0.0, -0.2, 1.0], [0.2, 0.2, 1.0], [-0.2, 0.15, 1.0]])
        observed = np.array(
            [[319.5777778, 181.8444444], [377.0222222, 297.1333333], [261.7954545, 282.8863636]]
        )
        projected = points @ K.T
        self.assertLess(float(np.linalg.norm(projected[:, :2] - observed, axis=1).max()), 1.0)

    def test_done_uses_native_failure_path_without_steps(self):
        self.api.stop("agent_exit")
        self.assertTrue(self.api.over)
        self.assertEqual(self.api.step, 0)
