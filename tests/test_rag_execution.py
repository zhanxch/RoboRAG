import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from roborag.control.execution import execute_delta_actions, expand_delta_actions
from roborag.control.homing import execute_home

HEXASHELL = Path(__file__).resolve().parents[1] / "vendor/HexaShell"
sys.path.insert(0, str(HEXASHELL))


class DeltaEncodingTests(unittest.TestCase):
    def test_home_opens_after_joint_motion_and_keeps_budget_receipt(self):
        events = []
        arm = SimpleNamespace(tag="left", joints=lambda: np.zeros(6), home_joints=np.zeros(6))
        api = SimpleNamespace(
            step=0,
            over=False,
            commands_used=59,
            arm=lambda _: arm,
            motion=SimpleNamespace(time_path=lambda path: path),
            run=lambda sequence: events.append("joint motion"),
            set_gripper=lambda arm, value: events.append(("gripper", value)),
        )

        def stop(reason):
            events.append(reason)
            api.over = True

        api.stop = stop
        result = execute_home(
            api, [dict(arm=side, home=True, gripper=1) for side in ("left", "right")]
        )
        self.assertEqual(events, ["joint motion", ("gripper", 1), "waypoint_budget"])
        self.assertEqual(result["executed_waypoints"], 1)
        self.assertEqual(api.commands_used, 60)

    def test_arbitrary_tcp_request_is_rejected_before_home_dispatch(self):
        api = SimpleNamespace()
        with self.assertRaises(ValueError):
            execute_home(
                api, [dict(arm="left", home=True, gripper=1), dict(arm="right", xyz=[0, 0, 1])]
            )

    def test_expansion_preserves_rows_and_does_not_alias_input(self):
        move = [0, 0, 0, 0, 0, 0, 1, 0.01, 0, 0, 0, 0, 0, 0]
        hold = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        actions = expand_delta_actions(
            segments=[dict(delta=move, repeat=3), dict(delta=hold, repeat=2)]
        )
        self.assertEqual(actions, [move, move, move, hold, hold])
        actions[0][7] = 0.02
        self.assertEqual(actions[1][7], 0.01)
        self.assertEqual(move[7], 0.01)

    def test_invalid_tail_and_total_horizon_are_rejected(self):
        row = [0] * 14
        for segments in (
            [dict(delta=row, repeat=24), dict(delta=row, repeat=1)],
            [dict(delta=row, repeat=1), dict(delta=[0] * 13, repeat=1)],
            [dict(delta=row, repeat=True)],
        ):
            with self.assertRaises(ValueError):
                expand_delta_actions(segments=segments)


@unittest.skipUnless(
    (HEXASHELL / "hexashell/server/core.py").exists(), "Optional HexaShell checkout required"
)
class ChunkContract(unittest.TestCase):
    def test_predispatch_rejection_preserves_prefix_and_does_not_close_either_gripper(self):
        import io

        from hexashell.server import geometry

        class Rejected(ValueError):
            event = {"type": "action_rejected_ik_continuity", "rejected_action_executed": False}

        poses = {side: np.eye(4) for side in ("left", "right")}
        arms = {
            side: SimpleNamespace(
                ee=lambda side=side: poses[side],
                joints=lambda: np.zeros(6),
                gripper_target=1.0,
                _robot=SimpleNamespace(gripper_name=side + "_gripper"),
            )
            for side in poses
        }
        api = SimpleNamespace(
            step=0,
            over=False,
            arm=lambda side: arms[side],
            geometry=geometry,
            _action_log=io.StringIO(),
        )
        accepted = []

        def take_action(command):
            if command["right_gripper_joint_state"] == [0]:
                raise Rejected()
            accepted.append(command)
            api.step += 1
            for side in poses:
                poses[side] = geometry.pose_to_matrix(command[side + "_ee_pose"])

        api._env = SimpleNamespace(
            take_action=take_action,
            robot_manager=SimpleNamespace(process_name=lambda name: name + "_joint_state"),
        )
        api.gripper_command_key = lambda side: side + "_gripper_joint_state"
        api.dispatch_action = take_action
        api.invalidate_observation = lambda: None

        def update_targets(side, gripper):
            arms[side].joint_target = arms[side].joints().copy()
            arms[side].gripper_target = gripper

        api.update_arm_targets = update_targets
        api.record_action = lambda record: api._action_log.write(str(record) + "\n")
        first = [0, 0, 0, 0, 0, 0, 1, 0.01, 0, 0, 0, 0, 0, 1]
        rejected = [0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0]
        result = execute_delta_actions(api, [first, rejected, first], rejection_errors=(Rejected,))
        self.assertEqual(result["executed_rows"], 1)
        self.assertEqual(result["discarded_rows"], 2)
        self.assertEqual(result["last_command"], accepted[0])
        self.assertEqual(result["step"], 1)
        self.assertFalse(result["rejection"]["rejected_action_executed"])
        self.assertEqual([arm.gripper_target for arm in arms.values()], [1.0, 1.0])
        self.assertEqual(len(api._action_log.getvalue().splitlines()), 1)
        # Non-opt-in users retain the original error behavior.
        with self.assertRaises(Rejected):
            execute_delta_actions(api, [rejected])

    def test_delta_targets_accumulate_from_commands_despite_observed_lag(self):
        import io

        from hexashell.server import geometry
        from scipy.spatial.transform import Rotation

        poses = {side: np.eye(4) for side in ("left", "right")}
        arms = {
            side: SimpleNamespace(
                ee=lambda side=side: poses[side],
                joints=lambda: np.zeros(6),
                _robot=SimpleNamespace(gripper_name=side + "_gripper"),
            )
            for side in poses
        }
        api = SimpleNamespace(
            step=0,
            over=False,
            arm=lambda side: arms[side],
            geometry=geometry,
            _action_log=io.StringIO(),
        )
        commands = []

        def take_action(command):
            commands.append(command)
            api.step += 1
            for side in poses:
                poses[side] = geometry.pose_to_matrix(command[side + "_ee_pose"])
                poses[side][0, 3] -= 0.003  # actual plant lags every requested target

        api._env = SimpleNamespace(
            take_action=take_action,
            robot_manager=SimpleNamespace(process_name=lambda name: name + "_joint_state"),
        )
        api.gripper_command_key = lambda side: side + "_gripper_joint_state"
        api.dispatch_action = take_action
        api.invalidate_observation = lambda: None

        def update_targets(side, gripper):
            arms[side].joint_target = arms[side].joints().copy()
            arms[side].gripper_target = gripper

        api.update_arm_targets = update_targets
        api.record_action = lambda record: api._action_log.write(str(record) + "\n")
        row = [0, 0, 0, 0, 0, 0, 1, 0.01, 0, 0, 0, 0, 0.1, 0]
        result = execute_delta_actions(api, [row, row])
        self.assertAlmostEqual(commands[1]["right_ee_pose"][0], 0.02)
        self.assertTrue(
            np.allclose(
                geometry.pose_to_matrix(commands[1]["right_ee_pose"])[:3, :3],
                Rotation.from_rotvec([0, 0, 0.2]).as_matrix(),
            )
        )
        self.assertEqual(result["executed_rows"], 2)
        self.assertEqual(commands[1]["right_gripper_joint_state"], [0])

    def test_native_termination_during_home_stops_tail(self):
        joints = np.full(6, 0.2)
        arm = SimpleNamespace(tag="right", joints=lambda: joints, home_joints=np.zeros(6))
        dispatched = []
        api = SimpleNamespace(
            step=0,
            over=False,
            commands_used=0,
            arm=lambda side: arm,
            motion=SimpleNamespace(time_path=lambda path: path),
        )

        def run(sequences):
            dispatched.append(sequences)
            api.step += 1
            api.over = True

        api.run = run
        result = execute_home(
            api,
            [
                {"arm": "right", "home": True, "gripper": 1},
                {"arm": "left", "home": True, "gripper": 1},
            ],
        )
        self.assertEqual(len(dispatched), 1)
        self.assertTrue(np.array_equal(dispatched[0]["right"][-1], arm.home_joints))
        self.assertEqual(result["executed_waypoints"], 1)
        self.assertTrue(result["episode_over"])


if __name__ == "__main__":
    unittest.main()
