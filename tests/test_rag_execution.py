from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

from roborag.rag_execution import execute_chunk, execute_delta_actions, expand_delta_actions

HEXASHELL = Path(__file__).resolve().parents[1] / "vendor/HexaShell"
sys.path.insert(0, str(HEXASHELL))


class DeltaEncodingTests(unittest.TestCase):
    def test_expansion_preserves_rows_and_does_not_alias_input(self):
        move = [0,0,0,0,0,0,1, .01,0,0,0,0,0,0]
        hold = [0,0,0,0,0,0,1, 0,0,0,0,0,0,0]
        actions = expand_delta_actions(segments=[dict(delta=move,repeat=3),dict(delta=hold,repeat=2)])
        self.assertEqual(actions,[move,move,move,hold,hold])
        actions[0][7]=.02
        self.assertEqual(actions[1][7],.01)
        self.assertEqual(move[7],.01)

    def test_long_expansion_has_no_24_row_cap(self):
        row = [0]*14
        self.assertEqual(expand_delta_actions(segments=[dict(delta=row, repeat=30), dict(delta=row, repeat=26)]),
                         [row]*56)
        self.assertEqual(expand_delta_actions(actions=[row]*56), [row]*56)

    def test_invalid_tail_and_repeat_are_rejected(self):
        row = [0]*14
        for segments in ([dict(delta=row,repeat=0)],
                         [dict(delta=row,repeat=1),dict(delta=[0]*13,repeat=1)],
                         [dict(delta=row,repeat=True)]):
            with self.assertRaises(ValueError):
                expand_delta_actions(segments=segments)


@unittest.skipUnless((HEXASHELL / "hexashell/server/core.py").exists(), "Optional HexaShell checkout required")
class ChunkContract(unittest.TestCase):
    def test_last_failed_waypoint_still_exhausts_budget(self):
        arm = SimpleNamespace(tcp=lambda: np.eye(4))
        api = SimpleNamespace(step=0, over=False, commands_used=59, arm=lambda side:arm,
                              move_tcp=lambda *args:2)
        stopped = []
        def stop(reason):
            api.over = True
            stopped.append(reason)
        api.stop = stop
        execute_chunk(api, [{"arm":"right", "xyz":[0,.1,.9]}])
        self.assertEqual(stopped, ["waypoint_budget"])
        self.assertEqual(api.commands_used, 60)
        execute_chunk(api, [{"arm":"right", "xyz":[0,.1,.9]}])
        self.assertEqual(api.commands_used, 60)

    def test_delta_targets_accumulate_from_commands_despite_observed_lag(self):
        import io
        from hexashell.server import geometry
        from scipy.spatial.transform import Rotation
        poses = {side:np.eye(4) for side in ("left", "right")}
        arms = {side:SimpleNamespace(ee=lambda side=side:poses[side], joints=lambda:np.zeros(6),
            _robot=SimpleNamespace(gripper_name=side+"_gripper")) for side in poses}
        api = SimpleNamespace(step=0, over=False, arm=lambda side:arms[side], geometry=geometry,
                              _action_log=io.StringIO())
        commands = []
        def take_action(command):
            commands.append(command)
            api.step += 1
            for side in poses:
                poses[side] = geometry.pose_to_matrix(command[side+"_ee_pose"])
                poses[side][0,3] -= .003  # actual plant lags every requested target
        api._env = SimpleNamespace(take_action=take_action,
            robot_manager=SimpleNamespace(process_name=lambda name:name+"_joint_state"))
        row = [0,0,0,0,0,0,1, .01,0,0,0,0,.1,0]
        result = execute_delta_actions(api, [row]*30)
        self.assertAlmostEqual(commands[1]["right_ee_pose"][0], .02)
        self.assertTrue(np.allclose(geometry.pose_to_matrix(commands[1]["right_ee_pose"])[:3,:3],
                                   Rotation.from_rotvec([0,0,.2]).as_matrix()))
        self.assertEqual(result["executed_rows"], 30)
        self.assertEqual(commands[1]["right_gripper_joint_state"], [0])

    def test_failed_motion_does_not_close_gripper_or_run_remaining_waypoints(self):
        moves, grips = [], []
        arm = SimpleNamespace(tcp=lambda: np.eye(4))
        def fail(arm, target, feedback):
            moves.append(target)
            feedback.update(plan_ok=False, plan_fail_reason="ik_unreachable")
            return 2
        api = SimpleNamespace(step=0, over=False, commands_used=0, arm=lambda side:arm,
            move_tcp=fail, set_gripper=lambda arm, value:grips.append(value))
        result = execute_chunk(api, [{"arm":"right", "xyz":[0,.1,.9], "gripper":0},
                                     {"arm":"right", "xyz":[0,.1,1.0]}])
        self.assertEqual(len(moves), 1)
        self.assertEqual(grips, [])
        self.assertTrue(result["waypoints"][0]["stopped_tail"])

    def test_long_delta_chunks_still_stop_at_native_end_or_tracking_error(self):
        import io

        from hexashell.server import geometry

        from roborag.action_targets import compile_targets
        for cause in ("native_end", "tracking_error", "invalid_tail"):
            with self.subTest(cause=cause):
                poses = {side: np.eye(4) for side in ("left", "right")}
                arms = {side: SimpleNamespace(ee=lambda side=side: poses[side], joints=lambda: np.zeros(6),
                    _robot=SimpleNamespace(gripper_name=side+"_gripper")) for side in poses}
                api = SimpleNamespace(step=0, over=False, arm=lambda side: arms[side], geometry=geometry,
                                      _action_log=io.StringIO())
                commands = []
                def take_action(command):
                    commands.append(command)
                    api.step += 1
                    for side in poses:
                        poses[side] = geometry.pose_to_matrix(command[side+"_ee_pose"])
                    if api.step == 3:
                        if cause == "native_end":
                            api.over = True
                        else:
                            poses["left"][0, 3] -= .05
                api._env = SimpleNamespace(take_action=take_action,
                    robot_manager=SimpleNamespace(process_name=lambda name: name+"_joint_state"))
                state = {f"{side}_{name}": value for side in poses for name, value in
                         (("ee_pose", [0, 0, 0, 1, 0, 0, 0]), ("ee_joint_state", [1]))}
                rows = compile_targets(state, [dict(steps=2, left=dict(tcp_xyz=[1.195, 0, 0]))])
                self.assertEqual(len(rows), 30)  # auto-extension also exceeds the old cap
                if cause == "invalid_tail":
                    with self.assertRaises(ValueError):
                        execute_delta_actions(api, rows+[[0]*13])
                    self.assertEqual(commands, [])
                else:
                    result = execute_delta_actions(api, rows)
                    self.assertEqual(len(commands), 3)
                    self.assertEqual(result["submitted_rows"], 30)
                    self.assertEqual(result["executed_rows"], 3)
                    self.assertEqual(result["discarded_rows"], 27)
                    self.assertEqual(result["episode_over"], cause == "native_end")

    def test_native_termination_during_home_stops_tail(self):
        joints = np.full(6, .2)
        arm = SimpleNamespace(tag="right", joints=lambda:joints, home_joints=np.zeros(6))
        dispatched = []
        api = SimpleNamespace(step=0, over=False, commands_used=0, arm=lambda side:arm,
            motion=SimpleNamespace(time_path=lambda path:path))
        def run(sequences):
            dispatched.append(sequences)
            api.step += 1
            api.over = True
        api.run = run
        result = execute_chunk(api, [{"arm":"right", "home":True, "gripper":1},
                                     {"arm":"left", "home":True}])
        self.assertEqual(len(dispatched), 1)
        self.assertTrue(np.array_equal(dispatched[0]["right"][-1], arm.home_joints))
        self.assertEqual(result["executed_waypoints"], 1)
        self.assertTrue(result["episode_over"])


if __name__ == "__main__":
    unittest.main()
