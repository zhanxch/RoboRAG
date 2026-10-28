import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from roborag.action_targets import compile_targets, compile_targets_with_timing, TCP_OFFSET_M
from roborag.rag_execution import validate_delta_actions


class ActionTargetsTests(unittest.TestCase):
    def state(self):
        return {f"{side}_{name}":value[:] for side in ("left","right")
                for name,value in (("ee_pose",[0,0,1,1,0,0,0]),("ee_joint_state",[1]))}

    def test_accumulated_rows_reach_requested_tcp_and_rotation(self):
        desired=[.2,.1,1.1]
        quat=[math.cos(.3),0,0,math.sin(.3)]
        rows=compile_targets(self.state(),[dict(steps=10,left=dict(tcp_xyz=desired,quaternion_wxyz=quat,gripper=0)),
                                          dict(steps=3,left=dict(gripper=1))])
        position=np.array([0.,0.,1.]); rotation=Rotation.identity()
        for row in rows:
            position+=row[:3]
            rotation=Rotation.from_rotvec(row[3:6])*rotation
            self.assertTrue(np.allclose(row[7:13],0))
            self.assertEqual(row[13],1)
        self.assertTrue(np.allclose(position+rotation.apply([TCP_OFFSET_M,0,0]),desired))
        self.assertTrue(np.allclose(rotation.as_quat(),np.array(quat)[[1,2,3,0]]))
        self.assertEqual([r[6] for r in rows],[0]*10+[1]*3)

    def test_later_segments_start_from_previous_target_and_preserve_rotation(self):
        rows=compile_targets(self.state(),[dict(steps=4,left=dict(tcp_xyz=[.185,0,1])),
                                          dict(steps=4,left=dict(tcp_xyz=[.225,0,1]))])
        self.assertTrue(np.allclose(np.asarray(rows)[:,0],.01))

    def test_invalid_late_targets_rejected_before_dispatch(self):
        for segments in ([dict(steps=0,left=dict(gripper=0))],
                         [dict(steps=True,left=dict(gripper=0))],
                         [dict(steps=2,left=dict(gripper=0)),dict(steps=2,right=dict(tcp_xyz=[0,0,float('nan')]))]):
            with self.assertRaises(ValueError):
                compile_targets(self.state(),segments)

    def test_long_segments_and_empty_holds_preserve_targets_and_grippers(self):
        rows = compile_targets(self.state(), [dict(steps=30, left=dict(tcp_xyz=[.245, 0, 1], gripper=0)),
                                              dict(steps=4, left={}), dict(steps=3)])
        self.assertEqual(len(rows), 37)
        self.assertTrue(np.allclose(np.asarray(rows)[:30, 0], .1 / 30))
        self.assertTrue(np.allclose(np.asarray(rows)[30:, :6], 0))
        self.assertTrue(np.allclose(np.asarray(rows)[:, 7:13], 0))
        self.assertTrue(all(r[6] == 0 and r[13] == 1 for r in rows))

    def test_two_steps_extend_to_ten_and_preserve_later_targets_and_gripper_events(self):
        segments = [dict(steps=2, left=dict(tcp_xyz=[.495, 0, 1])),
                    dict(steps=3, left=dict(gripper=0)),
                    dict(steps=1, left=dict(tcp_xyz=[.505, 0, 1]))]
        rows, timing = compile_targets_with_timing(self.state(), segments)
        self.assertEqual(timing, [dict(segment=1, requested_steps=2, minimum_steps=10, compiled_steps=10),
                                  dict(segment=2, requested_steps=3, minimum_steps=1, compiled_steps=3),
                                  dict(segment=3, requested_steps=1, minimum_steps=1, compiled_steps=1)])
        self.assertEqual(segments[0]["steps"], 2)  # original model request remains intact
        self.assertEqual(len(rows), 14)
        self.assertTrue(np.allclose(np.asarray(rows)[:10, 0], .035))
        self.assertTrue(np.allclose(np.asarray(rows)[10:13, :6], 0))
        self.assertAlmostEqual(rows[-1][0], .01)
        self.assertEqual([row[6] for row in rows], [1]*10 + [0]*4)
        self.assertTrue(np.allclose(np.asarray(rows)[:, 7:13], 0))
        validate_delta_actions(rows)

    def test_both_arms_share_steps_when_right_rotation_is_limiting(self):
        rotation = Rotation.from_rotvec([0, 0, 1.5])
        right_tcp = (np.array([0, 0, 1]) + rotation.apply([TCP_OFFSET_M, 0, 0])).tolist()
        rows, timing = compile_targets_with_timing(self.state(), [dict(steps=2,
            left=dict(tcp_xyz=[.215, 0, 1], gripper=0),
            right=dict(tcp_xyz=right_tcp, quaternion_wxyz=rotation.as_quat()[[3, 0, 1, 2]].tolist()))])
        self.assertEqual(timing[0]["compiled_steps"], 5)
        self.assertEqual(len(rows), 5)
        self.assertTrue(np.allclose(np.asarray(rows)[:, 0], .07/5))
        self.assertTrue(np.allclose(np.asarray(rows)[:, 12], 1.5/5))
        self.assertTrue(all(row[6] == 0 and row[13] == 1 for row in rows))
        for side, offset, expected_tcp, expected_rotation in (
                ("left", 0, [.215, 0, 1], Rotation.identity()), ("right", 7, right_tcp, rotation)):
            with self.subTest(side=side):
                position, orientation = np.array([0., 0., 1.]), Rotation.identity()
                for row in rows:
                    position += row[offset:offset+3]
                    orientation = Rotation.from_rotvec(row[offset+3:offset+6])*orientation
                self.assertTrue(np.allclose(position+orientation.apply([TCP_OFFSET_M, 0, 0]), expected_tcp))
                self.assertTrue(np.allclose(orientation.as_matrix(), expected_rotation.as_matrix()))
        validate_delta_actions(rows)

    def test_slower_request_is_not_shortened_or_resampled(self):
        rows, timing = compile_targets_with_timing(self.state(), [dict(steps=20, left=dict(tcp_xyz=[.495, 0, 1]))])
        self.assertEqual(timing[0], dict(segment=1, requested_steps=20, minimum_steps=10, compiled_steps=20))
        self.assertEqual(len(rows), 20)
        self.assertTrue(np.allclose(np.asarray(rows)[:, 0], .35/20))

    def test_translation_limit_boundary_uses_existing_tolerance(self):
        for distance, count in ((.035, 1), (.035001, 2), (.07, 2), (.070001, 3)):
            with self.subTest(distance=distance):
                rows, timing = compile_targets_with_timing(self.state(), [dict(steps=1,
                    left=dict(tcp_xyz=[TCP_OFFSET_M+distance, 0, 1]))])
                self.assertEqual(len(rows), count)
                self.assertEqual(timing[0]["minimum_steps"], count)
                validate_delta_actions(rows)


if __name__ == "__main__":
    unittest.main()
