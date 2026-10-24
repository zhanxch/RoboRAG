import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from roborag.action_targets import compile_targets, TCP_OFFSET_M


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

    def test_too_fast_targets_and_invalid_late_targets_rejected_before_dispatch(self):
        for segments in ([dict(steps=1,left=dict(tcp_xyz=[1,0,1]))],
                         [dict(steps=20,left=dict(gripper=0)),dict(steps=5,right=dict(gripper=0))],
                         [dict(steps=2,left=dict(gripper=0)),dict(steps=2,right=dict(tcp_xyz=[0,0,float('nan')]))]):
            with self.assertRaises(ValueError):
                compile_targets(self.state(),segments)


if __name__ == "__main__":
    unittest.main()
