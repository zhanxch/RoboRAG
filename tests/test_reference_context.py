import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from roborag.bank.reference import phase_context


class ReferenceContextTests(unittest.TestCase):
    def test_whole_phase_context_uses_source_state_and_stays_inside_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.hdf5"
            with h5py.File(path, "w") as data:
                data["additional_info/frequency"] = 25.0
                for arm in ("left", "right"):
                    poses = np.tile([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0], (30, 1))
                    if arm == "left":
                        poses[:, 0] = np.arange(30) * 0.001
                    data[f"state/{arm}_ee_poses"] = poses
                    data[f"state/{arm}_ee_joint_states"] = np.ones((30, 1))
                    data[f"action/{arm}_ee_joint_states"] = np.zeros((30, 1))
            result, images = phase_context(path, dict(start=5, end=25))
            self.assertEqual([s["frame"] for s in result["sampled_states"]], [5, 14, 24])
            self.assertEqual(
                result["sampled_states"][0]["recorded_state"]["left_ee_pose"][0], 0.005
            )
            self.assertEqual(result["measured_robot_motion"]["left"]["ee_path_length_m"], 0.019)
            self.assertEqual(
                images,
                [(5, "cam_head"), (14, "cam_head"), (24, "cam_head"), (14, "cam_left_wrist")],
            )
            self.assertEqual(result["duration_s"], 0.8)
            self.assertNotIn("actions", result)
            self.assertNotIn("success", result)
            with self.assertRaises(ValueError):
                phase_context(path, dict(start=5, end=31))
            _, images = phase_context(path, dict(start=29, end=30))
            self.assertEqual(images, [(29, "cam_head")])


if __name__ == "__main__":
    unittest.main()
