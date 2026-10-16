"""Run with the configured policy interpreter; stdlib-only installs skip this module."""

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec("numpy") is not None and (ROOT / "vendor/RoboICL").is_dir()
if AVAILABLE:
    sys.path.insert(0, str(ROOT / "vendor/RoboICL"))
    from roborag.robot_memory import select_diverse
    from roborag.vision import distance
    import numpy as np


@unittest.skipUnless(AVAILABLE, "optional robot runtime not installed")
class VisualRetrievalTests(unittest.TestCase):
    def test_quaternion_sign_does_not_change_relevance(self):
        state = {f"{arm}_ee_pose": [0, 0, 0, 1, 0, 0, 0] for arm in ("left", "right")}
        state.update({f"{arm}_ee_joint_state": [0.] for arm in ("left", "right")})
        equivalent = dict(state, left_ee_pose=[0, 0, 0, -1, 0, 0, 0])
        feature = np.array([[1., 0.]] * 3)
        self.assertAlmostEqual(distance(feature, state, feature, equivalent), 0.)

    def test_diversity_preserves_budget_and_anchors(self):
        scores = [(0., 15), (.1, 16), (.2, 17), (.3, 40), (.4, 70)]
        selected = select_diverse(scores, 4, 10, anchors=(0, 100))
        self.assertEqual(selected, [0, 15, 40, 100])
        self.assertEqual(select_diverse([], 3, 10), [])

    def test_gripper_and_pose_separate_identical_visual_candidates(self):
        state = {f"{arm}_ee_pose": [0, 0, 0, 1, 0, 0, 0] for arm in ("left", "right")}
        state.update({f"{arm}_ee_joint_state": [0.] for arm in ("left", "right")})
        other = dict(state, left_ee_joint_state=[1.], left_ee_pose=[.1, 0, 0, 1, 0, 0, 0])
        feature = np.array([[1., 0.]] * 3)
        self.assertGreater(distance(feature, state, feature, other), distance(feature, state, feature, state))
