"""Run with the configured policy interpreter; stdlib-only installs skip this module."""

import importlib.util
from pathlib import Path
import sys
import unittest
import json
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec("numpy") is not None and (ROOT / "vendor/RoboICL").is_dir()
if AVAILABLE:
    sys.path.insert(0, str(ROOT / "vendor/RoboICL"))
    from roborag.robot_memory import select_diverse, RetrievalModel, Model, render_selected_history, align_phase
    from roborag.vision import distance, scene_distance
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

    def test_initial_scene_preserves_left_right_position(self):
        scene = np.eye(4)
        mirrored = scene[[1, 0, 3, 2]]
        self.assertAlmostEqual(scene_distance(scene, scene), 0.)
        self.assertGreater(scene_distance(scene, mirrored), .9)

    def test_source_selection_never_mixes_equal_step_ids(self):
        model = object.__new__(RetrievalModel)
        model.multi_episode = True
        model._selected_source = None
        scene = np.eye(4)
        model.encoder = SimpleNamespace(scene=lambda _: scene)
        model.archive = {0: {"images": {"cam_head": "observed RGB"}}}
        model.episodes = {"a": {"scene": scene[[1, 0, 3, 2]]}, "b": {"scene": scene}}
        model.corpus = {"records": [{"source_sha256": sha, "start": 0, "record_id": sha}
                                    for sha in ("a", "b")]}
        model._select_source()
        self.assertEqual(model._selected_source, "b")
        self.assertEqual(model.long_records[0]["record_id"], "b")
        model.encoder.scene = lambda _: scene[[1, 0, 3, 2]]
        model._select_source()
        self.assertEqual(model._selected_source, "b")  # frozen within episode
        with patch.object(Model, "reset"):
            model.reset()
        self.assertIsNone(model._selected_source)
        self.assertEqual(model._feature_cache, {})
        model._select_source()
        self.assertEqual(model._selected_source, "a")

    def test_compressed_live_history_keeps_failure_intent_without_future(self):
        records = [{"start": n, "end": n+5, "items": [{"action": n}],
                    "summary": {"execution_note": "missed grasp" if n == 5 else f"intent {n}"}}
                   for n in (0, 5, 10, 15)]
        output = render_selected_history(records, {0, 10, 15}, 15, lambda n: {"observation": n})
        encoded = json.dumps(output)
        self.assertIn("missed grasp", encoded)
        self.assertNotIn('"action": 15', encoded)
        self.assertNotIn("intent 15", encoded)
        self.assertEqual(output[-1], {"observation": 15})
        output[1]["action"] = "changed"
        self.assertEqual(records[0]["items"][0]["action"], 0)

    def test_phase_alignment_disambiguates_initial_and_final_rest(self):
        scores = [(0.2, 0), (.6, 10), (.8, 20), (.1, 190)]
        cursor, ranked = align_phase(scores, 0, 5, initial=True)
        self.assertEqual(cursor, 0)
        self.assertEqual(min(ranked)[1], 0)
        cursor, _ = align_phase([(2., 0), (.4, 10), (.2, 20), (0., 190)], cursor, 5)
        self.assertEqual(cursor, 20)
        cursor, _ = align_phase([(0., 0), (.2, 10), (.4, 20), (0., 190)], cursor, 5)
        self.assertEqual(cursor, 10)  # limited recovery backtrack, no end jump
