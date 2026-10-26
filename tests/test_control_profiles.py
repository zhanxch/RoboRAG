import importlib.util
from pathlib import Path
import sys
import unittest

from roborag.control_profiles import apply_control_profile
ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec("numpy") is not None and (ROOT / "vendor/RoboICL").is_dir()
if AVAILABLE:
    sys.path.insert(0, str(ROOT / "vendor/RoboICL"))
    from roborag.continuous_policy import expand_waypoint_actions


class ControlProfileTests(unittest.TestCase):
    def test_standard_profile_keeps_reference_horizon(self):
        config = {"predict_horizon": 10, "execute_horizon": 10}
        metadata = apply_control_profile(
            config, "standard", task="put_bottles_into_dustbin", reference_horizon=10)
        self.assertEqual(config, {"predict_horizon": 10, "execute_horizon": 10})
        self.assertEqual(metadata["action_horizon"], 10)
        self.assertFalse(metadata["plan_continuity"])

    def test_hybrid_keeps_reference_and_declares_two_action_regimes(self):
        config = {"predict_horizon": 10, "execute_horizon": 10,
                  "max_translation_m": 0.03}
        metadata = apply_control_profile(
            config, "hybrid", task="put_bottles_into_dustbin", reference_horizon=10)
        self.assertEqual(config["predict_horizon"], 32)
        self.assertEqual(config["execute_horizon"], 32)
        self.assertEqual(config["max_translation_m"], 0.30)
        self.assertEqual(config["max_rotation_rad"], 0.60)
        self.assertFalse(config["fixed_predict_horizon"])
        self.assertEqual(metadata["reference_horizon"], 10)
        self.assertEqual(metadata["action_horizon"], 32)
        self.assertTrue(metadata["plan_continuity"])
        self.assertEqual(metadata["control_modes"], ["waypoint_transit", "high_frequency_precision"])
        self.assertEqual(metadata["transit_action_horizon"], 32)
        self.assertEqual(metadata["precision_action_horizon"], [4, 8])
        self.assertEqual(metadata["execution_step_translation_limit_m"], 0.03)

    def test_legacy_profile_name_remains_supported(self):
        config = {"predict_horizon": 10, "execute_horizon": 10}
        metadata = apply_control_profile(
            config, "continuous_fast", task="put_bottles_into_dustbin", reference_horizon=10)
        self.assertEqual(metadata["control_modes"], ["waypoint_transit", "high_frequency_precision"])

    def test_hybrid_is_task_scoped(self):
        with self.assertRaises(ValueError):
            apply_control_profile(
                {}, "hybrid", task="deposit_coin", reference_horizon=5)

    def test_queued_declares_persistent_segments_without_claiming_task_planning(self):
        config = {"predict_horizon": 10, "execute_horizon": 10}
        metadata = apply_control_profile(
            config, "queued", task="put_bottles_into_dustbin", reference_horizon=10)
        self.assertEqual(metadata["dispatch_window_steps"], 32)
        self.assertEqual(metadata["queue_capacity_steps"], 64)
        self.assertTrue(metadata["motion_queue_persistent"])
        self.assertFalse(metadata["plan_continuity"])
        self.assertFalse(metadata["phase_authorization"])
        self.assertEqual(metadata["execution_mode"], "synchronous")
        self.assertNotIn("precision_transition_distance_m", metadata)
        self.assertEqual(metadata["transit_waypoint_count"], [1, 6])
        with self.assertRaises(ValueError):
            apply_control_profile({}, "queued", task="stack_blocks", reference_horizon=10)

    @unittest.skipUnless(AVAILABLE, "optional robot runtime not installed")
    def test_sparse_waypoint_is_interpolated_with_gripper_change_at_endpoint(self):
        row = [0.09, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        actions = expand_waypoint_actions(
            [row], {"left_ee_joint_state": [1], "right_ee_joint_state": [1]})
        self.assertEqual(len(actions), 3)
        self.assertAlmostEqual(sum(action[0] for action in actions), 0.09)
        self.assertTrue(all(abs(action[0]) <= 0.03 + 1e-9 for action in actions))
        self.assertEqual([action[13] for action in actions], [1, 1, 1])

    @unittest.skipUnless(AVAILABLE, "optional robot runtime not installed")
    def test_precision_rows_remain_separate_and_small(self):
        rows = [[0.005, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1] for _ in range(5)]
        actions = expand_waypoint_actions(rows, {})
        self.assertEqual(len(actions), 5)
        self.assertEqual([action[0] for action in actions], [0.005] * 5)

    @unittest.skipUnless(AVAILABLE, "optional robot runtime not installed")
    def test_sparse_waypoint_limits_are_enforced(self):
        row = [0.31, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
        with self.assertRaises(ValueError):
            expand_waypoint_actions([row], {})


if __name__ == "__main__":
    unittest.main()
