import math
import unittest

from roborag.waypoints import expand_segment


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.state = {"left_ee_joint_state": [1.], "right_ee_joint_state": [0.25]}
        self.row = [0.30, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.25]

    def test_complete_plan_longer_than_dispatch_window(self):
        result = expand_segment([self.row] * 4, self.state, mode="transit")
        self.assertEqual(len(result), 40)
        self.assertAlmostEqual(sum(row[0] for row in result), 1.2)
        self.assertTrue(all(math.hypot(*row[:3]) <= .03 + 1e-9 for row in result))

    def test_over_capacity_rejects_instead_of_returning_prefix(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            expand_segment([self.row] * 4, self.state, mode="transit", max_motor_steps=32)

    def test_invalid_suffix_after_32_steps_is_validated(self):
        for value in (float("nan"), 10 ** 400):
            bad = self.row.copy()
            bad[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                expand_segment([self.row] * 4 + [bad], self.state, mode="transit")

    def test_transit_cannot_change_gripper(self):
        bad = self.row.copy()
        bad[6] = 0
        with self.assertRaisesRegex(ValueError, "preserve"):
            expand_segment([bad], self.state, mode="transit")

    def test_precision_allows_explicit_gripper_and_small_motion(self):
        row = self.row.copy()
        row[0], row[6] = .005, 0
        self.assertEqual(expand_segment([row], self.state, mode="precision"), [row])
        row[0] = .02
        with self.assertRaises(ValueError):
            expand_segment([row], self.state, mode="precision")

    def test_unknown_opening_or_bool_never_guessed(self):
        for state in ({}, {**self.state, "left_ee_joint_state": [True]}):
            with self.subTest(state=state), self.assertRaises(ValueError):
                expand_segment([self.row], state, mode="transit")
        row = self.row.copy()
        row[1] = False
        with self.assertRaises(ValueError):
            expand_segment([row], self.state, mode="transit")

    def test_world_rotation_is_split_along_same_axis(self):
        row = self.row.copy()
        row[0], row[5] = 0., .6
        result = expand_segment([row], self.state, mode="transit")
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(sum(r[5] for r in result), .6)

    def test_row_and_mode_budgets(self):
        for rows, mode in (([], "transit"), ([self.row] * 7, "transit"), ([self.row], "guess")):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                expand_segment(rows, self.state, mode=mode)


if __name__ == "__main__":
    unittest.main()
