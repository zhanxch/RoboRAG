import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from roborag.bank.chunks import select_reference_window


class ReferenceWindowsTests(unittest.TestCase):
    def test_grasp_window_contains_closure_instead_of_only_early_approach(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "demo.hdf5"
            with h5py.File(path, "w") as data:
                for side in ("left", "right"):
                    values = np.ones((30, 1))
                    if side == "left":
                        values[18:] = 0
                    data[f"action/{side}_ee_joint_states"] = values
                    data[f"state/{side}_ee_joint_states"] = np.ones((30, 1))
            windows = [dict(start=i, end=i + 10) for i in (0, 5, 10, 15, 20)]
            window, event = select_reference_window(path, windows, "close")
            self.assertEqual(window, dict(start=15, end=25))
            self.assertEqual(event["frame"], 18)
            self.assertEqual(event["arm"], "left")
            fallback, absent = select_reference_window(path, windows, "open")
            self.assertEqual(fallback, windows[0])
            self.assertFalse(absent["command_event_found"])


if __name__ == "__main__":
    unittest.main()
