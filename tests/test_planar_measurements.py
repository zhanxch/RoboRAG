import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp import Client

from roborag.backends.planar import measure_planar
from roborag.mcp.server import create_server
from roborag.mcp.session import EpisodeSession as EpisodeMemory

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data/graph-bank/current"


class PlanarMeasurementsTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "vendor/HexaShell").is_dir(), "optional HexaShell integration")
    def test_real_rgb_registration_and_released_check_without_robot_api(self):
        import cv2
        import numpy as np

        image = np.full((480, 640, 3), 220, dtype=np.uint8)

        def shape(offset, color):
            cv2.rectangle(image, (120 + offset, 140), (240 + offset, 170), color, -1)
            cv2.rectangle(image, (165 + offset, 170), (195 + offset, 260), color, -1)

        shape(0, (0, 0, 200))
        shape(220, (100, 100, 100))
        camera = dict(
            intrinsics=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            extrinsics_world=[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]],
        )
        api = SimpleNamespace(
            step=0,
            observe=lambda: {
                "png": {"cam_head": cv2.imencode(".png", image)[1].tobytes()},
                "cameras": {"cam_head": camera},
            },
        )
        result, code = measure_planar(
            api,
            ROOT / "vendor/HexaShell",
            dict(
                command="planar_match",
                arguments=dict(
                    u=180,
                    v=200,
                    ref_u=400,
                    ref_v=200,
                    surface_z=0.75,
                    reference_z=0.75,
                    height_fit="off",
                ),
            ),
        )
        self.assertEqual(code, 0, result)
        self.assertTrue(result["contact_available"])
        self.assertAlmostEqual(result["seed_goal_xy"][0] - result["seed_xyz"][0], 0.11, delta=0.001)
        self.assertAlmostEqual(result["yaw"], 0, delta=1)
        self.assertNotIn("transfer_args", result)
        image[:] = 220
        shape(220, (0, 0, 200))
        checked, code = measure_planar(api, ROOT / "vendor/HexaShell", {"command": "planar_check"})
        self.assertEqual(code, 0, checked)
        self.assertTrue(checked["alignment_verified"])
        self.assertEqual(checked["alignment_scope"], "requested_transform")

    def test_motion_command_cannot_enter_read_only_dispatch(self):
        with self.assertRaises(ValueError):
            measure_planar(SimpleNamespace(), "/unused", {"command": "planar_transfer"})

    def test_match_and_check_use_rgb_snapshot_and_preserve_visual_goal(self):
        observed = {
            "png": {"camera": b"recorded"},
            "cameras": {"camera": {"K": 1}},
            "object_pose": "must not enter measurement",
        }
        api = SimpleNamespace(observe=lambda: observed, step=15)
        goal = dict(x=0.1, y=0.2, z=0.8, to_x=0.2, to_y=0.3, open_deg=90, yaw=30, outline="opaque")
        calls = []

        def run(snapshot, command, arguments):
            self.assertFalse(hasattr(snapshot, "arms"))
            self.assertFalse(hasattr(snapshot, "env"))
            self.assertEqual(snapshot.observe(), {k: observed[k] for k in ("png", "cameras")})
            calls.append((command, arguments))
            if command == "planar_match":
                return dict(plan_ok=True, transfer_args=goal, contact_xyz=[0.1, 0.2, 0.8]), 0
            self.assertEqual(arguments, dict(goal, arm="right"))
            correction = dict(goal, x=0.21, y=0.31, yaw=-2)
            return dict(
                plan_ok=False,
                alignment_verified=False,
                correction_args=correction,
                check_args=arguments,
                residual_contact_m=0.014,
            ), 1

        with patch("roborag.backends.planar.geometry_tool", return_value=SimpleNamespace(run=run)):
            before, code = measure_planar(api, "/unused", {"command": "planar_check"})
            self.assertEqual(before["plan_fail_reason"], "no_saved_match")
            self.assertEqual(calls, [])
            result, code = measure_planar(api, "/unused", {"command": "planar_match"})
            self.assertNotIn("transfer_args", result)
            self.assertEqual(result["measured_at_step"], 15)
            result, code = measure_planar(api, "/unused", {"command": "planar_check"})
            self.assertFalse(result["alignment_verified"])
            self.assertEqual(result["correction_geometry"]["goal_xy"], [0.2, 0.3])
            self.assertNotIn("check_args", result)
            self.assertNotIn("correction_args", result)
            self.assertEqual(api._roborag_planar_goal, goal)

    def test_unknown_plane_stays_unknown(self):
        api = SimpleNamespace(observe=lambda: {"png": {}, "cameras": {}}, step=0)
        with patch(
            "roborag.backends.planar.geometry_tool",
            return_value=SimpleNamespace(run=lambda *args: ({"table_z": None, "conf": 0}, 0)),
        ):
            result, _ = measure_planar(api, "/unused", {"command": "table_plane"})
        self.assertFalse(result["plan_ok"])
        self.assertIsNone(result["table_z"])

    def test_outline_fit_cannot_put_contact_below_measured_support(self):
        api = SimpleNamespace(
            observe=lambda: {"png": {}, "cameras": {}},
            step=7,
            _roborag_support_z=0.76,
            _roborag_planar_goal={"old": "goal"},
        )
        invalid = dict(
            plan_ok=True,
            height_fit={"surface_z": 0.74},
            contact_xyz=[0, 0, 0.74],
            transfer_args={"bad": "goal"},
        )
        with patch(
            "roborag.backends.planar.geometry_tool",
            return_value=SimpleNamespace(run=lambda *args: (invalid, 0)),
        ):
            result, code = measure_planar(api, "/unused", {"command": "planar_match"})
        self.assertEqual(code, 1)
        self.assertEqual(result["plan_fail_reason"], "surface_below_support")
        self.assertNotIn("contact_xyz", result)
        self.assertNotIn("transfer_args", result)
        self.assertEqual(api._roborag_planar_goal, {"old": "goal"})


class PlanarProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_push_gets_measurement_and_terminal_state_cannot_submit(self):
        for task in ("push_T", "fold_clothes", "deposit_coin"):
            with tempfile.TemporaryDirectory() as tmp:
                memory = EpisodeMemory(BANK, ROOT / "data", tmp, tmp, task, controller="delta")
                memory.state = lambda: dict(step=0, episode_over=False)
                memory.submit = AsyncMock(return_value={"result": {"plan_ok": True}})
                async with Client(create_server(memory)) as client:
                    names = {t.name for t in (await client.list_tools()).tools}
                    self.assertEqual("measure_planar" in names, task == "push_T")
                    if task != "push_T":
                        continue
                    invalid = await client.call_tool("measure_planar", {"command": "planar_match"})
                    self.assertTrue(invalid.is_error)
                    memory.submit.assert_not_awaited()
                    measured = await client.call_tool("measure_planar", {"command": "table_plane"})
                    self.assertFalse(measured.is_error)
                    self.assertIn("planar_measure", memory.submit.call_args.args[0])
                    memory.state = lambda: dict(step=600, episode_over=True)
                    ended = await client.call_tool("measure_planar", {"command": "table_plane"})
                    self.assertFalse(ended.is_error)
                    self.assertIn("episode_over", ended.content[0].text)
                    self.assertEqual(memory.submit.await_count, 1)


if __name__ == "__main__":
    unittest.main()
