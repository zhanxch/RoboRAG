"""Cross-process MHS discovery/control and CaP failure-to-repair integration."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from roborag.cap.brains import CodexBrain, FixtureBrain
from roborag.cap.harness import run_episode
from roborag.cap.program import Program, ProgramError
from roborag.mhs.device import DeviceError
from roborag.mhs.synthetic import SyntheticDevice


class ProgramTests(unittest.TestCase):
    def test_rejects_unsupported_source_before_motion(self):
        for tail in ("import os", "robot.__class__", "while True: pass", "open('x')"):
            with self.subTest(tail=tail):
                robot = SyntheticDevice()
                with self.assertRaises(ProgramError):
                    Program(robot, lambda _: None).execute("robot.move('left', [0, 0, 0.2])\n" + tail)
                self.assertEqual(robot.seq, 0)

    def test_local_branch_and_loop_use_fresh_observations(self):
        robot = SyntheticDevice()
        calls = []
        Program(robot, calls.append).execute('''
for i in range(3):
    if i == 1:
        continue
    obs = robot.observe()
    z = obs["arms"]["left"]["position"][2] - 0.05
    robot.move("left", [0, 0, z])
''')
        self.assertEqual(robot.seq, 2)
        self.assertAlmostEqual(robot.position[2], 0.2)
        self.assertEqual(len(calls), 4)

    def test_failed_call_aborts_tail_without_replaying_prefix(self):
        robot = SyntheticDevice()
        calls = []
        with self.assertRaises(ValueError):
            Program(robot, calls.append).execute('''
robot.move("left", [0.2, 0, 0.3])
robot.move("left", [4, 0, 0.3])
robot.gripper("left", 0)
''')
        self.assertEqual(robot.seq, 1)
        self.assertEqual(robot.opening, 1)
        self.assertEqual(calls[-1]["status"], "error")

    def test_execution_budgets_abort_program(self):
        for kwargs, code in (
                ({"max_calls": 1}, "robot.observe()\nrobot.observe()"),
                ({"max_operations": 10}, "for i in range(100):\n    pass"),
                ({}, "for i in range(101):\n    pass")):
            with self.subTest(kwargs=kwargs, code=code), self.assertRaises(ProgramError):
                Program(SyntheticDevice(), lambda _: None, **kwargs).execute(code)

    def test_model_adapter_passes_selected_model_and_program_schema(self):
        with patch("roborag.transports.codex_cli.infer", return_value=({"kind": "finish"}, {})) as infer:
            brain = CodexBrain("unused", model="gpt-6-sol", effort="xhigh")
            brain({"observation": {}})
            self.assertEqual(infer.call_args.kwargs["model"], "gpt-6-sol")
            self.assertEqual(infer.call_args.args[1]["properties"]["kind"]["enum"], ["program", "finish"])

    def test_episode_failure_stops_device_and_writes_summary(self):
        robot = SyntheticDevice()
        def failing_brain(_):
            raise RuntimeError("offline")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with self.assertRaisesRegex(RuntimeError, "offline"):
                run_episode(robot, failing_brain, "task", output=out)
            self.assertTrue(robot.stopped)
            self.assertEqual(json.loads((out / "summary.json").read_text())["status"], "failed")

    def test_round_limit_stops_without_manufacturing_success(self):
        robot = SyntheticDevice()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_episode(robot, FixtureBrain(), "task", output=Path(tmp) / "run", max_rounds=1)
        self.assertEqual(result["status"], "round_limit")
        self.assertIsNone(result["task_success"])
        self.assertTrue(robot.stopped)


@unittest.skipUnless(importlib.util.find_spec("mcp"), "install roborag[mhs] for wire tests")
class MCPIntegrationTests(unittest.TestCase):
    def connect(self, *args):
        from roborag.mhs.client import MCPDevice
        return MCPDevice(sys.executable, ["-m", "roborag.mhs.server", *args],
                         env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}, timeout=5)

    def test_discovery_and_successful_program_over_actual_stdio(self):
        with self.connect() as robot, tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(robot.tools), 5)
            self.assertEqual(robot.describe()["transport"], "mcp-stdio")
            result = run_episode(robot, FixtureBrain(), "place block", output=Path(tmp) / "run")
            state = robot.observe()
            self.assertEqual(state["objects"]["block"], state["objects"]["target"])
            self.assertFalse(state["held"])
            self.assertEqual(result["rounds"], 2)
            self.assertIsNone(result["task_success"])
        self.assertFalse(robot.thread.is_alive())

    def test_failure_feedback_reaches_next_brain_turn(self):
        seen = []
        fixture = FixtureBrain()
        def brain(context):
            seen.append(deepcopy(context))
            return fixture(context)
        with self.connect("--reject-first-move") as robot, tempfile.TemporaryDirectory() as tmp:
            result = run_episode(robot, brain, "place block", output=Path(tmp) / "run")
            self.assertEqual(result["status"], "model_finished")
            self.assertEqual(result["rounds"], 3)
            feedback = seen[1]["history"][0]
            self.assertEqual(feedback["status"], "error")
            self.assertIn("injected rejection", feedback["error"])
            self.assertEqual(seen[1]["observation"]["arms"]["left"]["position"], [0, 0, 0.3])

    def test_invalid_calls_do_not_mutate_and_stop_is_latched(self):
        with self.connect() as robot:
            original = robot.observe()
            for arm, position in (("right", [0, 0, 0.2]), ("left", [0, 0]), ("left", [True, 0, 0.2])):
                with self.assertRaises(DeviceError):
                    robot.move(arm, position)
            with self.assertRaises(DeviceError):
                robot.gripper("left", True)
            self.assertEqual(robot.observe(), original)
            robot.stop()
            with self.assertRaises(DeviceError):
                robot.move("left", [0, 0, 0.2])

    def test_sessions_have_independent_state(self):
        with self.connect() as first, self.connect() as second:
            first.move("left", [0.5, 0, 0.2])
            a, b = first.observe(), second.observe()
            self.assertNotEqual(a["session_id"], b["session_id"])
            self.assertEqual(b["state_seq"], 0)

    def test_server_startup_failure_does_not_hang(self):
        from roborag.mhs.client import MCPDevice
        with self.assertRaises(Exception):
            MCPDevice(sys.executable, ["-c", "raise SystemExit(1)"], timeout=1)

    def test_transport_timeout_closes_session_without_retry(self):
        from roborag.mhs.client import MCPDevice
        script = '''import time
from roborag.mhs.server import create_server
from roborag.mhs.synthetic import SyntheticDevice
class Slow(SyntheticDevice):
    def move(self, arm, position):
        time.sleep(60)
        return super().move(arm, position)
create_server(Slow()).run(transport="stdio")
'''
        with MCPDevice(sys.executable, ["-c", script],
                       env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}, timeout=1) as robot:
            with self.assertRaisesRegex(DeviceError, "execution state unknown"):
                robot.move("left", [0, 0, 0.2])
            with self.assertRaisesRegex(DeviceError, "unavailable"):
                robot.observe()
        self.assertFalse(robot.thread.is_alive())

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "install roborag[cap]")
    def test_cli_program_transport_and_mcp_control_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp, self.connect("--reject-first-move") as robot:
            folder = Path(tmp)
            fake = folder / "fixture-model-cli"
            fake.write_text(f"#!{sys.executable}\n" + '''import json, sys
from pathlib import Path
from roborag.cap.brains import FixtureBrain
context = json.loads(sys.stdin.read().split("\\n\\n", 1)[1])
decision = FixtureBrain()(context)
target = sys.argv[sys.argv.index("--output-last-message") + 1]
Path(target).write_text(json.dumps(decision))
print(json.dumps({"type": "turn.started"}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 10}}))
''')
            fake.chmod(0o700)
            output = folder / "run"
            brain = CodexBrain(output / "model", model="gpt-6-sol", effort="xhigh")
            with patch.dict(os.environ, {"COMPARE_CODEX_BIN": str(fake)}):
                result = run_episode(robot, brain, "place block", output=output)
            self.assertEqual(result["status"], "model_finished")
            self.assertEqual(brain.calls, 3)
            invocation = json.loads((output / "model/call-001/invocation.json").read_text())
            self.assertEqual(invocation["model"], "gpt-6-sol")
            context = json.loads((output / "model/call-002/input.json").read_text())
            self.assertIn("injected rejection", context["history"][0]["error"])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "RoboDojo driver uses numpy")
class RoboDojoAdapterTests(unittest.TestCase):
    class Env:
        num_envs = 1
        def __init__(self, fail_at=None, track=True):
            self.state = {f"{s}_ee_pose": [0, 0, 0.3, 1, 0, 0, 0] for s in ("left", "right")}
            self.state.update({f"{s}_ee_joint_state": [0.7] for s in ("left", "right")})
            self.actions = []
            self.fail_at, self.track = fail_at, track
        def get_obs(self): return {"state": deepcopy(self.state), "hidden_task_score": 100}
        def is_episode_end(self): return False
        def take_action(self, action):
            if len(self.actions) == self.fail_at:
                raise RuntimeError("injected IK rejection")
            self.actions.append(deepcopy(action))
            if self.track:
                self.state = deepcopy(action)

    def test_bounded_interpolation_preserves_other_arm_orientation_and_grip(self):
        from roborag.mhs.robodojo import RoboDojoDevice
        env = self.Env()
        driver = RoboDojoDevice(env)
        receipt = driver.move("left", [0.1, 0, 0.3])
        self.assertEqual(receipt["executed_steps"], 5)
        for index, action in enumerate(env.actions):
            self.assertAlmostEqual(action["left_ee_pose"][0], (index + 1) * 0.02)
            self.assertEqual(list(action["left_ee_pose"][3:]), [1, 0, 0, 0])
            self.assertEqual(list(action["right_ee_pose"]), [0, 0, 0.3, 1, 0, 0, 0])
            self.assertEqual(list(action["left_ee_joint_state"]), [0.7])
        self.assertNotIn("hidden_task_score", driver.observe())

    def test_partial_execution_reports_prefix(self):
        from roborag.mhs.robodojo import RoboDojoDevice
        env = self.Env(fail_at=2)
        driver = RoboDojoDevice(env)
        with self.assertRaisesRegex(DeviceError, "executed_steps=2"):
            driver.move("left", [0.1, 0, 0.3])
        self.assertEqual(len(env.actions), 2)
        self.assertEqual(driver.observe()["state_seq"], 2)

    def test_tracking_error_interrupts_remaining_motion(self):
        from roborag.mhs.robodojo import RoboDojoDevice
        env = self.Env(track=False)
        driver = RoboDojoDevice(env)
        with self.assertRaisesRegex(DeviceError, "tracking residual"):
            driver.move("left", [0.1, 0, 0.3])
        self.assertEqual(len(env.actions), 2)

    def test_gripper_receipt_does_not_claim_a_grasp(self):
        from roborag.mhs.robodojo import RoboDojoDevice
        driver = RoboDojoDevice(self.Env())
        result = driver.gripper("left", 0)
        self.assertEqual(result["executed_steps"], 12)
        self.assertIsNone(result["task_success"])
        self.assertIsNone(driver.observe()["arms"]["left"]["gripper_measured"])


if __name__ == "__main__":
    unittest.main()
