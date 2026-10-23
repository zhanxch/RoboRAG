"""Standard MCP discovery/control and CaP failure-to-repair integration."""
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
from roborag.devices.arm import DeviceError
from roborag.devices.synthetic import SyntheticDevice


class LocalMCP:
    catalog = {"tools": [], "resources": []}
    def __init__(self):
        self.device = SyntheticDevice()
    def call_tool(self, name, arguments=None):
        return {"structuredContent": getattr(self.device, name)(**(arguments or {})), "content": []}


def fixture_episode(mcp, brain, task, **kwargs):
    return run_episode(mcp, brain, task, before_round=[{"tool": "observe"}],
                       on_exit=[{"tool": "stop"}], **kwargs)


class ProgramTests(unittest.TestCase):
    def test_rejects_unsupported_source_before_motion(self):
        for tail in ("import os", "mcp.__class__", "while True: pass", "open('x')"):
            with self.subTest(tail=tail):
                mcp = LocalMCP()
                robot = mcp.device
                with self.assertRaises(ProgramError):
                    Program(mcp, lambda _: None).execute('mcp.call_tool("move", {"arm": "left", "position": [0, 0, 0.2]})\n'  + tail)
                self.assertEqual(robot.seq, 0)

    def test_local_branch_and_loop_use_fresh_observations(self):
        mcp = LocalMCP()
        robot = mcp.device
        calls = []
        Program(mcp, calls.append).execute('''
for i in range(3):
    if i == 1:
        continue
    obs = mcp.call_tool("observe")["structuredContent"]
    z = obs["arms"]["left"]["position"][2] - 0.05
    mcp.call_tool("move", {"arm": "left", "position": [0, 0, z]})
''')
        self.assertEqual(robot.seq, 2)
        self.assertAlmostEqual(robot.position[2], 0.2)
        self.assertEqual(len(calls), 4)

    def test_failed_call_aborts_tail_without_replaying_prefix(self):
        mcp = LocalMCP()
        robot = mcp.device
        calls = []
        with self.assertRaises(ValueError):
            Program(mcp, calls.append).execute('''
mcp.call_tool("move", {"arm": "left", "position": [0.2, 0, 0.3]})
mcp.call_tool("move", {"arm": "left", "position": [4, 0, 0.3]})
mcp.call_tool("gripper", {"arm": "left", "opening": 0})
''')
        self.assertEqual(robot.seq, 1)
        self.assertEqual(robot.opening, 1)
        self.assertEqual(calls[-1]["status"], "error")

    def test_execution_budgets_abort_program(self):
        for kwargs, code in (
                ({"max_calls": 1}, "mcp.call_tool('observe')\nmcp.call_tool('observe')"),
                ({"max_operations": 10}, "for i in range(100):\n    pass"),
                ({}, "for i in range(101):\n    pass")):
            with self.subTest(kwargs=kwargs, code=code), self.assertRaises(ProgramError):
                Program(LocalMCP(), lambda _: None, **kwargs).execute(code)

    def test_model_adapter_passes_selected_model_and_program_schema(self):
        with patch("roborag.transports.codex_cli.infer", return_value=({"kind": "finish"}, {})) as infer:
            brain = CodexBrain("unused", model="gpt-6-sol", effort="xhigh")
            brain({"observation": {}})
            self.assertEqual(infer.call_args.kwargs["model"], "gpt-6-sol")
            self.assertEqual(infer.call_args.args[1]["properties"]["kind"]["enum"], ["program", "finish"])

    def test_standard_mcp_images_become_model_attachments(self):
        from roborag.cap.brains import model_content
        context = {"result": {"content": [{"type": "image", "mimeType": "image/jpeg", "data": "abc"}]}}
        self.assertEqual(model_content(context)["result"]["content"][0],
                         {"type": "input_image", "image_url": "data:image/jpeg;base64,abc"})
        self.assertEqual(context["result"]["content"][0]["type"], "image")

    def test_episode_failure_stops_device_and_writes_summary(self):
        mcp = LocalMCP()
        robot = mcp.device
        def failing_brain(_):
            raise RuntimeError("offline")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with self.assertRaisesRegex(RuntimeError, "offline"):
                fixture_episode(mcp, failing_brain, "task", output=out)
            self.assertTrue(robot.stopped)
            self.assertEqual(json.loads((out / "summary.json").read_text())["status"], "failed")

    def test_round_limit_stops_without_manufacturing_success(self):
        mcp = LocalMCP()
        robot = mcp.device
        with tempfile.TemporaryDirectory() as tmp:
            result = fixture_episode(mcp, FixtureBrain(), "task", output=Path(tmp) / "run", max_rounds=1)
        self.assertEqual(result["status"], "round_limit")
        self.assertIsNone(result["task_success"])
        self.assertTrue(robot.stopped)


@unittest.skipUnless(importlib.util.find_spec("mcp"), "install roborag[cap] for wire tests")
class MCPIntegrationTests(unittest.TestCase):
    def connect(self, *args):
        from roborag.cap.mcp_client import MCPClient
        return MCPClient(sys.executable, ["-m", "roborag.devices.synthetic_server", *args],
                         env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}, timeout=5)

    def test_discovery_and_successful_program_over_actual_stdio(self):
        with self.connect() as robot, tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(set(robot.tools), {"observe", "move", "gripper", "stop"})
            self.assertIn("capabilities", robot.catalog["initialization"])
            self.assertIn("contents", robot.read_resource("fixture://arm/manual"))
            result = fixture_episode(robot, FixtureBrain(), "place block", output=Path(tmp) / "run")
            state = robot.call_tool("observe")["structuredContent"]
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
            result = fixture_episode(robot, brain, "place block", output=Path(tmp) / "run")
            self.assertEqual(result["status"], "model_finished")
            self.assertEqual(result["rounds"], 3)
            feedback = seen[1]["history"][0]
            self.assertEqual(feedback["status"], "error")
            self.assertIn("injected rejection", feedback["error"])
            self.assertEqual(seen[1]["inputs"][0]["result"]["structuredContent"]["arms"]["left"]["position"], [0, 0, 0.3])

    def test_invalid_calls_do_not_mutate_and_stop_is_latched(self):
        with self.connect() as robot:
            original = robot.call_tool("observe")["structuredContent"]
            for arm, position in (("right", [0, 0, 0.2]), ("left", [0, 0]), ("left", [True, 0, 0.2])):
                with self.assertRaises(Exception):
                    robot.call_tool("move", {"arm": arm, "position": position})
            with self.assertRaises(Exception):
                robot.call_tool("gripper", {"arm": "left", "opening": True})
            self.assertEqual(robot.call_tool("observe")["structuredContent"], original)
            robot.call_tool("stop")
            with self.assertRaises(Exception):
                robot.call_tool("move", {"arm": "left", "position": [0, 0, 0.2]})

    def test_sessions_have_independent_state(self):
        with self.connect() as first, self.connect() as second:
            first.call_tool("move", {"arm": "left", "position": [0.5, 0, 0.2]})
            a, b = first.call_tool("observe")["structuredContent"], second.call_tool("observe")["structuredContent"]
            self.assertNotEqual(a["session_id"], b["session_id"])
            self.assertEqual(b["state_seq"], 0)

    def test_server_startup_failure_does_not_hang(self):
        from roborag.cap.mcp_client import MCPClient
        with self.assertRaises(Exception):
            MCPClient(sys.executable, ["-c", "raise SystemExit(1)"], timeout=1)

    def test_unrelated_tools_text_results_resources_and_schema(self):
        from jsonschema import ValidationError
        from roborag.cap.mcp_client import MCPClient
        script = '''from mcp.server.fastmcp import FastMCP
server = FastMCP("wheeled-fixture", log_level="WARNING")
@server.tool(structured_output=False)
def navigate_to(waypoint: str) -> str:
    return "arrived at " + waypoint
@server.resource("navigation://map")
def map_reference() -> str:
    return "dock, charging"
server.run(transport="stdio")
'''
        with MCPClient(sys.executable, ["-c", script], timeout=5) as mcp, tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(set(mcp.tools), {"navigate_to"})
            with self.assertRaises(ValidationError):
                mcp.call_tool("navigate_to", {"waypoint": 3})
            seen = []
            def brain(context):
                seen.append(context)
                if context["history"]:
                    return {"kind": "finish", "code": "", "note": "finished fixture"}
                return {"kind": "program", "note": "use discovered tool", "code": '''
reference = mcp.read_resource("navigation://map")
result = mcp.call_tool("navigate_to", {"waypoint": "dock"})
'''}
            result = run_episode(mcp, brain, "reach dock", output=Path(tmp) / "run")
            self.assertEqual(result["status"], "model_finished")
            reply = seen[1]["history"][0]["calls"][-1]["result"]
            self.assertEqual(reply["content"][0]["text"], "arrived at dock")
            self.assertNotIn("structuredContent", reply)
            self.assertEqual(seen[0]["inputs"], [])

    def test_paginated_discovery_and_repeated_cursor_failure(self):
        import asyncio
        from mcp.types import ListToolsResult, Tool
        from roborag.cap.mcp_client import MCPError, list_all
        calls = []
        async def page(*, params):
            calls.append(params.cursor)
            return ListToolsResult(tools=[Tool(name="a" if params.cursor is None else "b", inputSchema={"type": "object"})],
                                   nextCursor="page2" if params.cursor is None else None)
        self.assertEqual([x["name"] for x in asyncio.run(list_all(page, "tools"))], ["a", "b"])
        self.assertEqual(calls, [None, "page2"])
        async def broken(*, params):
            return ListToolsResult(tools=[], nextCursor="same")
        with self.assertRaisesRegex(MCPError, "repeated"):
            asyncio.run(list_all(broken, "tools"))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "native driver uses numpy")
    def test_robodojo_server_exposes_native_adapter_through_standard_mcp(self):
        from roborag.cap.mcp_client import MCPClient, ToolError
        script = '''import anyio
from mcp.server.stdio import stdio_server
from roborag.devices.robodojo_server import create_server
from roborag.devices.robodojo import RoboDojoDevice
from test_cap_mcp import RoboDojoAdapterTests
device = RoboDojoDevice(RoboDojoAdapterTests.Env(), joint_limits={s: [[-1, 1]] * 6 for s in ("left", "right")})
device.camera_observe = lambda cameras=None: [{"type": "image", "mimeType": "image/jpeg", "data": "YQ=="}]
server = create_server(device)
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
anyio.run(main)
'''
        root = Path(__file__).resolve().parents[1]
        with MCPClient(sys.executable, ["-c", script], timeout=5,
                       env={"PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")])}) as mcp:
            self.assertEqual(set(mcp.tools), {"robot_state", "camera_observe", "move_ee", "move_joints",
                                               "set_gripper", "execution_status", "stop_motion"})
            self.assertEqual(len(mcp.catalog["resources"]), 3)
            self.assertEqual(mcp.call_tool("camera_observe")["content"][0]["type"], "image")
            result = mcp.call_tool("move_ee", {"arm": "left", "position": [0.1, 0, 0.3]})
            self.assertEqual(result["structuredContent"]["executed_steps"], 5)
            mcp.call_tool("move_joints", {"arm": "right", "positions": [0.1] * 6})
            state = mcp.call_tool("robot_state")["structuredContent"]
            self.assertEqual(state["joints"]["right"], [0.1] * 6)
            mcp.call_tool("stop_motion")
            with self.assertRaises(ToolError) as failed:
                mcp.call_tool("set_gripper", {"arm": "left", "opening": 0})
            self.assertTrue(failed.exception.result["isError"])

    def test_transport_timeout_closes_session_without_retry(self):
        from roborag.cap.mcp_client import MCPClient
        script = '''import time
from roborag.devices.synthetic_server import create_server
from roborag.devices.synthetic import SyntheticDevice
class Slow(SyntheticDevice):
    def move(self, arm, position):
        time.sleep(60)
        return super().move(arm, position)
create_server(Slow()).run(transport="stdio")
'''
        with MCPClient(sys.executable, ["-c", script],
                       env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}, timeout=1) as robot:
            with self.assertRaisesRegex(RuntimeError, "execution state unknown"):
                robot.call_tool("move", {"arm": "left", "position": [0, 0, 0.2]})
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                robot.call_tool("observe")["structuredContent"]
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
                result = fixture_episode(robot, brain, "place block", output=output)
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
            self.state.update({f"{s}_arm_joint_state": [0.] * 6 for s in ("left", "right")})
            self.actions = []
            self.fail_at, self.track = fail_at, track
        def get_obs(self): return {"state": deepcopy(self.state), "hidden_task_score": 100}
        def is_episode_end(self): return False
        def take_action(self, action):
            if len(self.actions) == self.fail_at:
                raise RuntimeError("injected IK rejection")
            self.actions.append(deepcopy(action))
            if self.track:
                self.state.update(deepcopy(action))

    def test_bounded_interpolation_preserves_other_arm_orientation_and_grip(self):
        from roborag.devices.robodojo import RoboDojoDevice
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
        from roborag.devices.robodojo import RoboDojoDevice
        env = self.Env(fail_at=2)
        driver = RoboDojoDevice(env)
        with self.assertRaisesRegex(DeviceError, "executed_steps=2"):
            driver.move("left", [0.1, 0, 0.3])
        self.assertEqual(len(env.actions), 2)
        self.assertEqual(driver.observe()["state_seq"], 2)

    def test_tracking_error_interrupts_remaining_motion(self):
        from roborag.devices.robodojo import RoboDojoDevice
        env = self.Env(track=False)
        driver = RoboDojoDevice(env)
        with self.assertRaisesRegex(DeviceError, "tracking residual"):
            driver.move("left", [0.1, 0, 0.3])
        self.assertEqual(len(env.actions), 2)

    def test_gripper_receipt_does_not_claim_a_grasp(self):
        from roborag.devices.robodojo import RoboDojoDevice
        driver = RoboDojoDevice(self.Env())
        result = driver.gripper("left", 0)
        self.assertEqual(result["executed_steps"], 12)
        self.assertIsNone(result["task_success"])
        self.assertIsNone(driver.observe()["arms"]["left"]["gripper_measured"])

    def test_rotation_and_joint_limits(self):
        import math
        from roborag.devices.robodojo import RoboDojoDevice
        env = self.Env()
        driver = RoboDojoDevice(env, joint_limits={s: [[-1, 1]] * 6 for s in ("left", "right")})
        target = [math.cos(0.15), 0, 0, math.sin(0.15)]
        driver.move("left", [0, 0, 0.3], target)
        for a, b in zip(driver.observe()["arms"]["left"]["quaternion_wxyz"], target):
            self.assertAlmostEqual(a, b)
        driver.move_joints("left", [0.11] * 6)
        self.assertEqual(driver.execution_status()["last_receipt"]["executed_steps"], 3)
        before = len(env.actions)
        with self.assertRaisesRegex(ValueError, "limits"):
            driver.move_joints("left", [1.1] * 6)
        self.assertEqual(len(env.actions), before)


class WorkerIsolationTests(unittest.TestCase):
    def test_worker_error_can_be_corrected_and_eof_reaps_process(self):
        from roborag.devices.worker_proxy import WorkerProxy
        script = '''import json, sys
print(json.dumps({"ok": True, "value": {"joint_limits": {}}}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request["method"] == "move":
        result = {"ok": False, "error": "rejected target"}
    else:
        result = {"ok": True, "value": {"state_seq": 0}}
    print(json.dumps(result), flush=True)
'''
        proxy = WorkerProxy({"command": [sys.executable, "-c", script], "timeout": 2})
        try:
            with self.assertRaisesRegex(DeviceError, "rejected"):
                proxy.move("left", [0, 0, 0])
            self.assertEqual(proxy.observe(), {"state_seq": 0})
        finally:
            proxy.close()
        self.assertEqual(proxy.process.returncode, 0)

    def test_worker_crash_invalidates_proxy_without_replay(self):
        from roborag.devices.worker_proxy import WorkerProxy
        script = '''import json, sys
print(json.dumps({"ok": True, "value": {"joint_limits": {}}}), flush=True)
sys.stdin.readline()
raise SystemExit(1)
'''
        proxy = WorkerProxy({"command": [sys.executable, "-c", script], "timeout": 2})
        with self.assertRaisesRegex(EOFError, "unknown"):
            proxy.observe()
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            proxy.observe()
        self.assertIsNotNone(proxy.process.poll())


if __name__ == "__main__":
    unittest.main()
