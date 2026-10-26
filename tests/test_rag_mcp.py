import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from mcp import Client
from support import make_test_bank

from roborag.control.homing import validate_home
from roborag.mcp.server import create_server
from roborag.mcp.session import EpisodeSession as EpisodeMemory

ROOT = Path(__file__).resolve().parents[1]


class MCPIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.bank = make_test_bank(folder.name)

    async def test_cloth_measurement_and_compressed_actions_over_real_protocol(self):
        from mcp.types import CallToolResult, TextContent

        with tempfile.TemporaryDirectory() as temp:
            memory = EpisodeMemory(
                self.bank, self.bank, temp, temp, "fold_clothes", controller="delta"
            )
            memory.state = lambda: dict(step=0, episode_over=False)
            memory.submit = AsyncMock(
                return_value=dict(result=dict(plan_ok=True, measured_at_step=0))
            )
            memory.observation = lambda *a, **kw: CallToolResult(
                content=[TextContent(type="text", text="fresh observation")]
            )
            async with Client(create_server(memory)) as client:
                measured = await client.call_tool("measure_scene", {})
                self.assertFalse(measured.is_error)  # allowed before GPT planning
                self.assertIn("measure", memory.submit.call_args.args[0])
                memory.plan = {"steps": ["test"]}
                row = [0, 0, 0, 0, 0, 0, 1, 0.01, 0, 0, 0, 0, 0, 1]
                result = await client.call_tool(
                    "execute_chunk",
                    dict(
                        subgoal="test",
                        expected_feedback="moved",
                        segments=[dict(delta=row, repeat=3)],
                    ),
                )
                self.assertFalse(result.is_error)
                self.assertEqual(memory.submit.call_args.args[0]["actions"], [row, row, row])
                count = memory.submit.call_count
                invalid = await client.call_tool(
                    "execute_chunk",
                    dict(
                        subgoal="test",
                        expected_feedback="none",
                        segments=[dict(delta=row, repeat=24), dict(delta=row, repeat=1)],
                    ),
                )
                self.assertTrue(invalid.is_error)
                self.assertEqual(memory.submit.call_count, count)
                memory.state = lambda: dict(
                    step=0,
                    episode_over=False,
                    action_state={
                        f"{side}_{name}": value
                        for side in ("left", "right")
                        for name, value in (
                            ("ee_pose", [0, 0, 1, 1, 0, 0, 0]),
                            ("ee_joint_state", [1]),
                        )
                    },
                )
                targeted = await client.call_tool(
                    "execute_chunk",
                    dict(
                        subgoal="new measured target",
                        expected_feedback="moved",
                        targets=[dict(steps=4, left=dict(tcp_xyz=[0.185, 0, 1]))],
                    ),
                )
                self.assertFalse(targeted.is_error)
                request = memory.submit.call_args.args[0]
                self.assertEqual(len(request["actions"]), 4)
                self.assertAlmostEqual(request["actions"][0][0], 0.01)
                self.assertIn("targets", request["action_encoding"])
                count = memory.submit.call_count
                memory.state = lambda: dict(step=500, episode_over=True)
                ended = await client.call_tool("measure_scene", {})
                self.assertFalse(ended.is_error)
                self.assertIn("episode_over", ended.content[0].text)
                self.assertEqual(
                    memory.submit.call_count, count
                )  # simulator no longer consumes inbox

    async def test_delta_protocol_exposes_same_grammar_as_recorded_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            memory = EpisodeMemory(
                self.bank, self.bank, temp, temp, "stack_blocks", controller="delta"
            )
            async with Client(create_server(memory)) as client:
                result = await client.list_tools()
                tool = next(t for t in result.tools if t.name == "execute_chunk")
                self.assertIn("actions", tool.input_schema["properties"])
                self.assertNotIn("waypoints", tool.input_schema["properties"])
                result = await client.call_tool(
                    "execute_chunk",
                    {
                        "subgoal": "test",
                        "actions": [],
                        "return_home": True,
                        "expected_feedback": "home",
                    },
                )
                self.assertTrue(result.is_error)

    async def test_real_protocol_retrieval_plan_and_action_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "obs").mkdir()
            (path / "obs/state.json").write_text(json.dumps({"step": 0, "episode_over": False}))
            memory = EpisodeMemory(
                self.bank, self.bank, path, path, "stack_blocks", controller="delta"
            )
            async with Client(create_server(memory)) as client:
                tools = await client.list_tools()
                self.assertEqual(
                    {t.name for t in tools.tools},
                    {
                        "observe",
                        "retrieve_evidence",
                        "read_action_reference",
                        "record_plan",
                        "execute_chunk",
                    },
                )
                result = await client.call_tool(
                    "execute_chunk",
                    {
                        "subgoal": "move",
                        "waypoints": [{"arm": "right", "gripper": 1}],
                        "expected_feedback": "open",
                    },
                )
                self.assertTrue(result.is_error)
                self.assertIn("record_plan", result.content[0].text)
                result = await client.call_tool(
                    "retrieve_evidence",
                    {
                        "goal": "堆叠三块积木",
                        "observation": "三块不同纹理的积木分开放在桌面，夹爪空闲",
                    },
                )
                self.assertFalse(result.is_error)
                packet = json.loads(result.content[0].text)
                self.assertEqual(packet["purpose"], "evidence_for_gpt_planning")
                self.assertGreater(len(packet["memories"]), 0)
                source = packet["memories"][0]["source"]
                self.assertIn("episode_trace", packet["memories"][0])
                phase = packet["memories"][0]["episode_trace"][0]["id"]
                reference = await client.call_tool(
                    "read_action_reference",
                    {"source": source + ":" + phase, "include_actions": True, "images": True},
                )
                self.assertFalse(reference.is_error)
                evidence = json.loads(reference.content[0].text)
                self.assertEqual(evidence["role"], "HISTORICAL_REFERENCE")
                self.assertEqual(len(evidence["transition"]["actions"][0]), 14)
                self.assertEqual(len([c for c in reference.content if c.type == "image"]), 2)
                result = await client.call_tool(
                    "record_plan",
                    {
                        "current_scene": "白块已被右爪拿起",
                        "adaptation": "跳过示范抓取阶段",
                        "steps": [
                            {
                                "subgoal": "对齐并释放",
                                "current_objects": "右爪白块，黑色底块",
                                "completion_observation": "松爪后白块稳定留在黑块上",
                                "evidence_refs": [source],
                            }
                        ],
                    },
                )
                self.assertFalse(result.is_error)
                plan = json.loads((path / "gpt-plan.json").read_text())
                self.assertEqual(plan["role"], "gpt_generated_current_plan")
                self.assertEqual(len(plan["steps"]), 1)
                self.assertEqual(plan["observed_at_step"], 0)
                self.assertEqual(
                    [
                        json.loads(x)["tool"]
                        for x in (path / "mcp-calls.jsonl").read_text().splitlines()
                    ],
                    ["retrieve_evidence", "read_action_reference", "record_plan"],
                )

    def test_invalid_tail_cannot_partially_execute(self):
        with self.assertRaises(ValueError):
            validate_home(
                [{"arm": "right", "gripper": 0}, {"arm": "right", "xyz": [float("nan"), 0, 0]}]
            )
        with self.assertRaises(ValueError):
            validate_home([{"arm": "right", "hold_steps": -1}])


if __name__ == "__main__":
    unittest.main()
