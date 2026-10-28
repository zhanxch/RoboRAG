"""Contracts captured from 1e9ffa0 before moving any implementation.

Do not regenerate the fixture from the refactor to make a test pass.
"""

import ast
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mcp import Client
from roborag.rag_mcp import EpisodeMemory, create_server
from roborag.rag_tasks import prompt_for

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "tests/fixtures/v3-contract.json").read_text())


def expected_tools():
    """Explicit user-approved interface changes; keep the original fixture intact."""
    tools = copy.deepcopy(CONTRACT["tools"])
    retrieve = next(t for t in tools if t["name"] == "retrieve_evidence")
    retrieve["input_schema"]["properties"]["arm"]["enum"] = ["", "left", "right", "both", "左臂", "右臂", "双臂"]
    motion = next(t for t in tools if t["name"] == "execute_chunk")
    motion["description"] = motion["description"].replace("Submit 1-24 NEW", "Submit NEW").replace(
        "Segments expand EXACTLY to actions, with <=24 total rows;", "Segments expand EXACTLY to actions;").replace(
        "Total steps<=24; same delta limits. No IK/grasp/fold planner is added.",
        "No per-call step cap; same delta limits and native episode budget apply.\n            No IK/grasp/fold planner is added.")
    motion["description"] = motion["description"].replace(
        "motion during rotation. Gripper applies from the segment's first row.",
        "motion during rotation. Gripper applies from the segment's first row.\n"
        "            Too-short target segments are extended to meet both arms' per-row limits;\n"
        "            requested steps are never shortened. execution.target_timing reports requested\n"
        "            and compiled steps when extended; executed_rows reports actual execution.")
    definitions = motion["input_schema"]["$defs"]
    del definitions["DeltaSegment"]["properties"]["repeat"]["maximum"]
    del definitions["TargetSegment"]["properties"]["steps"]["maximum"]
    for name in ("ArmTarget", "TargetSegment"):
        definitions[name]["additionalProperties"] = False
    return tools


class V3ContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_tool_order_schemas_descriptions_and_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = create_server(
                EpisodeMemory(tmp, tmp, tmp, tmp, "fold_clothes", controller="delta")
            )
            async with Client(server) as client:
                tools = (await client.list_tools()).tools
            self.assertEqual(
                [t.model_dump(mode="json", exclude_none=True) for t in tools], expected_tools()
            )
        # Only the documented step cap removal and target timing changes affect the
        # assembled prompt. Task instructions and plan requirements stay v3.
        original_prompt = prompt_for(ROOT, "fold_clothes", "delta").replace(
            "Generate action rows at 25 Hz;", "Generate 1-24 rows at 25 Hz;").replace(
            "without a per-call step cap.", "with at most 24 expanded rows.").replace(
            "it does not choose targets or grasps. It only extends too-short target segments "
            "to meet per-row motion limits and reports adjusted timing.",
            "it does not choose targets, timing or grasps.")
        self.assertEqual(
            hashlib.sha256(original_prompt.encode()).hexdigest(),
            CONTRACT["prompt_sha256"],
        )
        tree = ast.parse((ROOT / "src/roborag/mcp/server.py").read_text())
        server_call = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "MCPServer"
        )
        self.assertEqual(server_call.keywords[0].value.value, CONTRACT["instructions"])

    def test_untouched_v3_definitions_have_identical_ast_including_docstrings(self):
        definitions = {}
        for path in (ROOT / "src/roborag/mcp").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    definitions[node.name] = hashlib.sha256(
                        ast.dump(node, include_attributes=False).encode()
                    ).hexdigest()
        for name, digest in CONTRACT["mcp_definitions_ast_sha256"].items():
            if name in {"retrieve_evidence", "execute_delta_chunk", "DeltaSegment", "ArmTarget", "TargetSegment"}:
                continue  # Explicit changes covered by exact schemas and behavior tests.
            with self.subTest(definition=name):
                self.assertEqual(definitions[name], digest)

    def test_retrieval_actions_geometry_transport_and_prompts_are_unchanged(self):
        for name, digest in CONTRACT["unchanged_policy_files"].items():
            if name in {"prompts/robot.delta.txt", "src/roborag/action_targets.py",
                        "src/roborag/rag_execution.py", "src/roborag/graph_retrieval.py"}:
                continue  # Targeted, documented interface fixes; not an exact v3 policy claim.
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
