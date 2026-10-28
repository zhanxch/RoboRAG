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
        # Execution prompt is restored exactly; SOP adaptation is applied by the runner.
        original_prompt = prompt_for(ROOT, "fold_clothes", "delta")
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
                    if node.name in {"EpisodeMemory", "__init__"}:
                        node = copy.deepcopy(node)
                        for call in ast.walk(node):
                            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "open_evidence_bank":
                                call.func.id = "BankStore"  # Only the evidence store factory is substituted.
                    definitions[node.name] = hashlib.sha256(
                        ast.dump(node, include_attributes=False).encode()
                    ).hexdigest()
        for name, digest in CONTRACT["mcp_definitions_ast_sha256"].items():
            if name in {"retrieve_evidence", "read_action_reference"}:
                continue  # Explicit changes covered by exact schemas and behavior tests.
            with self.subTest(definition=name):
                self.assertEqual(definitions[name], digest)

    def test_retrieval_actions_geometry_transport_and_prompts_are_unchanged(self):
        for name, digest in CONTRACT["unchanged_policy_files"].items():
            if name in {"src/roborag/graph_retrieval.py"}:
                continue  # Targeted, documented interface fixes; not an exact v3 policy claim.
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
