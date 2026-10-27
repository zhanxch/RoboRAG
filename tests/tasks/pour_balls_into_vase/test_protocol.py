"""An explicit upgraded contract, separate from the old migration catalog."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mcp import Client

from roborag.mcp.server import create_server
from roborag.mcp.session import EpisodeSession
from roborag.tasks import prompt_for

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


class PourProtocolContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_versioned_task_prompt_and_measurement_contract(self):
        expected = json.loads((HERE / "protocol-v3.json").read_text())
        config = json.loads((ROOT / "src/roborag/tasks/pour_balls_into_vase/task.json").read_text())
        self.assertEqual(config, expected["task_config"])
        self.assertEqual(config["capabilities"], ["rim_geometry"])
        self.assertTrue(config["whole_phase_reference"])
        self.assertEqual(config["ik_joint_increment_rad"], 0.25)
        with tempfile.TemporaryDirectory() as temp:
            for no_rag, key in ((False, "prompt_sha256"), (True, "no_rag_prompt_sha256")):
                self.assertEqual(
                    hashlib.sha256(
                        prompt_for(temp, config["name"], no_rag=no_rag).encode()
                    ).hexdigest(),
                    expected[key],
                )
            async with Client(
                create_server(EpisodeSession(temp, temp, temp, temp, config["name"]))
            ) as client:
                tools = [
                    tool.model_dump(mode="json", exclude_none=True)
                    for tool in (await client.list_tools()).tools
                ]
            self.assertEqual([tool["name"] for tool in tools], expected["tool_names"])
            self.assertEqual(
                hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest(),
                expected["tool_contract_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
