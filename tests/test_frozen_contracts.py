"""Golden interfaces captured directly from the evaluated policy checkouts.

Do not regenerate these from the refactored implementation to make a test pass.
Changing a model-facing contract requires a new policy version and evaluation.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mcp import Client

from roborag.control.targets import compile_targets
from roborag.mcp.server import create_server
from roborag.mcp.session import EpisodeSession
from roborag.tasks import prompt_for


class FrozenContracts(unittest.IsolatedAsyncioTestCase):
    async def test_evaluated_task_contracts(self):
        fixtures = Path(__file__).parent / "fixtures"
        for path in sorted(fixtures.glob("*-contract.json")):
            baseline = json.loads(path.read_text())
            with self.subTest(task=baseline["task"]), tempfile.TemporaryDirectory() as tmp:
                session = EpisodeSession(tmp, tmp, tmp, tmp, baseline["task"])
                async with Client(create_server(session)) as client:
                    tools = (await client.list_tools()).tools
                self.assertEqual(
                    [tool.model_dump(mode="json", exclude_none=True) for tool in tools],
                    baseline["tools"],
                )
                self.assertEqual(
                    hashlib.sha256(prompt_for(tmp, baseline["task"]).encode()).hexdigest(),
                    baseline["prompt_sha256"],
                )
                for example in baseline["target_examples"]:
                    np.testing.assert_array_equal(
                        compile_targets(example["state"], example["segments"]), example["actions"]
                    )
