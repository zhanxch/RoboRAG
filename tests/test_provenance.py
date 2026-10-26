"""Freeze task configuration and prompts alongside shared code in each run."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from roborag.runtime.provenance import protocol_sources


class TaskProvenanceTests(unittest.TestCase):
    def test_task_config_and_prompt_changes_are_visible_in_run_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            resources = root / "src/roborag/resources"
            resources.mkdir(parents=True)
            task = resources.parent / "tasks/sample"
            task.mkdir(parents=True)
            config = task / "task.json"
            prompt = task / "prompt.txt"
            config.write_text('{"name":"sample"}')
            prompt.write_text("Initial task instruction")
            for relative in (
                "vendor/HexaShell/tools/table_plane/tool.py",
                "vendor/HexaShell/tasks/push_T/tools/planar_contact/tool.py",
                "native/third_party/RoboDojo/src/eval_client/ik_continuity.py",
                "native/third_party/RoboDojo/src/eval_client/eval_env.py",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with (
                patch("roborag.runtime.provenance.RESOURCE_ROOT", resources),
                patch(
                    "roborag.runtime.provenance.RuntimeConfig.load",
                    return_value=SimpleNamespace(upstream=root / "native"),
                ),
            ):
                before = protocol_sources(root / "bank", root)
                config.write_text('{"name":"sample","random_variant":true}')
                prompt.write_text("Changed task instruction")
                after = protocol_sources(root / "bank", root)
            changed = {path for path in before if before[path] != after[path]}
            self.assertEqual(changed, {str(config.resolve()), str(prompt.resolve())})
            self.assertEqual(
                after[str(prompt.resolve())], hashlib.sha256(prompt.read_bytes()).hexdigest()
            )
