import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from roborag.config import RuntimeConfig
from roborag.runtime.doctor import inspect_runtime


class ConfigurationTests(unittest.TestCase):
    def test_virtualenv_interpreter_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interpreter = root / "venv-python"
            interpreter.symlink_to(sys.executable)
            values = {name: "venv-python" for name in RuntimeConfig.__dataclass_fields__}
            (root / "local.runtime.json").write_text(json.dumps(values))
            self.assertEqual(RuntimeConfig.load(root).policy_python, interpreter)

    def test_relative_config_paths_resolve_against_explicit_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = {name: f"local/{name}" for name in RuntimeConfig.__dataclass_fields__}
            (root / "local.runtime.json").write_text(json.dumps(values))
            config = RuntimeConfig.load(root)
            self.assertEqual(config.sim_python, root / "local/sim_python")
            values["assets"] = ""
            (root / "local.runtime.json").write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "assets"):
                RuntimeConfig.load(root)

    def test_doctor_reports_missing_configuration_without_launching_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            checks = inspect_runtime(tmp)
            self.assertFalse(checks[0]["ok"])
            self.assertIn("local.runtime.json", checks[0]["detail"])

    def test_global_root_reaches_lazy_subcommand(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-m", "roborag", "--root", tmp, "doctor", "--json"],
                env=os.environ.copy(),
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(tmp, json.loads(result.stdout)[0]["detail"])

    def test_tasks_does_not_load_simulation_or_mcp(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from roborag.cli import main; "
                "sys.argv=['roborag','tasks']; main(); "
                "assert 'mcp' not in sys.modules; assert 'isaaclab' not in sys.modules",
            ],
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("push_T", result.stdout)
