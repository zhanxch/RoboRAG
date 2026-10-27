"""Catalog growth, early configuration errors, and frozen policy compatibility."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import Client

from roborag.capabilities import capability, measurement_for
from roborag.mcp.server import create_server
from roborag.mcp.session import EpisodeSession
from roborag.tasks import TASK_SPECS, TaskSpec, load_task_registry, prompt_for, simulation_task
from roborag.tasks.cli import catalog


class RegistryValidationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.configs = self.root / "tasks"
        self.prompts = self.root / "prompts"
        self.configs.mkdir()
        self.prompts.mkdir()
        (self.prompts / "shared.txt").write_text("Shared robot protocol")

    def write(self, name="sample", **overrides):
        folder = self.configs / name
        folder.mkdir(exist_ok=True)
        (folder / "prompt.txt").write_text("Task instruction")
        value = dict(name=name, protocol_prompt="shared.txt")
        value.update(overrides)
        (folder / "task.json").write_text(json.dumps(value))

    def load(self):
        return load_task_registry(self.configs, self.prompts)

    def test_forty_two_tasks_load_from_config_without_runtime_edits(self):
        for number in range(42):
            self.write(f"task_{number:02d}", capabilities=["planar_geometry"])
        registry = self.load()
        self.assertEqual(len(registry), 42)
        self.assertEqual(registry["task_41"].capabilities, ("planar_geometry",))
        with self.assertRaises(TypeError):
            registry["injected"] = registry["task_41"]

    def test_invalid_fields_fail_before_launching_any_runtime(self):
        invalid = [
            ({"typo": True}, "Unknown task fields"),
            ({"public_instruction": "false"}, "must be a boolean"),
            ({"capabilities": ["unknown_tool"]}, "Unknown measurement capability"),
            ({"capabilities": ["cloth_geometry", "cloth_geometry"]}, "Duplicate capabilities"),
            ({"capabilities": "cloth_geometry"}, "JSON array"),
            ({"ik_joint_increment_rad": True}, "finite positive"),
            ({"ik_joint_increment_rad": float("inf")}, "finite positive"),
            ({"ik_joint_increment_rad": 0}, "finite positive"),
        ]
        for overrides, message in invalid:
            with self.subTest(overrides=overrides):
                self.write(**overrides)
                with self.assertRaisesRegex(ValueError, message):
                    self.load()

    def test_missing_and_escaping_prompts_fail_at_registration(self):
        for filename, message in [
            ("missing.txt", "Missing or invalid prompt"),
            ("../outside.txt", "plain .txt filenames"),
        ]:
            with self.subTest(filename=filename):
                self.write(prompt_file=filename)
                with self.assertRaisesRegex(ValueError, message):
                    self.load()
        (self.configs / "sample/escape.txt").symlink_to(self.root / "outside.txt")
        (self.root / "outside.txt").write_text("not a packaged prompt")
        self.write(prompt_file="escape.txt")
        with self.assertRaisesRegex(ValueError, "Missing or invalid prompt"):
            self.load()
        self.write(protocol_prompt="missing.txt")
        with self.assertRaisesRegex(ValueError, "Missing or invalid prompt"):
            self.load()

    def test_duplicate_ids_keys_and_empty_catalog_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "No task configurations"):
            self.load()
        self.write()
        duplicate = self.configs / "z_duplicate/task.json"
        duplicate.parent.mkdir()
        duplicate.write_bytes((self.configs / "sample/task.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate task"):
            self.load()
        duplicate.unlink()
        (self.configs / "sample/task.json").write_text('{"name":"sample","name":"other"}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON field"):
            self.load()

    def test_task_name_must_match_its_directory(self):
        self.write()
        (self.configs / "sample").rename(self.configs / "different")
        with self.assertRaisesRegex(ValueError, "Task name must match its directory"):
            self.load()

    def test_display_status_reads_results_without_entering_task_configuration(self):
        rows = catalog(self.root)
        self.assertTrue(
            all(row["evaluation"]["status"] == "no_selected_suite_report" for row in rows)
        )
        self.assertTrue(all("validation" not in row for row in rows))
        (self.root / "results").mkdir()
        report = dict(
            tasks=dict(
                push_T=dict(
                    source_commit="recorded-revision",
                    metrics=dict(successes=3, scored=4, planned=5),
                )
            )
        )
        (self.root / "results/current.json").write_text(json.dumps(report))
        result = next(row for row in catalog(self.root) if row["name"] == "push_T")["evaluation"]
        self.assertEqual(result["policy_commit"], "recorded-revision")
        self.assertEqual((result["successes"], result["scored"], result["planned"]), (3, 4, 5))
        self.assertEqual(TASK_SPECS["push_T"].capabilities, ("planar_geometry",))


class CapabilityTests(unittest.IsolatedAsyncioTestCase):
    def test_dispatch_checks_task_capabilities_before_backend_import(self):
        with patch("roborag.capabilities.import_module") as imported:
            with self.assertRaisesRegex(ValueError, "not enabled"):
                measurement_for({"measure": {}}, ("planar_geometry",))
            with self.assertRaisesRegex(ValueError, "only one"):
                measurement_for(
                    {"measure": {}, "planar_measure": {}}, ("cloth_geometry", "planar_geometry")
                )
            imported.assert_not_called()
        self.assertIsNone(measurement_for({"actions": []}, ()))

    def test_dispatch_preserves_measurement_arguments_and_receipt(self):
        request = dict(command="planar_check", arguments={})
        selected = measurement_for({"planar_measure": request}, ("planar_geometry",))
        with patch(
            "roborag.tasks.push_T.measurements.measure_planar",
            return_value=({"measured_at_step": 7}, 0),
        ) as measured:
            api = object()
            self.assertEqual(selected.measure(api, "/test", request), ({"measured_at_step": 7}, 0))
            measured.assert_called_once_with(api, "/test", request)
        self.assertEqual(selected, capability("planar_geometry"))

    async def test_new_task_can_compose_existing_measurements(self):
        spec = TaskSpec(
            name="composed",
            capabilities=("cloth_geometry", "planar_geometry"),
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("roborag.mcp.server.task_spec", return_value=spec),
        ):
            session = EpisodeSession(temp, temp, temp, temp, spec.name)
            async with Client(create_server(session)) as client:
                names = [tool.name for tool in (await client.list_tools()).tools]
        self.assertEqual(
            names,
            [
                "measure_planar",
                "retrieve_evidence",
                "read_action_reference",
                "observe",
                "measure_scene",
                "record_plan",
                "execute_chunk",
            ],
        )


class CatalogCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_unchanged_task_protocols_match_pre_migration_catalog(self):
        baseline = json.loads((Path(__file__).parent / "fixtures/task-catalog.json").read_text())
        self.assertTrue(set(baseline["tasks"]) <= set(TASK_SPECS))
        for name, expected in baseline["tasks"].items():
            with self.subTest(task=name), tempfile.TemporaryDirectory() as temp:
                spec = TASK_SPECS[name]
                for field in (
                    "random_variant",
                    "public_instruction",
                    "whole_phase_reference",
                    "support_robots",
                    "ik_joint_increment_rad",
                ):
                    pouring_overrides = {
                        "whole_phase_reference": True,
                        "ik_joint_increment_rad": 0.25,
                    }
                    expected_value = expected["spec"][field]
                    if name == "pour_balls_into_vase":
                        # Intentional rim-v1 changes; preserve the old catalog.
                        expected_value = pouring_overrides.get(field, expected_value)
                    self.assertEqual(getattr(spec, field), expected_value)
                for variant, native in expected["simulation_tasks"].items():
                    self.assertEqual(simulation_task(name, variant), native)
                if name == "pour_balls_into_vase":
                    # rim-v1 intentionally changes this task's prompt and tools.
                    # Keep its historical fixture intact; the new contract is
                    # checked in tests/tasks/pour_balls_into_vase/.
                    continue
                measurement = expected["spec"]["measurement"]
                self.assertEqual(
                    spec.capabilities, (measurement + "_geometry",) if measurement else ()
                )
                self.assertEqual(
                    hashlib.sha256(prompt_for(temp, name).encode()).hexdigest(),
                    expected["prompt_sha256"],
                )
                self.assertEqual(
                    hashlib.sha256(prompt_for(temp, name, no_rag=True).encode()).hexdigest(),
                    expected["no_rag_prompt_sha256"],
                )
                async with Client(
                    create_server(EpisodeSession(temp, temp, temp, temp, name))
                ) as client:
                    tools = [
                        t.model_dump(mode="json", exclude_none=True)
                        for t in (await client.list_tools()).tools
                    ]
                self.assertEqual(
                    hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest(),
                    expected["tool_contract_sha256"],
                )
