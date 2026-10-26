import hashlib
import unittest
from pathlib import Path

from roborag.tasks import GENERAL_TASKS, prompt_for, public_task_instruction, simulation_task

ROOT = Path(__file__).resolve().parents[1]


class TaskConfigurationTests(unittest.TestCase):
    def test_cloth_task_uses_its_own_goal_and_random_scene(self):
        self.assertEqual(simulation_task("fold_clothes", "random"), "fold_clothes_random")
        prompt = prompt_for(ROOT, "fold_clothes", "delta")
        self.assertIn("sleeves", prompt)
        self.assertIn("RETURN BOTH ARMS", prompt)
        self.assertNotIn("table z=0.74", prompt)
        self.assertNotIn("three blocks", prompt)

    def test_no_rag_explicitly_disables_demo_expansion(self):
        prompt = prompt_for(ROOT, "fold_clothes", "delta", no_rag=True)
        self.assertIn("Do not call read_action_reference", prompt)
        self.assertIn("evidence_refs=[]", prompt)

    def test_unsupported_controller_fails_before_starting_simulator(self):
        with self.assertRaises(ValueError):
            prompt_for(ROOT, "fold_clothes", "tcp")

    def test_frozen_v3_fold_prompt_is_unchanged(self):
        self.assertEqual(
            hashlib.sha256(prompt_for(ROOT, "fold_clothes", "delta").encode()).hexdigest(),
            "26aa5ab01f3a0b78546089a84fdebf04c428ded0f2bb5fbf3945f4e95df8796b",
        )

    def test_basic_task_registry_rejects_nonexistent_random_variants(self):
        for task in GENERAL_TASKS:
            self.assertEqual(simulation_task(task, "standard"), task)
            if task == "push_T":
                self.assertEqual(simulation_task(task, "random"), "push_T_random")
            else:
                with self.assertRaises(ValueError):
                    simulation_task(task, "random")
            prompt = prompt_for(ROOT, task, "delta")
            self.assertIn("task_instruction", prompt)
            self.assertNotIn("measure_scene", prompt)
            self.assertNotIn("garment", prompt)

    def test_only_native_public_language_crosses_task_boundary(self):
        self.assertEqual(
            public_task_instruction(
                {"instruction": "Pick up the red cup.", "target_label": "secret", "score": 100}
            ),
            "Pick up the red cup.",
        )
        for value in (None, "", " ", {"target": "red"}, 100):
            with self.assertRaises(ValueError):
                public_task_instruction({"instruction": value})


if __name__ == "__main__":
    unittest.main()
