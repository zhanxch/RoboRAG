from pathlib import Path
import unittest
from roborag.rag_tasks import prompt_for, simulation_task

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


if __name__ == "__main__":
    unittest.main()
