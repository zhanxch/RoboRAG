"""Cross-task prompt/configuration checks without running or mocking a robot."""
import hashlib
import json
from pathlib import Path
import unittest

from roborag.rag_tasks import GENERALIZATION_TASKS, TASKS, prompt_for, simulation_task
from roborag.sop_store import default_bank

ROOT = Path(__file__).resolve().parents[1]


class GeneralizationTaskTests(unittest.TestCase):
    def test_remaining_suite_is_supported_in_both_variants(self):
        roster = json.loads((ROOT / "experiments/robodojo-generalization-test-suite-v1.json").read_text())
        remaining = {row["task"] for row in roster["tasks"]} - {"push_T", "fold_clothes"}
        self.assertEqual(remaining, GENERALIZATION_TASKS | {"stack_blocks"})
        for task in remaining:
            self.assertIn(task, TASKS)
            for variant in ("standard", "random"):
                native = simulation_task(task, variant)
                self.assertEqual(native, task + ("_random" if variant == "random" else ""))
            prompt = prompt_for(ROOT, task, "delta")
            self.assertIn("record", prompt)
            self.assertIn("24", prompt)
            self.assertIn("execute_chunk", prompt)

    def test_new_tasks_have_goal_and_full_bank_without_cloth_directives(self):
        for task in GENERALIZATION_TASKS:
            prompt = prompt_for(ROOT, task, "delta")
            self.assertIn(task, prompt)
            self.assertIn("Completion requirement:", prompt)
            self.assertNotIn("cloth_landmarks", prompt)
            self.assertNotIn("garment", prompt)
            self.assertNotIn("cloth feedback", prompt)
            self.assertEqual(default_bank(ROOT, task), ROOT / "artifacts/generalization-v3-full" / task)
            with self.assertRaises(ValueError):
                prompt_for(ROOT, task, "tcp")

    def test_established_fold_prompt_remains_exact(self):
        contract = json.loads((ROOT / "tests/fixtures/v3-contract.json").read_text())
        self.assertEqual(hashlib.sha256(prompt_for(ROOT, "fold_clothes", "delta").encode()).hexdigest(),
                         contract["prompt_sha256"])
        for task in ("fold_clothes", "stack_blocks"):
            for variant in ("standard", "random"):
                self.assertEqual(simulation_task(task, variant),
                                 task + ("_random" if variant == "random" else ""))
        self.assertEqual(prompt_for(ROOT, "stack_blocks", "delta"),
                         (ROOT / "prompts/stack_blocks.delta.txt").read_text())

    def test_digit_count_is_grounded_in_live_scene(self):
        prompt = prompt_for(ROOT, "arrange_largest_number", "delta")
        self.assertNotIn("four numbers", prompt)
        self.assertIn("number of tiles from the current scene", prompt)


if __name__ == "__main__":
    unittest.main()
