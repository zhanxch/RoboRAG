"""Task instructions stay separate from the robot protocol and process runner."""
from pathlib import Path

GENERALIZATION_TASKS = {
    "stack_bowls", "pack_objects_into_box", "hang_mugs", "sweep_blocks",
    "pour_liquid_into_cup", "make_toast", "arrange_largest_number",
    "sort_nesting_dolls_by_size", "store_laptop_and_headphones",
}
TASKS = {"stack_blocks", "fold_clothes", *GENERALIZATION_TASKS}


def simulation_task(task, variant):
    if task not in TASKS or variant not in ("standard", "random"):
        raise ValueError("Unsupported task or scene variant")
    return task + ("_random" if variant == "random" else "")


def prompt_for(root, task, controller, *, no_rag=False):
    simulation_task(task, "standard")
    directory = Path(root) / "prompts"
    specific = directory / f"{task}.{controller}.txt"
    if specific.is_file():
        prompt = specific.read_text()
    elif controller == "delta":
        protocol = "robot.generalization.delta.txt" if task in GENERALIZATION_TASKS else "robot.delta.txt"
        prompt = (directory / f"{task}.txt").read_text() + "\n\n" + (directory / protocol).read_text()
    else:
        raise ValueError(f"No {controller} protocol configured for {task}")
    if no_rag:
        prompt += "\n\nCONTROL CONDITION: Historical retrieval is disabled in this episode. Call retrieve_evidence once for the empty control record, then record your plan with evidence_refs=[]. Do not call read_action_reference or invent sources. All instructions above to expand demonstrations are inapplicable."
    return prompt
