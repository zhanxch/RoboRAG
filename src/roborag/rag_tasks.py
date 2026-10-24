"""Task instructions stay separate from the robot protocol and process runner."""
from pathlib import Path

TASKS = {"stack_blocks", "fold_clothes"}


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
        prompt = (directory / f"{task}.txt").read_text() + "\n\n" + (directory / "robot.delta.txt").read_text()
    else:
        raise ValueError(f"No {controller} protocol configured for {task}")
    if no_rag:
        prompt += "\n\nCONTROL CONDITION: Historical retrieval is disabled in this episode. Call retrieve_evidence once for the empty control record, then record your plan with evidence_refs=[]. Do not call read_action_reference or invent sources. All instructions above to expand demonstrations are inapplicable."
    return prompt
