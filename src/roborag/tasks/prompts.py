"""Assemble explicit prompt fragments without changing their frozen contents."""

from roborag.config import RESOURCE_ROOT
from roborag.tasks.registry import TASK_ROOT, task_spec


def public_task_instruction(observation):
    """Forward only the native public language field, never evaluator state."""
    instruction = observation.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Native public task instruction is missing or invalid")
    return instruction


def prompt_for(root, task, controller="delta", *, no_rag=False):
    """Load packaged prompts; root is retained for existing launchers."""
    spec = task_spec(task)
    if controller != "delta":
        raise ValueError("Only the validated delta controller is maintained")
    prompt = (TASK_ROOT / spec.name / spec.prompt_file).read_text()
    if spec.protocol_prompt:
        prompt += "\n\n" + (RESOURCE_ROOT / "prompts" / spec.protocol_prompt).read_text()
    if no_rag:
        prompt += "\n\nCONTROL CONDITION: Historical retrieval is disabled in this episode. Call retrieve_evidence once for the empty control record, then record your plan with evidence_refs=[]. Do not call read_action_reference or invent sources. All instructions above to expand demonstrations are inapplicable."
    return prompt
