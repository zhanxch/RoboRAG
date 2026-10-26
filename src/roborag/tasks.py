"""One registry shared by prompts, MCP tools, simulation and experiment manifests."""

from dataclasses import dataclass
from typing import Literal

from roborag.config import RESOURCE_ROOT


@dataclass(frozen=True)
class TaskSpec:
    name: str
    random_variant: bool = False
    measurement: Literal["cloth", "planar"] | None = None
    public_instruction: bool = True
    whole_phase_reference: bool = False
    support_robots: bool = False
    ik_joint_increment_rad: float | None = None
    validation: str = "integration only; no complete suite result"

    @property
    def prompt_protocol(self):
        return "robot.basic.delta.txt" if self.public_instruction else "robot.delta.txt"


_SPECS = [
    TaskSpec(
        "fold_clothes",
        random_variant=True,
        measurement="cloth",
        public_instruction=False,
        validation="v3 1e9ffa0: seed1 37/50; refactor not natively re-evaluated",
    ),
    TaskSpec(
        "push_T",
        random_variant=True,
        measurement="planar",
        whole_phase_reference=True,
        ik_joint_increment_rad=0.25,
        validation="62f538d: seed1 development 6/6 + new scenes 44/44; refactor not natively re-evaluated",
    ),
    TaskSpec(
        "stack_blocks",
        random_variant=True,
        public_instruction=False,
        validation="development pilot only",
    ),
    TaskSpec("pour_balls_into_vase"),
    TaskSpec("deposit_coin"),
    TaskSpec("put_bottles_into_dustbin"),
    TaskSpec("organize_table"),
    TaskSpec("cover_blocks"),
    TaskSpec("imitate_sorting_sequence", support_robots=True),
    TaskSpec("general_pickup"),
    TaskSpec("pick_from_conveyor_by_image"),
]
TASK_SPECS = {spec.name: spec for spec in _SPECS}
TASKS = frozenset(TASK_SPECS)
GENERAL_TASKS = frozenset(name for name, spec in TASK_SPECS.items() if spec.public_instruction)
RANDOM_TASKS = frozenset(name for name, spec in TASK_SPECS.items() if spec.random_variant)


def task_spec(task: str) -> TaskSpec:
    try:
        return TASK_SPECS[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported task: {task}") from exc


def simulation_task(task, variant):
    spec = task_spec(task)
    if variant not in ("standard", "random"):
        raise ValueError("Unsupported scene variant")
    if variant == "random" and not spec.random_variant:
        raise ValueError("This task has no random variant in the configured suite")
    return task + ("_random" if variant == "random" else "")


def public_task_instruction(observation):
    """Forward only the native public language field, never evaluator state."""
    instruction = observation.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Native public task instruction is missing or invalid")
    return instruction


def prompt_for(root, task, controller="delta", *, no_rag=False):
    """Load one packaged prompt version; root is retained for existing launchers."""
    spec = task_spec(task)
    if controller != "delta":
        raise ValueError("Only the validated delta controller is maintained")
    directory = RESOURCE_ROOT / "prompts"
    specific = directory / f"{task}.delta.txt"
    if specific.is_file():
        prompt = specific.read_text()
    else:
        prompt = (
            (directory / f"{task}.txt").read_text()
            + "\n\n"
            + (directory / spec.prompt_protocol).read_text()
        )
    if no_rag:
        prompt += "\n\nCONTROL CONDITION: Historical retrieval is disabled in this episode. Call retrieve_evidence once for the empty control record, then record your plan with evidence_refs=[]. Do not call read_action_reference or invent sources. All instructions above to expand demonstrations are inapplicable."
    return prompt
