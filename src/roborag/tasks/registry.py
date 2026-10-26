"""Load the packaged task catalog; adding a task does not require a Python edit."""

import json
from pathlib import Path
from types import MappingProxyType

from roborag.capabilities import capability
from roborag.config import RESOURCE_ROOT
from roborag.tasks.spec import TaskSpec

TASK_ROOT = Path(__file__).resolve().parent


def _unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON field: {name}")
        result[name] = value
    return result


def load_task_registry(directory=TASK_ROOT, prompt_directory=RESOURCE_ROOT / "prompts"):
    directory, prompt_directory = Path(directory), Path(prompt_directory)
    paths = sorted(directory.glob("*/task.json"))
    if not paths:
        raise ValueError(f"No task configurations found in {directory}")
    tasks = {}
    for path in paths:
        try:
            spec = TaskSpec.from_dict(
                json.loads(path.read_text(), object_pairs_hook=_unique_fields)
            )
            if spec.name in tasks:
                raise ValueError(f"Duplicate task: {spec.name}")
            if path.parent.name != spec.name:
                raise ValueError("Task name must match its directory")
            for name in spec.capabilities:
                capability(name)
            prompts = [(path.parent, spec.prompt_file)]
            if spec.protocol_prompt:
                prompts.append((prompt_directory, spec.protocol_prompt))
            for folder, name in prompts:
                prompt = (folder / name).resolve()
                if not prompt.is_relative_to(folder.resolve()) or not prompt.is_file():
                    raise ValueError(f"Missing or invalid prompt file: {name}")
            tasks[spec.name] = spec
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid task configuration {path}: {exc}") from exc
    return MappingProxyType(tasks)


TASK_SPECS = load_task_registry()
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
