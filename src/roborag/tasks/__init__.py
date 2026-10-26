"""Stable task API backed by validated, per-task JSON resources."""

from roborag.tasks.prompts import prompt_for, public_task_instruction
from roborag.tasks.registry import (
    GENERAL_TASKS,
    RANDOM_TASKS,
    TASK_SPECS,
    TASKS,
    load_task_registry,
    simulation_task,
    task_spec,
)
from roborag.tasks.spec import TaskSpec

__all__ = [
    "GENERAL_TASKS",
    "RANDOM_TASKS",
    "TASK_SPECS",
    "TASKS",
    "TaskSpec",
    "load_task_registry",
    "prompt_for",
    "public_task_instruction",
    "simulation_task",
    "task_spec",
]
