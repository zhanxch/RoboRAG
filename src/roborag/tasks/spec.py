"""Validated task configuration, independent of experiment results and runtime."""

import math
import re
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class TaskSpec:
    name: str
    prompt_file: str = "prompt.txt"
    protocol_prompt: str | None = None
    display_name: str | None = None
    capabilities: tuple[str, ...] = ()
    random_variant: bool = False
    public_instruction: bool = True
    whole_phase_reference: bool = False
    support_robots: bool = False
    ik_joint_increment_rad: float | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name):
            raise ValueError("name must be a task identifier")
        if self.display_name is None:
            object.__setattr__(self, "display_name", self.name)
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display_name must be nonempty text")
        prompt_names = [self.prompt_file]
        if self.protocol_prompt is not None:
            prompt_names.append(self.protocol_prompt)
        for name in prompt_names:
            if not isinstance(name, str) or not re.fullmatch(
                r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\.txt", name
            ):
                raise ValueError("Prompt references must be plain .txt filenames")
        if not isinstance(self.capabilities, tuple) or not all(
            isinstance(n, str) for n in self.capabilities
        ):
            raise ValueError("capabilities must be a tuple of names")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("Duplicate capabilities")
        for name in (
            "random_variant",
            "public_instruction",
            "whole_phase_reference",
            "support_robots",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        increment = self.ik_joint_increment_rad
        if increment is not None and (
            type(increment) not in (int, float) or not math.isfinite(increment) or increment <= 0
        ):
            raise ValueError("ik_joint_increment_rad must be a finite positive number")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Task configuration must be a JSON object")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown task fields: {', '.join(sorted(unknown))}")
        arguments = dict(value)
        if "capabilities" in arguments:
            if not isinstance(arguments["capabilities"], list):
                raise ValueError("capabilities must be a JSON array")
            arguments["capabilities"] = tuple(arguments["capabilities"])
        return cls(**arguments)
