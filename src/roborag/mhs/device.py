"""RoboRAG MHS v0.1: a project-owned embodied interface carried over MCP."""
from __future__ import annotations

import math
from typing import Protocol


class DeviceError(RuntimeError):
    """A driver error; earlier commands may already have moved the device."""


class Device(Protocol):
    def describe(self) -> dict: ...
    def observe(self) -> dict: ...
    def move(self, arm: str, position: list[float]) -> dict: ...
    def gripper(self, arm: str, opening: float) -> dict: ...
    def stop(self) -> dict: ...


def position3(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError("position must contain three finite numbers in metres")
    return list(value)


def opening01(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("opening must be a finite number in [0, 1]")
    return value


def description(device_id, arms, frame, *, backend, limits):
    return {
        "interface": "roborag-mhs/0.1", "standard_owner": "RoboRAG",
        "device_id": device_id, "backend": backend, "arms": arms,
        "frame": frame, "units": "metres", "limits": limits,
        "skills": {
            "observe()": "Read fresh state and available camera images.",
            "move(arm, position)": "Move EE to absolute xyz, preserving observed orientation and gripper target.",
            "gripper(arm, opening)": "Command 0=closed, 1=open; completion does not prove grasp success.",
            "stop()": "Stop dispatching new commands in this synchronous simulation adapter.",
        },
        "execution": "Blocking calls. Failure aborts the program; already executed motion is not rolled back.",
    }
