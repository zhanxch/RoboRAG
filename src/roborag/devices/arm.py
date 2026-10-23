"""Helpers for the example arm adapters; these are not protocol requirements."""
from __future__ import annotations

import math


class DeviceError(RuntimeError):
    """A driver error; earlier commands may already have moved the device."""


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
        "device_id": device_id, "backend": backend, "arms": arms,
        "frame": frame, "units": "metres", "limits": limits,
        "capabilities": "Discover supported operations and their input schemas through MCP tools/list.",
        "execution": "Blocking calls. Failure aborts the program; already executed motion is not rolled back.",
    }
