"""Unchanged v3 plan and action inputs."""

from typing import Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    subgoal: str
    current_objects: str
    completion_observation: str
    evidence_refs: list[str] = Field(default_factory=list)


class Waypoint(BaseModel):
    arm: Literal["left", "right"]
    home: bool = False
    xyz: list[float] | None = None
    point: Literal["down", "forward", "down45"] | None = None
    open_axis: Literal["x", "y", "z"] = "x"
    gripper: float | None = None
    hold_steps: int = 0


class DeltaSegment(BaseModel):
    delta: list[float]
    repeat: int = Field(ge=1, le=24, strict=True)


class ArmTarget(BaseModel):
    tcp_xyz: list[float] | None = None
    quaternion_wxyz: list[float] | None = None
    gripper: float | None = None


class TargetSegment(BaseModel):
    steps: int = Field(ge=1, le=24, strict=True)
    left: ArmTarget | None = None
    right: ArmTarget | None = None
