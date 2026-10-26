"""Typed model-facing plan and action inputs."""

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    subgoal: str
    current_objects: str
    completion_observation: str
    evidence_refs: list[str] = Field(default_factory=list)


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
