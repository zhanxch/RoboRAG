"""Small model-facing contracts for the staged-v1 episode protocol."""

from typing import Literal

from pydantic import BaseModel, Field


class ExperienceUse(BaseModel):
    source: str = Field(min_length=1)
    adopted: str = Field(min_length=1)
    adaptation: str = Field(min_length=1)


class GraphPhase(BaseModel):
    id: str = Field(min_length=1)
    sop_stage: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    object_bindings: dict[str, str]
    depends_on: list[str] = Field(default_factory=list)
    expected_effects: list[str] = Field(min_length=1)
    experience: list[ExperienceUse] = Field(default_factory=list)


class EffectCheck(BaseModel):
    effect: str = Field(min_length=1)
    status: Literal["met", "unmet", "unknown"]
    evidence_ids: list[str] = Field(default_factory=list)
    observation: str = Field(min_length=1)


class PlanInvalidation(BaseModel):
    kind: Literal[
        "object_unavailable", "goal_changed", "dependency_invalidated", "sop_inapplicable"
    ]
    description: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
