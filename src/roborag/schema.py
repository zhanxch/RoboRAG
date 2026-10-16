"""Minimal schema; observation strings stand in for future multimodal records."""

from dataclasses import asdict, dataclass
import math


def nonnegative_int(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


@dataclass(frozen=True)
class Experience:
    record_id: str
    episode_id: str
    start_step: int
    end_step: int
    robot: str
    action_space: str
    task: str
    stage: str
    observation_before: str
    action_chunk: tuple[tuple[float, ...], ...]
    receipt: str
    observation_after: str
    outcome: str
    source: str
    split: str

    def __post_init__(self) -> None:
        for name in (
            "record_id", "episode_id", "robot", "action_space", "task", "stage",
            "observation_before", "receipt", "observation_after",
        ):
            require_text(getattr(self, name), name)
        nonnegative_int(self.start_step, "start_step")
        nonnegative_int(self.end_step, "end_step")
        if self.end_step <= self.start_step:
            raise ValueError("prototype records require a positive-length transition")
        if self.outcome not in {"success", "failure", "unknown"}:
            raise ValueError("invalid outcome")
        if self.source not in {"demonstration", "curated", "live"}:
            raise ValueError("invalid source")
        if self.split not in {"train", "validation", "test", "online"}:
            raise ValueError("invalid split")
        # Normalize nested lists from JSON so frozen records are deeply immutable.
        try:
            actions = tuple(tuple(row) for row in self.action_chunk)
        except TypeError as exc:
            raise ValueError("action_chunk must be a numeric matrix") from exc
        if not actions or not actions[0]:
            raise ValueError("action_chunk must not be empty")
        if any(len(row) != len(actions[0]) for row in actions):
            raise ValueError("action_chunk rows must have equal dimensions")
        if any(
            type(x) not in (int, float) or not math.isfinite(x)
            for row in actions for x in row
        ):
            raise ValueError("action_chunk must contain finite numbers")
        object.__setattr__(self, "action_chunk", actions)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Query:
    episode_id: str
    step: int
    robot: str
    action_space: str
    task: str
    stage: str
    observation: str

    def __post_init__(self) -> None:
        nonnegative_int(self.step, "step")
        for name in ("episode_id", "robot", "action_space", "task", "stage", "observation"):
            require_text(getattr(self, name), name)


@dataclass(frozen=True)
class Context:
    query: Query
    snapshot_version: str
    snapshot_sha256: str
    long_term_examples: tuple[Experience, ...]
    short_term_retrieved: tuple[Experience, ...]
    latest_interaction: Experience | None

    def to_dict(self) -> dict:
        # Keep provenance, outcome and source boundaries visible to the adapter.
        return asdict(self)

