"""Offline snapshots and a separately owned, disposable episode memory."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .schema import Experience, nonnegative_int, require_text


@dataclass(frozen=True)
class LongTermMemory:
    version: str
    records: tuple[Experience, ...]

    def __post_init__(self) -> None:
        require_text(self.version, "snapshot version")
        object.__setattr__(self, "records", tuple(self.records))
        ids = set()
        for record in self.records:
            if record.split != "train" or record.source not in {"demonstration", "curated"}:
                raise ValueError("long-term memory accepts only curated or demonstration training data")
            if record.record_id in ids:
                raise ValueError(f"duplicate record ID: {record.record_id}")
            ids.add(record.record_id)

    def serialize(self) -> str:
        return json.dumps(
            {"schema_version": 1, "version": self.version,
             "records": [record.to_dict() for record in self.records]},
            ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
        ) + "\n"

    @property
    def sha256(self) -> str:
        """Hash of canonical snapshot content, independent of JSON whitespace."""
        return hashlib.sha256(self.serialize().encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "LongTermMemory":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported snapshot schema")
        return cls(payload["version"], tuple(Experience(**item) for item in payload["records"]))


def build_long_term(
    records: Iterable[Experience], path: str | Path, *, version: str,
) -> LongTermMemory:
    """Build an offline artifact. Never overwrite an existing snapshot."""
    memory = LongTermMemory(version, tuple(records))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(memory.serialize())
    return memory


class EpisodeMemory:
    def __init__(self, episode_id: str) -> None:
        require_text(episode_id, "episode_id")
        self._episode_id = episode_id
        self._records: list[Experience] = []
        self._closed = False

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("episode memory is closed")

    def append(self, record: Experience) -> None:
        self._ensure_open()
        if record.episode_id != self.episode_id:
            raise ValueError("cannot write another episode into short-term memory")
        if record.source != "live" or record.split != "online":
            raise ValueError("short-term memory accepts only live online interactions")
        if any(item.record_id == record.record_id for item in self._records):
            raise ValueError(f"duplicate record ID: {record.record_id}")
        if self._records and record.start_step < self._records[-1].end_step:
            raise ValueError("interactions must be appended in non-overlapping temporal order")
        self._records.append(record)

    def records_before(self, step: int) -> tuple[Experience, ...]:
        self._ensure_open()
        nonnegative_int(step, "step")
        return tuple(record for record in self._records if record.end_step <= step)

    def close(self) -> None:
        self._records.clear()
        self._closed = True

    def __len__(self) -> int:
        return len(self._records)

