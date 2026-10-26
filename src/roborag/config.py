"""Explicit runtime locations; importing core modules starts no external services."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("ROBORAG_ROOT", Path(__file__).resolve().parents[2])).resolve()
RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
DEFAULT_BANK = PROJECT_ROOT / "data/graph-bank/current"


@dataclass(frozen=True)
class RuntimeConfig:
    sim_python: Path
    policy_python: Path
    upstream: Path
    assets: Path
    sim_library: Path

    @classmethod
    def load(cls, root=PROJECT_ROOT):
        path = Path(root) / "local.runtime.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Configure {path} from local.runtime.example.json; see docs/runtime.md"
            )
        values = json.loads(path.read_text())
        resolved = {}
        for name in cls.__dataclass_fields__:
            value = values.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Runtime configuration requires a path for {name}")
            # A venv Python is often a symlink. Resolving it would silently run
            # the base interpreter and lose that environment's dependencies.
            resolved[name] = Path(os.path.abspath(Path(root) / Path(value).expanduser()))
        return cls(**resolved)

    def as_dict(self):
        """Subprocess environments and command arrays require string paths."""
        return {name: str(getattr(self, name)) for name in self.__dataclass_fields__}
