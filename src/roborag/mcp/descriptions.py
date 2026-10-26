"""Frozen model-facing text kept independent of Python source formatting."""

import json
from functools import lru_cache

from roborag.config import RESOURCE_ROOT


@lru_cache(maxsize=1)
def _descriptions():
    return json.loads((RESOURCE_ROOT / "protocol/tool-descriptions.json").read_text())


def tool_description(name: str) -> str:
    return _descriptions()[name]
