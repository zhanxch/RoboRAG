"""Reuse the baseline's RGB geometry without its folding motion programs."""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path


@lru_cache(maxsize=2)
def cloth_tool(hexashell):
    path = Path(hexashell) / "tasks/fold_clothes/tools/cloth_view/tool.py"
    spec = importlib.util.spec_from_file_location("roborag_cloth_rgb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure_scene(api, hexashell, request):
    """Return current measured points, excluding upstream's heuristic fold plan."""
    command = request["command"]
    if command not in ("cloth_landmarks", "table_xy"):
        raise ValueError("Only cloth_landmarks and table_xy are measurement commands")
    result, code = cloth_tool(str(Path(hexashell).resolve())).run(api, command, request.get("arguments", {}))
    result = dict(result)
    result.pop("fold_construction", None)
    # Upstream recovery strings mention its CLI/motion programs. They are not
    # part of this interface; the model authors any camera or garment motion.
    result.pop("retry_options", None)
    result.update(role="CURRENT_RGB_MEASUREMENT", measured_at_step=api.step,
                  source="HexaShell cloth_view RGB geometry; fold_construction omitted")
    return result, code
