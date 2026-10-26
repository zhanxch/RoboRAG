"""Read-only calibrated planar geometry; no task state or motion programs."""

import importlib.util
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


@lru_cache(maxsize=2)
def geometry_tool(hexashell, name):
    relative = {
        "planar": "tasks/push_T/tools/planar_contact/tool.py",
        "table": "tools/table_plane/tool.py",
    }[name]
    spec = importlib.util.spec_from_file_location(
        "roborag_planar_" + name, Path(hexashell) / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure_planar(api, hexashell, request):
    command, arguments = request["command"], request.get("arguments", {})
    if command not in ("table_plane", "planar_height", "planar_match", "planar_check"):
        raise ValueError("Only observation commands are allowed")
    # The imported measurement code receives only a snapshot, never the robot API.
    observed = api.observe()
    observation = {key: observed[key] for key in ("png", "cameras")}
    snapshot = SimpleNamespace(observe=lambda: observation)
    if command == "table_plane":
        table = geometry_tool(str(Path(hexashell).resolve()), "table")
        result, code = table.run(snapshot, command, arguments)
        result = dict(result, plan_ok=result.get("table_z") is not None)
        if result["plan_ok"]:
            api._roborag_support_z = result["table_z"]
    else:
        tool = geometry_tool(str(Path(hexashell).resolve()), "planar")
        if command == "planar_check":
            saved = getattr(api, "_roborag_planar_goal", None)
            if saved is None:
                return dict(
                    plan_ok=False, plan_fail_reason="no_saved_match", measured_at_step=api.step
                ), 1
            arguments = dict(saved, arm="right")  # Used only to label geometry; no arm access.
        result, code = tool.run(snapshot, command, arguments)
        result = dict(result)
        support = getattr(api, "_roborag_support_z", None)
        if command == "planar_match" and result.get("plan_ok") and support is not None:
            height = result["height_fit"]["surface_z"]
            if not math.isfinite(height) or height < support - 0.003:
                # Outline size differences can masquerade as a height change.
                # Do not save an impossible contact or replace the prior visual goal.
                result = dict(
                    plan_ok=False,
                    plan_fail_reason="surface_below_support",
                    fitted_surface_z=height,
                    measured_support_z=support,
                    plan_detail="Measure the source height from calibrated views; "
                    "use height_fit=off when outlines differ in size.",
                )
                code = 1
        if command == "planar_match" and result.get("plan_ok"):
            # A new match replaces the old goal, including when contact is unavailable.
            api._roborag_planar_goal = result.get("transfer_args")
            api._roborag_planar_goal_step = api.step
        if command == "planar_check":
            result["goal_measured_at_step"] = getattr(api, "_roborag_planar_goal_step", None)
        correction = result.pop("correction_args", None)
        if correction is not None:
            result["correction_geometry"] = {
                "contact_xyz": [correction[k] for k in ("x", "y", "z")],
                "goal_xy": [correction[k] for k in ("to_x", "to_y")],
                "open_deg": correction["open_deg"],
                "yaw": correction["yaw"],
            }
        for key in ("transfer_args", "check_args"):
            result.pop(key, None)
    result.update(
        role="CURRENT_RGB_MEASUREMENT",
        measured_at_step=api.step,
        source="HexaShell calibrated planar RGB geometry; motion programs excluded",
    )
    return result, code
