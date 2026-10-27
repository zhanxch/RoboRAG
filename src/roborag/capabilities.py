"""Reusable measurement capabilities shared by MCP and the simulator.

Only trusted code registers implementations here. Task JSON selects names;
it cannot import arbitrary modules. Imports stay lazy so listing tasks needs
neither MCP nor simulation dependencies.
"""

from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True)
class MeasurementCapability:
    name: str
    request_key: str
    tool_module: str
    register_function: str
    backend_module: str
    measure_function: str
    # Preserve the order of model-facing tools in the evaluated protocols.
    tool_position: Literal["before_evidence", "after_observe"]

    def register_tools(self, server, session, read_only):
        register = getattr(import_module(self.tool_module), self.register_function)
        register(server, session, read_only)

    def measure(self, api, hexashell, request):
        measure = getattr(import_module(self.backend_module), self.measure_function)
        return measure(api, hexashell, request)


CAPABILITIES = MappingProxyType(
    {
        capability.name: capability
        for capability in (
            MeasurementCapability(
                name="cloth_geometry",
                request_key="measure",
                tool_module="roborag.tasks.fold_clothes.tools",
                register_function="register_tools",
                backend_module="roborag.tasks.fold_clothes.measurements",
                measure_function="measure_scene",
                tool_position="after_observe",
            ),
            MeasurementCapability(
                name="planar_geometry",
                request_key="planar_measure",
                tool_module="roborag.tasks.push_T.tools",
                register_function="register_planar_tool",
                backend_module="roborag.tasks.push_T.measurements",
                measure_function="measure_planar",
                tool_position="before_evidence",
            ),
            MeasurementCapability(
                name="rim_geometry",
                request_key="rim_measure",
                tool_module="roborag.tasks.pour_balls_into_vase.tools",
                register_function="register_tools",
                backend_module="roborag.tasks.pour_balls_into_vase.measurements",
                measure_function="measure_rim",
                tool_position="before_evidence",
            ),
        )
    }
)


def capability(name: str) -> MeasurementCapability:
    try:
        return CAPABILITIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown measurement capability: {name}") from exc


def measurement_for(request, enabled) -> MeasurementCapability | None:
    """Resolve one measurement, enforcing the same capabilities as the MCP server."""
    matches = [item for item in CAPABILITIES.values() if item.request_key in request]
    if len(matches) > 1:
        raise ValueError("A request must contain only one measurement")
    if not matches:
        return None
    selected = matches[0]
    if selected.name not in enabled:
        raise ValueError(f"Measurement capability is not enabled: {selected.name}")
    return selected
