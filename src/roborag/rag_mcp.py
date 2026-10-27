"""Compatibility entry point for the original v3 episode runner."""

from .mcp.schemas import ArmTarget, DeltaSegment, PlanStep, TargetSegment, Waypoint
from .mcp.server import create_server, main
from .mcp.session import EpisodeMemory

__all__ = [
    "ArmTarget", "DeltaSegment", "PlanStep", "TargetSegment", "Waypoint",
    "EpisodeMemory", "create_server", "main",
]

if __name__ == "__main__":
    main()
