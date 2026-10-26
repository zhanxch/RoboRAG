"""Register observations tools for the current episode."""

from typing import Literal

from mcp.types import CallToolResult

from roborag.mcp.descriptions import tool_description


def register_tools(server, memory, read_only):
    @server.tool(annotations=read_only, description=tool_description("observe"))
    def observe(
        cameras: list[Literal["head", "wrist_l", "wrist_r"]] = ["head"], calibration: bool = False
    ) -> CallToolResult:
        memory.audit(
            "observe",
            dict(cameras=cameras, calibration=calibration),
            dict(step=memory.state()["step"]),
        )
        return memory.observation(cameras, calibration=calibration)
