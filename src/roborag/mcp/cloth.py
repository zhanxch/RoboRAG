"""Register cloth tools for the current episode."""

import json
from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError

from roborag.mcp.descriptions import tool_description


def register_tools(server, memory, read_only):
    @server.tool(annotations=read_only, description=tool_description("measure_scene"))
    async def measure_scene(
        command: Literal["cloth_landmarks", "table_xy"] = "cloth_landmarks",
        u: int | None = None,
        v: int | None = None,
        outline: list[list[int]] | None = None,
    ) -> dict:
        if (u is None) != (v is None) or (command == "table_xy" and u is None):
            raise ToolError("Provide both pixel coordinates u,v")
        arguments = dict(u=u, v=v, outline=json.dumps(outline) if outline is not None else None)
        async with memory.lock:
            state = memory.state()
            if state["episode_over"]:
                return dict(
                    episode_over=True,
                    measured_at_step=state["step"],
                    plan_ok=False,
                    plan_fail_reason="episode_over",
                )
            request = dict(
                measure=dict(command=command, arguments=arguments),
                subgoal="Measure current RGB geometry",
            )
            receipt = await memory.submit(request)
            memory.audit("measure_scene", request, receipt)
            return receipt["result"]
