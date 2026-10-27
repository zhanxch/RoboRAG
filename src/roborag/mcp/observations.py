"""Unchanged v3 live observation and cloth measurement tools."""

import json
from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult


def register_tools(server, memory, read_only):
    @server.tool(annotations=read_only)
    def observe(cameras: list[Literal["head", "wrist_l", "wrist_r"]] = ["head"],
                calibration: bool = False) -> CallToolResult:
        """Current RGB and measured fingertip TCP state; no simulation step or object ground truth.

        Set calibration=true initially for camera ray/plane calculations. World z is height in meters.
        Reuse fixed head calibration; moving wrist extrinsics need refreshing after movement.
        """
        memory.audit("observe", dict(cameras=cameras, calibration=calibration), dict(step=memory.state()["step"]))
        return memory.observation(cameras, calibration=calibration)

    if memory.task == "fold_clothes":
        @server.tool(annotations=read_only)
        async def measure_scene(command: Literal["cloth_landmarks", "table_xy"] = "cloth_landmarks",
                                u: int | None = None, v: int | None = None,
                                outline: list[list[int]] | None = None) -> dict:
            """Current RGB geometry: cloth boundary, measured plane and interior grasp points.

            No motion or action steps. Use once initially before writing your plan.
            Optional u,v seed selects cloth; outline supplies ordered visible boundary pixels
            when color segmentation misses patterned cloth. table_xy projects one pixel
            using the measured plane. Check plan_ok, uncertainty and outline against RGB.
            Points describe the visible silhouette, not layer identity, grasp or task success.
            Fold targets, motion and completion decisions remain yours.
            """
            if (u is None) != (v is None) or (command == "table_xy" and u is None):
                raise ToolError("Provide both pixel coordinates u,v")
            arguments = dict(u=u, v=v, outline=json.dumps(outline) if outline is not None else None)
            async with memory.lock:
                state = memory.state()
                if state["episode_over"]:
                    return dict(episode_over=True, measured_at_step=state["step"],
                                plan_ok=False, plan_fail_reason="episode_over")
                request = dict(measure=dict(command=command, arguments=arguments), subgoal="Measure current RGB geometry")
                receipt = await memory.submit(request)
                memory.audit("measure_scene", request, receipt)
                return receipt["result"]
