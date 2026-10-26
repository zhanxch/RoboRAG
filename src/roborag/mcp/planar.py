"""Opt-in planar perception for the push_T experiment only."""

from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError

from roborag.mcp.descriptions import tool_description


def register_planar_tool(server, memory, read_only):
    @server.tool(annotations=read_only, description=tool_description("measure_planar"))
    async def measure_planar(
        command: Literal["table_plane", "planar_height", "planar_match", "planar_check"],
        u: int | None = None,
        v: int | None = None,
        ref_u: int | None = None,
        ref_v: int | None = None,
        other_u: int | None = None,
        other_v: int | None = None,
        camera: Literal["head", "wrist_l", "wrist_r"] = "head",
        other_camera: Literal["head", "wrist_l", "wrist_r"] = "wrist_l",
        z: float | None = None,
        surface_z: float | None = None,
        reference_z: float | None = None,
        color_tol: float = 45,
        height_fit: Literal["auto", "off"] = "off",
    ) -> dict:
        arguments = dict(
            u=u,
            v=v,
            ref_u=ref_u,
            ref_v=ref_v,
            other_u=other_u,
            other_v=other_v,
            camera=camera,
            other_camera=other_camera,
            z=z,
            surface_z=surface_z,
            reference_z=reference_z,
            color_tol=color_tol,
            height_fit=height_fit,
        )
        required = {
            "table_plane": (),
            "planar_check": (),
            "planar_height": ("u", "v", "other_u", "other_v", "z"),
            "planar_match": ("u", "v", "ref_u", "ref_v", "surface_z", "reference_z"),
        }
        if any(arguments[k] is None for k in required[command]):
            raise ToolError(
                "Missing required measurement arguments: " + ", ".join(required[command])
            )
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
                planar_measure=dict(command=command, arguments=arguments),
                subgoal="Measure current planar RGB geometry",
            )
            receipt = await memory.submit(request)
            memory.audit("measure_planar", request, receipt)
            return receipt["result"]
