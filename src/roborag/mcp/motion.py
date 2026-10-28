"""V3 action chunks with target timing normalization and original stop checks."""

from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult

from roborag.action_targets import compile_targets_with_timing
from roborag.rag_execution import expand_delta_actions, validate_chunk

from .schemas import DeltaSegment, TargetSegment, Waypoint


def register_tools(server, memory, read_only):
    @server.tool()
    async def execute_chunk(subgoal: str, waypoints: list[Waypoint], expected_feedback: str,
                            cameras: list[Literal["head", "wrist_l", "wrist_r"]] = ["head"]) -> CallToolResult:
        """Execute 1-8 GPT-authored absolute world TCP waypoints; return final RGB/state.

        xyz is fingertip center in meters (link6 +0.145m local X); point sets approach,
        open_axis sets finger separation direction. gripper 1=open/0=closed, applied AFTER
        reaching xyz. Each waypoint may add hold_steps (0-25). IK interpolates locally.
        home=true (without xyz/point) returns that arm to its starting joint pose;
        withdraw from objects before returning home. Success may require both arms home.
        Stop a chunk after uncertain grasp/lift or release so you can inspect feedback.
        Failed IK, clipped/unreached poses stop the tail. Commands do not verify grasp.
        """
        if memory.plan is None:
            raise ToolError("record_plan must precede execution")
        values = [w.model_dump() for w in waypoints]
        try:
            validate_chunk(values)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        async with memory.lock:
            if memory.state()["episode_over"]:
                return memory.observation(cameras)
            request = dict(chunk=values, subgoal=subgoal, expected_feedback=expected_feedback)
            receipt = await memory.submit(request)
            memory.audit("execute_chunk", request, receipt)
            return memory.observation(cameras, receipt=receipt["result"])

    if memory.controller == "delta":
        server.remove_tool("execute_chunk")

        @server.tool(name="execute_chunk")
        async def execute_delta_chunk(subgoal: str, expected_feedback: str, actions: list[list[float]] | None = None,
                                      segments: list[DeltaSegment] | None = None,
                                      targets: list[TargetSegment] | None = None,
                                      return_home: bool = False,
                                      cameras: list[Literal["head", "wrist_l", "wrist_r"]] = ["head", "wrist_r"]) -> CallToolResult:
            """Submit NEW action rows in the SAME format as historical action references.

            Each row: [L dx,dy,dz,rx,ry,rz,gripper, R dx,dy,dz,rx,ry,rz,gripper].
            EE=link6, not TCP. World translation meters, WORLD ROTVEC radians (not Euler).
            Start from current action_state EE poses. Accumulate subsequent rows from
            PREVIOUS REQUESTED targets. Gripper 0=closed/1=open, held per row. 25Hz.
            Per-arm limits per row: translation norm<=0.035m; rotation norm<=0.35rad.
            Prefer segments=[{delta:[14 numbers],repeat:N},...] for repeated rows.
            Segments expand EXACTLY to actions; do not supply both.
            To avoid calculating deltas, use targets=[{steps:N,left:{tcp_xyz:[x,y,z],
            quaternion_wxyz:[w,x,y,z],gripper:0},right:{...}},...]. Select ONLY one
            of actions, segments, targets. Omitted arms/fields hold previous targets.
            Targets linearly interpolate LINK6 translation and world rotation to the
            requested fingertip endpoints (local +X 0.145m); this is not straight TCP
            motion during rotation. Gripper applies from the segment's first row.
            Too-short target segments are extended to meet both arms' per-row limits;
            requested steps are never shortened. execution.target_timing reports requested
            and compiled steps when extended; executed_rows reports actual execution.
            No per-call step cap; same delta limits and native episode budget apply.
            No IK/grasp/fold planner is added.
            Both arms execute together. Large measured tracking error (>4cm) stops tail.
            No implicit grasp verification. Inspect returned RGB and revise from new state.
            When the object is released and fingers clear, return_home=true returns
            both arms to their initial joint poses, satisfying the task's final requirement.
            """
            if memory.plan is None:
                raise ToolError("record_plan must precede execution")
            if return_home and (actions or segments or targets):
                raise ToolError("Leave actions, segments and targets empty for return_home")
            if not return_home and sum(bool(value) for value in (actions, segments, targets)) != 1:
                raise ToolError("Choose exactly one of actions, segments or targets")
            encoding = [s.model_dump() for s in segments] if segments else None
            target_values = [s.model_dump(exclude_none=True) for s in targets] if targets else None
            if not return_home and not targets:
                try:
                    actions = expand_delta_actions(actions, encoding)
                except ValueError as exc:
                    raise ToolError(str(exc)) from exc
            async with memory.lock:
                if memory.state()["episode_over"]:
                    return memory.observation(cameras)
                target_timing = None
                if target_values:
                    try:
                        actions, timing = compile_targets_with_timing(memory.state()["action_state"], target_values)
                    except ValueError as exc:
                        raise ToolError(str(exc)) from exc
                    if any(t["compiled_steps"] != t["requested_steps"] for t in timing):
                        target_timing = timing
                request = dict(subgoal=subgoal, expected_feedback=expected_feedback)
                if return_home:
                    request["chunk"] = [dict(arm=side,home=True,gripper=1) for side in ("left", "right")]
                else:
                    request["actions"] = actions
                    if encoding:
                        request["action_encoding"] = dict(segments=encoding)
                    elif target_values:
                        request["action_encoding"] = dict(targets=target_values)
                        if target_timing is not None:
                            request["action_encoding"]["target_timing"] = target_timing
                receipt = await memory.submit(request)
                if target_timing is not None:
                    receipt = {**receipt, "result": {**receipt["result"], "target_timing": target_timing}}
                memory.audit("execute_chunk", request, receipt)
                return memory.observation(cameras, receipt=receipt["result"])
