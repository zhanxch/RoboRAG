"""Register motion tools for the current episode."""

from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult

from roborag.control.execution import expand_delta_actions
from roborag.control.targets import compile_targets
from roborag.mcp.descriptions import tool_description
from roborag.mcp.schemas import DeltaSegment, TargetSegment


def register_tools(server, memory, read_only):
    @server.tool(name="execute_chunk", description=tool_description("execute_chunk"))
    async def execute_delta_chunk(
        subgoal: str,
        expected_feedback: str,
        actions: list[list[float]] | None = None,
        segments: list[DeltaSegment] | None = None,
        targets: list[TargetSegment] | None = None,
        return_home: bool = False,
        cameras: list[Literal["head", "wrist_l", "wrist_r"]] = ["head", "wrist_r"],
    ) -> CallToolResult:
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
            if target_values:
                try:
                    actions = compile_targets(memory.state()["action_state"], target_values)
                except ValueError as exc:
                    raise ToolError(str(exc)) from exc
            request = dict(subgoal=subgoal, expected_feedback=expected_feedback)
            if return_home:
                request["chunk"] = [
                    dict(arm=side, home=True, gripper=1) for side in ("left", "right")
                ]
            else:
                request["actions"] = actions
                if encoding:
                    request["action_encoding"] = dict(segments=encoding)
                elif target_values:
                    request["action_encoding"] = dict(targets=target_values)
            receipt = await memory.submit(request)
            memory.audit("execute_chunk", request, receipt)
            return memory.observation(cameras, receipt=receipt["result"])
