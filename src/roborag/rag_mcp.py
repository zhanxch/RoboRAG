"""RoboRAG evidence + live robot tools, exposed through the official MCP SDK.

One server per episode. History stays in the caller's persistent GPT session.
The server retrieves evidence; it never authors a task plan or replays a demo.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import time
from typing import Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from pydantic import BaseModel, Field

from .graph_server import BankStore, frame_bytes
from .rag_execution import validate_chunk, validate_delta_actions


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


class PlanStep(BaseModel):
    subgoal: str
    current_objects: str
    completion_observation: str
    evidence_refs: list[str] = Field(default_factory=list)


class Waypoint(BaseModel):
    arm: Literal["left", "right"]
    home: bool = False
    xyz: list[float] | None = None
    point: Literal["down", "forward", "down45"] | None = None
    open_axis: Literal["x", "y", "z"] = "x"
    gripper: float | None = None
    hold_steps: int = 0


class EpisodeMemory:
    def __init__(self, bank, data, workspace, output, task, no_rag=False, controller="tcp"):
        self.bank = BankStore(bank, data)
        self.workspace, self.output = Path(workspace), Path(output)
        self.task, self.no_rag = task, no_rag
        self.controller = controller
        self.retrieved = False
        self.plan = None
        self.references = set()
        self.lock = asyncio.Lock()

    def audit(self, tool, request, result):
        with (self.output / "mcp-calls.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": time.time(), "tool": tool, "request": request,
                                     "result": result}, ensure_ascii=False) + "\n")

    def state(self):
        return json.loads((self.workspace / "obs/state.json").read_text())

    def observation(self, cameras, *, calibration=False, receipt=None):
        state = self.state()
        if not calibration:
            state.pop("cameras", None)
        else:
            state["cameras"] = {c: state["cameras"][c] for c in cameras}
        packet = {"current_observation": state}
        if receipt is not None:
            packet["execution"] = receipt
        content = [TextContent(type="text", text=json.dumps(packet, separators=(",", ":")))]
        for camera in cameras:
            if camera not in ("head", "wrist_l", "wrist_r"):
                raise ToolError("Unknown camera")
            content += [TextContent(type="text", text=f"CURRENT {camera}, step {state['step']}"),
                        ImageContent(type="image", mimeType="image/png", data=base64.b64encode(
                            (self.workspace / f"obs/{camera}.png").read_bytes()).decode())]
        return CallToolResult(content=content)

    async def submit(self, request):
        name = f"{time.time_ns():020d}.json"
        write_json(self.workspace / "inbox" / name, request)
        response = self.workspace / "outbox" / name
        started = time.monotonic()
        while not response.exists():
            failure = self.output / "sim_failure.json"
            if failure.exists():
                detail = json.loads(failure.read_text())
                raise ToolError(f"Simulator unavailable: {detail['error']}. End this episode as incomplete; do not retry motion.")
            if time.monotonic()-started > 540:
                raise TimeoutError("Robot response timeout; do not resubmit without observing")
            await asyncio.sleep(.05)
        return json.loads(response.read_text())


def create_server(memory):
    server = MCPServer("RoboRAG", instructions=(
        "First observe, retrieve relevant historical evidence, then record YOUR new plan. "
        "Historical graphs are examples, not executable plans. Ground every TCP in current images. "
        "execute_chunk is a short batch of your new actions; it returns current images and measured robot state."))
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    @server.tool(annotations=read_only)
    def retrieve_evidence(goal: str, observation: str, mode: Literal["plan", "action"] = "plan",
                          arm: str = "", held_object: str = "", target_object: str = "") -> str:
        """Retrieve 1-2 cited examples from recorded demonstrations, not a new plan.

        Describe the CURRENT visible scene. Use once before planning; query again only
        for an information gap/recovery. Text uses Chinese labels; Chinese queries work best.
        Raw action arrays/images stay behind read_action_reference to keep context small.
        """
        query = dict(mode=mode, task=memory.task, goal=goal, observation=observation,
                     arm=arm, held_object=held_object, target_object=target_object,
                     k=2, max_context_chars=6000)
        if memory.no_rag:
            result = dict(model_context=json.dumps({"purpose": "no_rag_control", "memories": []}), diagnostics={})
        else:
            result = memory.bank.retriever().retrieve(query)
        packet = json.loads(result["model_context"])
        for example in packet.get("memories", []):
            memory.references.add(example["source"])
            for phase in example.get("episode_trace", []):
                memory.references.add(example["source"] + ":" + phase["id"])
        memory.retrieved = True
        memory.audit("retrieve_evidence", query, result)
        return result["model_context"]

    @server.tool(annotations=read_only)
    def read_action_reference(source: str, include_actions: bool = False, images: bool = True,
                              section: Literal["start", "end"] = "start") -> CallToolResult:
        """Expand graph_id:phase_id from a retrieved graph into historical before/action/after evidence.

        Recorded actions are world link6 deltas. When the live controller uses that same
        interface, learn their scale/orientation/gripper timing, but adapt to CURRENT images.
        An absolute TCP controller requires a coordinate conversion; never blindly replay rows.
        Images are explicitly historical and must not replace current feedback.
        section=end selects the final window of a phase, useful for release/withdrawal.
        Default summaries include displacement/gripper timing; raw arrays are optional.
        """
        if source not in memory.references or memory.no_rag:
            raise ToolError("Choose a source from retrieve_evidence first")
        graph_id, node_id = source.split(":", 1)
        retriever = memory.bank.retriever()
        phase = next(p for p in retriever.index["phases"] if p["id"] == source)
        window = retriever.by_phase[(graph_id, node_id)][0 if section == "start" else -1]
        transition = retriever.chunk(window["id"])
        if not include_actions:
            transition.pop("actions", None)
        packet = {"role": "HISTORICAL_REFERENCE", "source": source,
                  "phase_interval": [phase["start"], phase["end"]], "window_section": section,
                  "before": phase["cue"], "operation": phase["action"], "phase_feedback": phase["feedback"],
                  "transition": transition}
        memory.audit("read_action_reference", dict(source=source, include_actions=include_actions, images=images, section=section), packet)
        content = [TextContent(type="text", text=json.dumps(packet, ensure_ascii=False, separators=(",", ":")))]
        if images:
            graph = memory.bank.graph(graph_id)
            for label, frame in (("before", window["start"]), ("after", window["end"])):
                content += [TextContent(type="text", text=f"HISTORICAL {source} {label} frame {frame}"),
                            ImageContent(type="image", mimeType="image/jpeg", data=base64.b64encode(
                                frame_bytes(str(memory.bank.source(graph)), "cam_head", frame)).decode())]
        return CallToolResult(content=content)

    @server.tool(annotations=read_only)
    def observe(cameras: list[Literal["head", "wrist_l", "wrist_r"]] = ["head"],
                calibration: bool = False) -> CallToolResult:
        """Current RGB and measured fingertip TCP state; no simulation step or object ground truth.

        Set calibration=true initially for camera ray/plane calculations. World z is height in meters.
        Reuse fixed head calibration; moving wrist extrinsics need refreshing after movement.
        """
        memory.audit("observe", dict(cameras=cameras, calibration=calibration), dict(step=memory.state()["step"]))
        return memory.observation(cameras, calibration=calibration)

    @server.tool()
    def record_plan(current_scene: str, steps: list[PlanStep], adaptation: str) -> dict:
        """Record YOUR new plan based on current scene + retrieved evidence, before movement.

        Explain what is reused, skipped or changed from examples. Cite adopted source IDs;
        steps can differ from historical graph nodes. Call again only if your plan changes.
        """
        if not memory.retrieved:
            raise ToolError("Retrieve planning evidence before recording the new plan")
        if not steps:
            raise ToolError("Plan requires at least one step")
        for step in steps:
            if any(ref not in memory.references for ref in step.evidence_refs):
                raise ToolError("Cite only retrieved source IDs")
        memory.plan = dict(role="gpt_generated_current_plan", current_scene=current_scene,
                           steps=[s.model_dump() for s in steps], adaptation=adaptation,
                           observed_at_step=memory.state()["step"])
        memory.audit("record_plan", memory.plan, {"recorded": True})
        write_json(memory.output / "gpt-plan.json", memory.plan)
        return {"recorded": True, "steps": len(steps), "role": memory.plan["role"]}

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
        async def execute_delta_chunk(subgoal: str, actions: list[list[float]], expected_feedback: str,
                                      return_home: bool = False,
                                      cameras: list[Literal["head", "wrist_l", "wrist_r"]] = ["head", "wrist_r"]) -> CallToolResult:
            """Submit 1-24 NEW action rows in the SAME format as historical action references.

            Each row: [L dx,dy,dz,rx,ry,rz,gripper, R dx,dy,dz,rx,ry,rz,gripper].
            EE=link6, not TCP. World translation meters, WORLD ROTVEC radians (not Euler).
            Start from current action_state EE poses. Accumulate subsequent rows from
            PREVIOUS REQUESTED targets. Gripper 0=closed/1=open, held per row. 25Hz.
            Per-arm limits per row: translation norm<=0.035m; rotation norm<=0.35rad.
            Both arms execute together. Large measured tracking error (>4cm) stops tail.
            No implicit grasp verification. Inspect returned RGB and revise from new state.
            When tower stable and fingers clear, return_home=true with actions=[] returns
            both arms to their initial joint poses, satisfying the task's final requirement.
            """
            if memory.plan is None:
                raise ToolError("record_plan must precede execution")
            if return_home and actions:
                raise ToolError("Use actions=[] for return_home")
            if not return_home:
                try:
                    validate_delta_actions(actions)
                except ValueError as exc:
                    raise ToolError(str(exc)) from exc
            async with memory.lock:
                if memory.state()["episode_over"]:
                    return memory.observation(cameras)
                request = dict(subgoal=subgoal, expected_feedback=expected_feedback)
                if return_home:
                    request["chunk"] = [dict(arm=side,home=True,gripper=1) for side in ("left", "right")]
                else:
                    request["actions"] = actions
                receipt = await memory.submit(request)
                memory.audit("execute_chunk", request, receipt)
                return memory.observation(cameras, receipt=receipt["result"])

    return server


def main():
    parser = argparse.ArgumentParser()
    for name in ("bank", "data", "workspace", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--task", default="stack_blocks")
    parser.add_argument("--port", type=int, default=8773)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--controller", choices=["tcp", "delta"], default="delta")
    args = parser.parse_args()
    memory = EpisodeMemory(args.bank, args.data, args.workspace, args.output, args.task, args.no_rag, args.controller)
    import uvicorn
    app = create_server(memory).streamable_http_app(json_response=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
