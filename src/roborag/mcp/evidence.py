"""Unchanged v3 whole-example retrieval and reference expansion."""

import base64
import json
from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent

from roborag.graph_chunks import select_reference_window
from roborag.graph_server import frame_bytes


def register_tools(server, memory, read_only):
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
                              section: Literal["start", "end", "close", "open"] = "start") -> CallToolResult:
        """Expand graph_id:phase_id from a retrieved graph into historical before/action/after evidence.

        Recorded actions are world link6 deltas. When the live controller uses that same
        interface, learn their scale/orientation/gripper timing, but adapt to CURRENT images.
        An absolute TCP controller requires a coordinate conversion; never blindly replay rows.
        Images are explicitly historical and must not replace current feedback.
        section=end selects the final window of a phase, useful for release/withdrawal.
        section=close/open selects a window around the first gripper command crossing
        0.5 in this phase. Use close for grasp posture; phase starts can be far away.
        If no such event exists, the response explicitly falls back to phase start.
        Default summaries include displacement/gripper timing; raw arrays are optional.
        """
        if source not in memory.references or memory.no_rag:
            raise ToolError("Choose a source from retrieve_evidence first")
        graph_id, node_id = source.split(":", 1)
        retriever = memory.bank.retriever()
        phase = next(p for p in retriever.index["phases"] if p["id"] == source)
        graph = memory.bank.graph(graph_id)
        window, selection = select_reference_window(memory.bank.source(graph),
            retriever.by_phase[(graph_id, node_id)], section)
        transition = retriever.chunk(window["id"])
        if not include_actions:
            transition.pop("actions", None)
        packet = {"role": "HISTORICAL_REFERENCE", "source": source,
                  "phase_interval": [phase["start"], phase["end"]], "window_section": section,
                  "window_selection": selection,
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
