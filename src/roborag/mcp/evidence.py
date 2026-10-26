"""Register evidence tools for the current episode."""

import base64
import json
from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent

from roborag.bank.chunks import select_reference_window
from roborag.bank.store import frame_bytes
from roborag.mcp.descriptions import tool_description
from roborag.tasks import task_spec


def register_tools(server, memory, read_only):
    @server.tool(annotations=read_only, description=tool_description("retrieve_evidence"))
    def retrieve_evidence(
        goal: str,
        observation: str,
        mode: Literal["plan", "action"] = "plan",
        arm: str = "",
        held_object: str = "",
        target_object: str = "",
    ) -> str:
        query = dict(
            mode=mode,
            task=memory.task,
            goal=goal,
            observation=observation,
            arm=arm,
            held_object=held_object,
            target_object=target_object,
            k=2,
            max_context_chars=6000,
        )
        if memory.no_rag:
            result = dict(
                model_context=json.dumps({"purpose": "no_rag_control", "memories": []}),
                diagnostics={},
            )
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

    @server.tool(annotations=read_only, description=tool_description("read_action_reference"))
    def read_action_reference(
        source: str,
        include_actions: bool = False,
        images: bool = True,
        section: Literal["start", "end", "close", "open"] = "start",
    ) -> CallToolResult:
        if source not in memory.references or memory.no_rag:
            raise ToolError("Choose a source from retrieve_evidence first")
        graph_id, node_id = source.split(":", 1)
        retriever = memory.bank.retriever()
        phase = next(p for p in retriever.index["phases"] if p["id"] == source)
        graph = memory.bank.graph(graph_id)
        window, selection = select_reference_window(
            memory.bank.source(graph), retriever.by_phase[(graph_id, node_id)], section
        )
        transition = retriever.chunk(window["id"])
        if not include_actions:
            transition.pop("actions", None)
        packet = {
            "role": "HISTORICAL_REFERENCE",
            "source": source,
            "phase_interval": [phase["start"], phase["end"]],
            "window_section": section,
            "window_selection": selection,
            "before": phase["cue"],
            "operation": phase["action"],
            "phase_feedback": phase["feedback"],
            "transition": transition,
        }
        context_images = None
        if task_spec(memory.task).whole_phase_reference:
            from roborag.bank.reference import phase_context

            packet["phase_context"], context_images = phase_context(
                memory.bank.source(graph), phase
            )
        if phase.get("interaction"):
            from roborag.bank.interactions import interaction_card

            packet["interaction_evidence"] = interaction_card(phase, retriever.index["phases"])
        memory.audit(
            "read_action_reference",
            dict(source=source, include_actions=include_actions, images=images, section=section),
            packet,
        )
        content = [
            TextContent(
                type="text", text=json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
            )
        ]
        if images:
            graph = memory.bank.graph(graph_id)
            views = (
                [(f"phase context {camera}", frame, camera) for frame, camera in context_images]
                if context_images is not None
                else [
                    (label, frame, "cam_head")
                    for label, frame in (("before", window["start"]), ("after", window["end"]))
                ]
            )
            for label, frame, camera in views:
                content += [
                    TextContent(type="text", text=f"HISTORICAL {source} {label} frame {frame}"),
                    ImageContent(
                        type="image",
                        mimeType="image/jpeg",
                        data=base64.b64encode(
                            frame_bytes(str(memory.bank.source(graph)), camera, frame)
                        ).decode(),
                    ),
                ]
        return CallToolResult(content=content)
