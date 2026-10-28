"""Retrieve example overviews and expand explicitly cited historical experiences."""

import base64
import json
from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent

from roborag.graph_chunks import select_reference_window
from roborag.graph_retrieval import RetrievalArm
from roborag.graph_server import frame_bytes
from roborag.sop_store import SOPStore

SOP_RETRIEVAL_DESCRIPTION = """Retrieve 1-2 manipulation SOP demonstrations as historical examples, not a new plan.

Describe the CURRENT visible scene. Use once before planning; query again only
for an information gap/recovery. Chinese queries match the original Chinese labels.
Each overview preserves the v3 historical scene, operation directory and reported
outcome. available_sections lists the independently stored experience sections.
Use read_action_reference with the cited source and section to expand one directly.
The SOP's recorded order does not constrain your current plan or execution.
This package contains measured summaries and selected images, without raw action arrays.
"""

SOP_REFERENCE_DESCRIPTION = """Expand a cited manipulation SOP experience directly, without searching again.

Use source=demonstration_id:phase_id and one of its available_sections from retrieval.
Returns the archived before/after robot states, measured displacement and gripper
summary, operation, historical feedback and optional recorded head images.
Images are historical and do not replace current feedback. Command events do not
prove physical grasp/release. Missing sections return an explicit error; there is
no fallback to a full trajectory or another demonstration. include_actions must
remain false because raw action arrays are not stored in this package.
"""


def register_tools(server, memory, read_only):
    @server.tool(annotations=read_only,
                 description=SOP_RETRIEVAL_DESCRIPTION if isinstance(memory.bank, SOPStore) else None)
    def retrieve_evidence(goal: str, observation: str, mode: Literal["plan", "action"] = "plan",
                          arm: RetrievalArm = "", held_object: str = "", target_object: str = "") -> str:
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
            try:
                result = memory.bank.retriever().retrieve(query)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
        packet = json.loads(result["model_context"])
        for example in packet.get("memories", []):
            memory.references.add(example["source"])
            for phase in example.get("episode_trace", []):
                memory.references.add(example["source"] + ":" + phase["id"])
        memory.retrieved = True
        memory.audit("retrieve_evidence", query, result)
        return result["model_context"]

    @server.tool(annotations=read_only,
                 description=SOP_REFERENCE_DESCRIPTION if isinstance(memory.bank, SOPStore) else None)
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
        illustrations = None
        if isinstance(memory.bank, SOPStore):
            try:
                packet, illustrations = memory.bank.reference(source, section, include_actions=include_actions, images=images)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
        else:
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
        if illustrations is not None:
            for label, raw in illustrations:
                content += [TextContent(type="text", text=label),
                            ImageContent(type="image", mimeType="image/jpeg", data=base64.b64encode(raw).decode())]
        elif images:
            graph = memory.bank.graph(graph_id)
            for label, frame in (("before", window["start"]), ("after", window["end"])):
                content += [TextContent(type="text", text=f"HISTORICAL {source} {label} frame {frame}"),
                            ImageContent(type="image", mimeType="image/jpeg", data=base64.b64encode(
                                frame_bytes(str(memory.bank.source(graph)), "cam_head", frame)).decode())]
        return CallToolResult(content=content)
