"""One live episode: references, plan, observation snapshots and simulator exchange.

This is session state, not the retired short-term retrieval memory.
"""

import asyncio
import base64
import json
import time
from pathlib import Path

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent

from roborag.bank.store import BankStore
from roborag.io import write_json


class EpisodeSession:
    def __init__(
        self,
        bank,
        data,
        workspace,
        output,
        task,
        no_rag=False,
        controller="delta",
        architecture="legacy",
    ):
        if architecture not in ("legacy", "staged-v1"):
            raise ValueError("Unknown execution architecture")
        if architecture == "staged-v1" and no_rag:
            raise ValueError("staged-v1 no-rag ablation is not supported; use legacy --no-rag")
        self.bank = BankStore(bank, data)
        self.workspace, self.output = Path(workspace), Path(output)
        self.task, self.no_rag = task, no_rag
        self.controller = controller
        self.architecture = architecture
        self.retrieved = False
        self.plan = None
        self.references = set()
        self.lock = asyncio.Lock()
        self.execution = None
        if architecture == "staged-v1":
            from roborag.execution.episode import StagedEpisode

            self.execution = StagedEpisode(self)

    def audit(self, tool, request, result):
        with (self.output / "mcp-calls.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {"time": time.time(), "tool": tool, "request": request, "result": result},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def state(self):
        return json.loads((self.workspace / "obs/state.json").read_text())

    def observation(self, cameras, *, calibration=False, receipt=None):
        unresolved_response = bool(
            self.execution
            and (self.execution.last_action or {}).get("id") == self.execution.in_flight
            and (self.execution.last_action or {}).get("receipt_status") == "unknown"
        )
        if self.execution and self.execution.in_flight is not None and not unresolved_response:
            raise ToolError("Wait for the pending action before reading a coherent observation")
        state = self.state()
        evidence_id = self.execution.observation(state, cameras) if self.execution else None
        if not calibration:
            state.pop("cameras", None)
        else:
            state["cameras"] = {c: state["cameras"][c] for c in cameras}
        packet = {"current_observation": state}
        if self.execution:
            packet["observation_id"] = evidence_id
            packet["stage_execution"] = dict(
                plan_version=self.plan["version"] if self.plan else None,
                stage_id=self.execution.active,
                latest_action_id=(self.execution.last_action or {}).get("id"),
                latest_action_stage_id=(self.execution.last_action or {}).get("stage_id"),
                effect_status="unknown until assessed against current evidence",
            )
        if receipt is not None:
            packet["execution"] = receipt
        content = [TextContent(type="text", text=json.dumps(packet, separators=(",", ":")))]
        for camera in cameras:
            if camera not in ("head", "wrist_l", "wrist_r"):
                raise ToolError("Unknown camera")
            content += [
                TextContent(type="text", text=f"CURRENT {camera}, step {state['step']}"),
                ImageContent(
                    type="image",
                    mimeType="image/png",
                    data=base64.b64encode(
                        (self.workspace / f"obs/{camera}.png").read_bytes()
                    ).decode(),
                ),
            ]
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
                raise ToolError(
                    f"Simulator unavailable: {detail['error']}. End this episode as incomplete; do not retry motion."
                )
            if time.monotonic() - started > 540:
                raise TimeoutError("Robot response timeout; do not resubmit without observing")
            await asyncio.sleep(0.05)
        receipt = json.loads(response.read_text())
        if self.execution and any(
            key in request for key in ("measure", "planar_measure", "rim_measure")
        ):
            native_receipt = self.output / "outbox" / name
            snapshot = (
                json.loads(native_receipt.read_text()).get("observation")
                if native_receipt.exists()
                else None
            )
            self.execution.measurement(request, receipt, self.state(), snapshot)
        return receipt
