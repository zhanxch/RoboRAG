"""Unchanged v3 episode state and simulator exchange."""

import asyncio
import base64
import json
import time
from pathlib import Path

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent

from roborag.episode_files import write_json
from roborag.graph_server import BankStore


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
