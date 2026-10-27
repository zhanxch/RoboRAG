"""Compose the original v3 tool surface in its original order."""

import argparse
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from . import evidence, motion, observations, planning
from .session import EpisodeMemory


def create_server(memory):
    server = MCPServer("RoboRAG", instructions=(
        "First observe, retrieve relevant historical evidence, then record YOUR new plan. "
        "Historical graphs are examples, not executable plans. Ground every TCP in current images. "
        "execute_chunk is a short batch of your new actions; it returns current images and measured robot state."))
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    evidence.register_tools(server, memory, read_only)
    observations.register_tools(server, memory, read_only)
    planning.register_tools(server, memory, read_only)
    motion.register_tools(server, memory, read_only)
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
