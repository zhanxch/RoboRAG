"""An example arm MCP server. Its tools are specific to this fixture."""
import argparse
import json


def create_server(device):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult, TextContent
    from pydantic import StrictFloat, StrictStr
    server = FastMCP("Synthetic arm", log_level="WARNING")
    def pack(value):
        return CallToolResult(structuredContent=value, content=[TextContent(type="text", text=json.dumps(value))])

    @server.resource("fixture://arm/manual")
    def manual() -> str:
        return json.dumps(device.describe())

    @server.tool()
    def observe() -> CallToolResult:
        """Read synthetic fixture state; object coordinates and held are not real sensors."""
        return pack(device.observe())

    @server.tool()
    def move(arm: StrictStr, position: list[StrictFloat]) -> CallToolResult:
        """Move to absolute fixture-world xyz metres; preserve orientation and gripper."""
        return pack(device.move(arm, position))

    @server.tool()
    def gripper(arm: StrictStr, opening: StrictFloat) -> CallToolResult:
        """Command 0 closed, 1 open; inspect observation for retention."""
        return pack(device.gripper(arm, opening))

    @server.tool()
    def stop() -> CallToolResult:
        """Latch this synthetic session stopped."""
        return pack(device.stop())

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reject-first-move", action="store_true")
    args = parser.parse_args()
    from .synthetic import SyntheticDevice
    device = SyntheticDevice(reject_first_move=args.reject_first_move)
    try:
        create_server(device).run(transport="stdio")
    finally:
        device.stop()


if __name__ == "__main__":
    main()
