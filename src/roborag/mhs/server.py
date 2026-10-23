"""MCP stdio binding for the RoboRAG MHS skill contract."""
import argparse
import json
import threading


def create_server(device):
    from mcp.server.fastmcp import FastMCP
    from pydantic import StrictFloat, StrictStr
    server = FastMCP("RoboRAG MHS", log_level="WARNING")
    lock = threading.Lock()

    @server.tool()
    def mhs_describe() -> dict:
        """Discover device identity, skill semantics, coordinate frame and enforced limits."""
        with lock:
            return device.describe()

    @server.resource("mhs://device/capabilities")
    def capabilities() -> str:
        """RoboRAG MHS v0.1 device reference."""
        return json.dumps(mhs_describe(), ensure_ascii=False)

    @server.tool()
    def mhs_observe() -> dict:
        """Read a fresh session-tagged observation, with state provenance and available RGB."""
        with lock:
            return device.observe()

    @server.tool()
    def mhs_move(arm: StrictStr, position: list[StrictFloat]) -> dict:
        """Blocking move to absolute xyz metres in the declared frame; preserve orientation and grip."""
        with lock:
            return device.move(arm, position)

    @server.tool()
    def mhs_gripper(arm: StrictStr, opening: StrictFloat) -> dict:
        """Command normalized opening (0 closed, 1 open). Receipt is not grasp-success evidence."""
        with lock:
            return device.gripper(arm, opening)

    @server.tool()
    def mhs_stop() -> dict:
        """Latch stop after the active blocking call; this is not an asynchronous hardware emergency stop."""
        with lock:
            return device.stop()

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
