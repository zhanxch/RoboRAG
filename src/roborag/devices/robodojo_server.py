"""Standard MCP tools/resources backed by native RoboDojo observation and control."""
import json


def create_server(device):
    from mcp.server.lowlevel import Server
    from mcp.types import Tool, Resource, TextContent, CallToolResult
    from mcp.server.lowlevel.helper_types import ReadResourceContents
    server = Server("RoboDojo")
    arm = {"type": "string", "enum": ["left", "right"]}
    vector = lambda n: {"type": "array", "items": {"type": "number"}, "minItems": n, "maxItems": n}
    def schema(properties, required=()):
        return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}
    tools = {
        "robot_state": (device.observe, "Read measured link6 poses/joints, commanded gripper and task instruction. No object truth or score.", schema({})),
        "camera_observe": (device.camera_observe, "Read fresh RGB as MCP image content; camera labels precede images.",
                           schema({"cameras": {"type": "array", "items": {"type": "string"}, "minItems": 1}})),
        "move_ee": (device.move, "Move link6 to absolute position (metres, environment origin/world axes). Optional unit quaternion wxyz; omitted preserves orientation. Local interpolation, IK and tracking checks; preserves grippers.",
                    schema({"arm": arm, "position": vector(3), "quaternion_wxyz": vector(4)}, ("arm", "position"))),
        "set_gripper": (device.gripper, "Command gripper opening 0 closed..1 open. Holds both arm poses. Completion does not establish grasp success.",
                        schema({"arm": arm, "opening": {"type": "number", "minimum": 0, "maximum": 1}}, ("arm", "opening"))),
        "execution_status": (device.execution_status, "Read last execution receipt, known executed step count and stopped state.", schema({})),
        "stop_motion": (device.stop, "Latch no-further-dispatch after the current blocking call. Not an asynchronous emergency stop.", schema({})),
    }
    if device.joint_limits:
        tools["move_joints"] = (device.move_joints, "Move one arm to absolute joint positions in radians within native limits; interpolate locally, keep other arm and grippers.",
                                schema({"arm": arm, "positions": vector(len(device.joint_limits["left"]))}, ("arm", "positions")))
    resources = {
        "robodojo://robot": ("Robot reference", lambda: dict(device.describe(), joint_limits_rad=device.joint_limits)),
        "robodojo://state": ("Current proprioception", device.observe),
        "robodojo://execution": ("Execution receipt", device.execution_status),
    }

    @server.list_tools()
    async def list_tools():
        return [Tool(name=name, description=description, inputSchema=parameters)
                for name, (_, description, parameters) in tools.items()]

    @server.call_tool()
    async def call_tool(name, arguments):
        try:
            value = tools[name][0](**arguments)
            if name == "camera_observe":
                return CallToolResult(content=value)
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(value, allow_nan=False))],
                                  structuredContent=value)
        except Exception as exc:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=f"{type(exc).__name__}: {exc}")])

    @server.list_resources()
    async def list_resources():
        return [Resource(uri=uri, name=name, mimeType="application/json") for uri, (name, _) in resources.items()]

    @server.list_resource_templates()
    async def list_templates():
        return []

    @server.read_resource()
    async def read_resource(uri):
        return [ReadResourceContents(content=json.dumps(resources[str(uri)][1](), allow_nan=False), mime_type="application/json")]

    return server


def main():
    import argparse
    import signal
    from pathlib import Path
    from .worker_proxy import WorkerProxy
    import anyio
    from mcp.server.stdio import stdio_server
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    device = WorkerProxy(json.loads(args.worker_config.read_text()))
    try:
        server = create_server(device)
        async def run():
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())
        anyio.run(run)
    finally:
        device.close()


if __name__ == "__main__":
    main()
