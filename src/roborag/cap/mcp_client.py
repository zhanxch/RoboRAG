"""Standard MCP stdio client for CaP; no robot shape or tool names are assumed."""
import asyncio
from concurrent.futures import Future
from copy import deepcopy
from datetime import timedelta
import json
import queue
import threading


class MCPError(RuntimeError):
    pass


class ToolError(MCPError):
    def __init__(self, result):
        self.result = result
        super().__init__("; ".join(c.get("text", "") for c in result.get("content", [])) or "tool failed")


def wire(value):
    return value.model_dump(mode="json", by_alias=True, exclude_none=True)


async def list_all(method, field):
    from mcp.types import PaginatedRequestParams
    rows, seen, cursor = [], set(), None
    while True:
        page = await method(params=PaginatedRequestParams(cursor=cursor))
        rows.extend(wire(x) for x in getattr(page, field))
        cursor = page.nextCursor
        if cursor is None:
            return rows
        if cursor in seen:
            raise MCPError("server repeated a discovery cursor")
        seen.add(cursor)


class MCPClient:
    def __init__(self, command, args=(), *, env=None, timeout=30, errlog=None):
        self.command, self.args, self.env, self.timeout = command, list(args), env, timeout
        self.errlog = errlog
        self.requests, self.ready = queue.Queue(), Future()
        self.closed, self.failure = False, None
        self.loop = self.task = None
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()
        try:
            self.catalog = self.ready.result(timeout=timeout + 5)
        except BaseException:
            self.close()
            raise
        self.tools = {tool["name"]: tool for tool in self.catalog["tools"]}

    def _thread_main(self):
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            if not self.ready.done():
                self.ready.set_exception(MCPError(self.failure))
            while not self.requests.empty():
                pending = self.requests.get_nowait()
                if pending is not None and not pending[2].done():
                    pending[2].set_exception(MCPError(self.failure))

    async def _serve(self):
        from mcp import ClientSession, StdioServerParameters, McpError
        from mcp.client.stdio import stdio_client
        from pydantic import AnyUrl
        self.loop, self.task = asyncio.get_running_loop(), asyncio.current_task()
        parameters = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        async with stdio_client(parameters, **({"errlog": self.errlog} if self.errlog else {})) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self.timeout)) as session:
                init = await session.initialize()
                caps = init.capabilities
                catalog = {"initialization": wire(init), "tools": [], "resources": [], "resource_templates": []}
                if caps.tools is not None:
                    catalog["tools"] = await list_all(session.list_tools, "tools")
                if caps.resources is not None:
                    catalog["resources"] = await list_all(session.list_resources, "resources")
                    catalog["resource_templates"] = await list_all(session.list_resource_templates, "resourceTemplates")
                self.ready.set_result(catalog)
                while True:
                    request = await asyncio.to_thread(self.requests.get)
                    if request is None:
                        break
                    method, arguments, future = request
                    try:
                        if method == "call_tool":
                            result = wire(await session.call_tool(**arguments))
                        else:
                            result = wire(await session.read_resource(AnyUrl(arguments["uri"])))
                        future.set_result(result)
                    except McpError as exc:
                        # Protocol errors are responses, not permission to replay a call.
                        # SDK timeouts use error code 408 and make execution uncertain.
                        if exc.error.code != 408:
                            future.set_exception(MCPError(str(exc)))
                            continue
                        self.failure = str(exc)
                        future.set_exception(MCPError("MCP response timeout; execution state unknown; session closed"))
                        raise
                    except BaseException as exc:
                        self.failure = f"{type(exc).__name__}: {exc}"
                        future.set_exception(MCPError("MCP session failed; execution state unknown; no automatic retry"))
                        raise

    def _request(self, method, **arguments):
        if self.closed or self.failure is not None:
            raise MCPError(f"MCP session unavailable: {self.failure}")
        future = Future()
        self.requests.put((method, arguments, future))
        try:
            return future.result(timeout=self.timeout + 5)
        except TimeoutError:
            self.close()
            raise MCPError("MCP response timeout; execution state unknown; session closed")

    def call_tool(self, name, arguments=None):
        from jsonschema import validators
        from referencing import Registry
        if not isinstance(name, str) or name not in self.tools:
            raise ValueError(f"tool not discovered: {name!r}")
        arguments = {} if arguments is None else arguments
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        json.dumps(arguments, allow_nan=False)
        schema = self.tools[name]["inputSchema"]
        validator = validators.validator_for(schema)
        validator.check_schema(schema)
        validator(schema, registry=Registry()).validate(arguments)
        result = self._request("call_tool", name=name, arguments=deepcopy(arguments))
        if result.get("isError", False):
            raise ToolError(result)
        return result  # Preserve structuredContent AND text/image/resource content.

    def read_resource(self, uri):
        if self.catalog["initialization"]["capabilities"].get("resources") is None:
            raise ValueError("server does not advertise resources")
        if not isinstance(uri, str) or not uri:
            raise ValueError("resource URI must be a nonempty string")
        return self._request("read_resource", uri=uri)

    def close(self):
        if not self.closed:
            self.closed = True
            self.requests.put(None)
            self.thread.join(timeout=5)
            if self.thread.is_alive() and self.loop is not None and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.task.cancel)
                self.thread.join(timeout=5)

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
