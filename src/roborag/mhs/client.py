"""Synchronous CaP facade over one persistent MCP stdio session.

The event-loop thread only transports RPC; it does not run another model.
"""
import asyncio
from concurrent.futures import Future
from datetime import timedelta
import json
import queue
import threading

from .device import DeviceError


class MCPDevice:
    def __init__(self, command, args=(), *, env=None, timeout=30):
        self.command, self.args, self.env, self.timeout = command, list(args), env, timeout
        self.requests = queue.Queue()
        self.ready = Future()
        self.closed = False
        self.failure = None
        self.loop = self.task = None
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()
        try:
            self.tools = self.ready.result(timeout=timeout + 5)
        except BaseException:
            self.close()
            raise

    def _thread_main(self):
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            if not self.ready.done():
                self.ready.set_exception(DeviceError(self.failure))
            while not self.requests.empty():
                pending = self.requests.get_nowait()
                if pending is not None and not pending[2].done():
                    pending[2].set_exception(DeviceError(self.failure))

    async def _serve(self):
        self.loop, self.task = asyncio.get_running_loop(), asyncio.current_task()
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        parameters = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self.timeout)) as session:
                await session.initialize()
                discovered = await session.list_tools()
                names = {tool.name for tool in discovered.tools}
                expected = {f"mhs_{x}" for x in ("describe", "observe", "move", "gripper", "stop")}
                if not expected <= names:
                    raise DeviceError(f"server is missing MHS tools: {sorted(expected - names)}")
                self.ready.set_result([tool.model_dump(mode="json") for tool in discovered.tools])
                while True:
                    request = await asyncio.to_thread(self.requests.get)
                    if request is None:
                        break
                    method, arguments, future = request
                    try:
                        result = await session.call_tool(f"mhs_{method}", arguments)
                        if result.isError:
                            raise DeviceError("; ".join(getattr(c, "text", "") for c in result.content))
                        payload = result.structuredContent
                        if payload is None:
                            payload = json.loads(result.content[0].text)
                        if not isinstance(payload, dict):
                            raise DeviceError("MHS result must be a JSON object")
                        future.set_result(payload)
                    except DeviceError as exc:
                        # Do not export a live coroutine's traceback across threads.
                        # Callers (including unittest) may clear its frames.
                        future.set_exception(DeviceError(str(exc)))
                    except BaseException as exc:
                        self.failure = f"{type(exc).__name__}: {exc}"
                        future.set_exception(DeviceError("MHS transport failed; execution state unknown; session closed"))
                        raise

    def _call(self, method, **arguments):
        if self.closed or self.failure is not None:
            raise DeviceError(f"MHS session unavailable: {self.failure}")
        future = Future()
        self.requests.put((method, arguments, future))
        try:
            return future.result(timeout=self.timeout + 5)
        except TimeoutError:
            # No automatic retry: a timed-out command might already have moved.
            self.close()
            raise DeviceError("MHS response timeout; execution state unknown, session closed")

    def describe(self):
        result = self._call("describe")
        if result.get("interface") != "roborag-mhs/0.1":
            raise DeviceError("unsupported RoboRAG MHS version")
        return dict(result, transport="mcp-stdio", tools=self.tools)

    def observe(self): return self._call("observe")
    def move(self, arm, position): return self._call("move", arm=arm, position=position)
    def gripper(self, arm, opening): return self._call("gripper", arm=arm, opening=opening)
    def stop(self): return self._call("stop")

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
