"""Serialized local process isolation for Isaac; no MCP dependency in the worker."""
import json
import os
import selectors
import signal
import subprocess
import time

from .arm import DeviceError


class WorkerProxy:
    def __init__(self, config):
        self.timeout = config.get("timeout", 240)
        self.buffer, self.failed = b"", False
        self.process = subprocess.Popen(config["command"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        env=dict(os.environ, **config.get("env", {}), ROBORAG_SERVER_PID=str(os.getpid())),
                                        start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self.joint_limits = self._receive()["joint_limits"]
        except BaseException:
            self.close()
            raise

    def _receive(self):
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError("native worker timed out; execution state unknown; no retry")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise EOFError("native worker exited; execution state unknown")
            self.buffer += chunk
            if len(self.buffer) > 16 * 1024 * 1024:
                raise ValueError("native worker response exceeds 16 MiB")
        line, self.buffer = self.buffer.split(b"\n", 1)
        result = json.loads(line)
        if not result["ok"]:
            raise DeviceError(result["error"])
        return result["value"]

    def _call(self, method, **arguments):
        if self.failed:
            raise RuntimeError("native worker unavailable after transport failure")
        try:
            self.process.stdin.write((json.dumps({"method": method, "arguments": arguments}, allow_nan=False) + "\n").encode())
            self.process.stdin.flush()
            return self._receive()
        except DeviceError:
            raise
        except BaseException:
            self.failed = True
            self.close()
            raise

    def describe(self): return self._call("describe")
    def observe(self): return self._call("observe")
    def camera_observe(self, cameras=None): return self._call("camera_observe", cameras=cameras)
    def move(self, arm, position, quaternion_wxyz=None):
        return self._call("move", arm=arm, position=position, quaternion_wxyz=quaternion_wxyz)
    def move_joints(self, arm, positions): return self._call("move_joints", arm=arm, positions=positions)
    def gripper(self, arm, opening): return self._call("gripper", arm=arm, opening=opening)
    def execution_status(self): return self._call("execution_status")
    def stop(self): return self._call("stop")

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()  # EOF lets the worker latch stop and close Isaac.
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    continue
        self.selector.close()
        self.process.stdout.close()
