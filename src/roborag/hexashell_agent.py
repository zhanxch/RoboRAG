"""One native Codex CLI process per robot episode, with its own tool loop."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from roborag.hexashell_accounting import read_events, event_accounting


# The model can inspect this transport client; frozen perception/motion tool
# implementations and all simulator state remain outside its readable roots.
CLIENT = '''#!/usr/bin/python3
import fcntl,json,os,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
with (root/".robo.lock").open("w") as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    state=json.loads((root/"obs/state.json").read_text())
    if state.get("episode_over"):
        print(json.dumps({"episode_over":True,"step":state["step"]}))
        sys.exit(3)
    name=f"{time.time_ns():020d}-{os.getpid()}.json"
    target=root/"inbox"/name
    temporary=target.with_suffix(".tmp")
    temporary.write_text(json.dumps({"argv":["robo",*sys.argv[1:]]}))
    temporary.replace(target)
    response=root/"outbox"/name
    start=time.monotonic()
    while not response.exists():
        if time.monotonic()-start>600:
            print(json.dumps({"error":"client_timeout","use":"robo status"}))
            sys.exit(4)
        time.sleep(.05)
    receipt=json.loads(response.read_text())
    print(json.dumps(receipt["result"]))
    sys.exit(receipt["exit_code"])
'''


def prepare_workspace(workspace, hexa):
    workspace, hexa = Path(workspace), Path(hexa)
    for directory in ("bin", "obs", "inbox", "outbox"):
        (workspace / directory).mkdir(parents=True, exist_ok=True)
    client = workspace / "bin/robo"
    client.write_text(CLIENT)
    client.chmod(0o755)
    python = workspace / "bin/python"
    if not python.exists():
        python.symlink_to("/usr/bin/python3")
    (workspace / "frame.png").write_bytes((hexa / "interface/frame.png").read_bytes())


def native_command(workspace, model="gpt-6-astra", effort="xhigh", output=None, python_packages=None, mcp_url=None):
    workspace = Path(workspace).resolve()
    binary = os.environ.get("COMPARE_CODEX_BIN", "/home/zxc/.local/bin/codex")
    binary_dir = str(Path(binary).resolve().parent)
    permissions = {":minimal": "read", binary_dir: "read", str(workspace): "write"}
    shell_environment = {"PATH": str(workspace / "bin") + ":/usr/local/bin:/usr/bin:/bin"}
    if python_packages:
        permissions["/opt/hexashell-python"] = "read"
        shell_environment["PYTHONPATH"] = "/opt/hexashell-python"
    permissions_toml = "{" + ", ".join(json.dumps(k) + "=" + json.dumps(v) for k, v in permissions.items()) + "}"
    environment_toml = "{" + ", ".join(k + "=" + json.dumps(v) for k, v in shell_environment.items()) + "}"
    command = [binary, "--no-daemon", "exec", "--ignore-user-config", "--ignore-rules",
               "--ephemeral", "--skip-git-repo-check", "--model", model, "--cd", str(workspace),
               "-c", f'model_reasoning_effort="{effort}"',
               "-c", 'approval_policy="never"',
               "-c", 'default_permissions="hexashell"',
               "-c", 'permissions.hexashell.filesystem=' + permissions_toml,
               "-c", 'permissions.hexashell.network.enabled=false',
               "-c", 'project_doc_max_bytes=0',
               "-c", 'skills.max_context_tokens=1',
               "-c", 'web_search="disabled"',
               "-c", 'features.respect_system_proxy=true',
               "-c", 'shell_environment_policy.set=' + environment_toml,
               "-c", 'shell_environment_policy.experimental_use_profile=false']
    # Preserve native shell, image viewing, history and compaction. Disable
    # unrelated external capabilities and delegation; this process is the robot
    # policy itself, not a coding subagent working on the repository.
    for name in ("multi_agent", "apps", "plugins", "hooks", "browser_use", "computer_use",
                 "image_generation", "memories", "goals"):
        command += ["-c", f"features.{name}=false"]
    command += ["-c", "features.code_mode=true", "-c", "features.code_mode_host=true"]
    if mcp_url:
        command += ["-c", "mcp_servers.roborag.url=" + json.dumps(mcp_url),
                    "-c", "mcp_servers.roborag.tool_timeout_sec=600",
                    "-c", 'mcp_servers.roborag.default_tools_approval_mode="approve"']
    if output:
        command += ["--output-last-message", str(workspace / "agent-final.txt")]
    command += ["--json", "-"]
    return command


def isolated_command(command, workspace, python_packages=None):
    """Clean mount namespace around the unchanged CLI and its native sandbox.

    The host's Snap nsfs mounts have non-absolute roots (mnt:[...]), rejected by
    this CLI version's mountinfo parser. A minimal outer namespace omits those
    mounts; the inner Codex permission profile still restricts shell/image IO.
    Auth stays at its existing path and is NOT readable by model tool commands.
    No project, benchmark, simulation, scoring or historical session is mounted.
    """
    binary = Path(command[0]).resolve()
    command = [str(binary), *command[1:]]
    account_dir = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    wrapper = ["/usr/bin/bwrap", "--die-with-parent", "--unshare-pid",
               "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
               "--ro-bind", "/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
               "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--dir", str(account_dir), "--ro-bind", str(binary.parent), str(binary.parent),
               "--bind", str(workspace), str(workspace)]
    if (Path(workspace) / "bin").is_dir():
        wrapper += ["--ro-bind", str(Path(workspace) / "bin"), "/usr/local/bin"]
    if python_packages:
        wrapper += ["--ro-bind", str(Path(python_packages).resolve()), "/opt/hexashell-python"]
    # Bind only the files consumed by the CLI authentication/model-catalog
    # implementation. Never read, print, copy, or put credentials into logs.
    for name in ("auth.json", "models_cache.json"):
        path = account_dir / name
        if path.is_file():
            wrapper += ["--ro-bind", str(path), str(path)]
    resolver = Path("/etc/resolv.conf").resolve()
    if str(resolver).startswith("/run/"):
        wrapper += ["--ro-bind", str(resolver.parent), str(resolver.parent)]
    return [*wrapper, "--chdir", str(workspace), "--", *command]


def run_agent(workspace, output, prompt, *, model, effort, timeout, python_packages=None, mcp_url=None):
    workspace, output = Path(workspace), Path(output)
    (workspace / "bin").mkdir(exist_ok=True)
    sandbox_executable = workspace / "bin/codex-linux-sandbox"
    if not sandbox_executable.exists():
        sandbox_executable.symlink_to(Path(os.environ.get("COMPARE_CODEX_BIN", "/home/zxc/.local/bin/codex")).resolve())
    command = isolated_command(native_command(workspace, model, effort, output, python_packages, mcp_url), workspace, python_packages)
    (output / "agent-prompt.txt").write_text(prompt)
    (output / "agent-invocation.json").write_text(json.dumps(dict(command=command, workspace=str(workspace),
        model=model, effort=effort, persistent=True, image_history="native Codex", shell_tools=True), indent=2))
    environment = {k:v for k,v in os.environ.items() if not k.startswith((
        "PYTHON", "LD_", "CUDA", "CONDA", "VIRTUAL_ENV", "ASTRA", "ROBODOJO", "ROBOICL"))}
    for key in list(environment):
        if key.startswith("CODEX_") and key not in ("CODEX_HOME",):
            environment.pop(key)
    environment["PATH"] = str(workspace / "bin") + ":/usr/local/bin:/usr/bin:/bin"
    started = time.monotonic()
    interrupted = None
    with (output / "agent-events.jsonl").open("w") as log, (output / "agent-stderr.log").open("w") as err:
        process = subprocess.Popen(command, cwd=workspace, env=environment, stdin=subprocess.PIPE,
                                   stdout=log, stderr=err, text=True, start_new_session=True)
        (output / "agent-process.json").write_text(json.dumps(dict(pid=process.pid)))
        try:
            process.communicate(prompt, timeout=timeout)
        except BaseException as exc:
            interrupted = exc
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    events, malformed = read_events(output / "agent-events.jsonl")
    final = workspace / "agent-final.txt"
    if final.is_file():
        (output / "agent-final.txt").write_bytes(final.read_bytes())
    completed = [e for e in events if e.get("type") == "turn.completed"]
    result = dict(returncode=process.returncode, wall_seconds=time.monotonic()-started,
                  completed_turns=len(completed), usage=completed[-1].get("usage") if completed else None)
    result["accounting"] = event_accounting(events, malformed)
    if interrupted:
        result["interruption"] = dict(type=type(interrupted).__name__, detail=str(interrupted))
    (output / "agent-exit.json").write_text(json.dumps(result, indent=2))
    if interrupted:
        raise interrupted
    text_log = (output / "agent-events.jsonl").read_text()
    infra_markers = ("mountinfo path is not absolute", "code-mode host is disabled", "Code Mode is unavailable", "fs sandbox helper failed", "sandbox executable is missing", "sandbox executable was missing")
    if process.returncode or not completed or any(e.get("type") == "turn.failed" for e in events) or any(m in text_log for m in infra_markers):
        raise RuntimeError("Persistent Codex agent failed; see agent-events.jsonl and agent-stderr.log")
    return result
