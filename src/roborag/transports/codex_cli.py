"""Logged, tool-free Codex transport for local robot policy adaptations.

This is a transport adaptation, not an equivalent replay of the upstream API.
Role history is serialized and provider reasoning/cache items are not replayed.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def validate_cli_events(events, directory):
    """Allow only the observed startup capability notice, never execution errors."""
    capability_notice = "Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`."
    started = False
    notices, unexpected = [], []
    for event in events:
        if event.get("type") == "turn.started":
            started = True
        item = event.get("item", {})
        if (not started and event.get("type") == "item.completed"
                and item.get("type") == "error" and item.get("message") == capability_notice):
            notices.append(event)
        elif item.get("type") not in (None, "agent_message", "reasoning"):
            unexpected.append(event)
    complete = [e for e in events if e.get("type") == "turn.completed"]
    write_json(Path(directory) / "event_validation.json", {
        "disabled_capability_notices": notices, "unexpected_events": unexpected,
        "completed_turns": len(complete)})
    if not complete or unexpected or any(e.get("type") in ("turn.failed", "error") for e in events):
        raise RuntimeError(f"Codex failed or attempted an external tool; see {directory}")
    return complete[-1].get("usage", {})


def materialize_images(value, directory):
    images = []
    def visit(item):
        if isinstance(item, list):
            return [visit(x) for x in item]
        if not isinstance(item, dict):
            return item
        if item.get("type") in ("input_image", "image_url"):
            url = item["image_url"]
            if isinstance(url, dict):
                url = url["url"]
            if not url.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")):
                raise ValueError("Only supplied inline RGB images are supported")
            header, encoded = url.split(",", 1)
            raw = base64.b64decode(encoded, validate=True)
            path = directory / f"image-{len(images) + 1:03d}.{'png' if 'png' in header else 'jpg'}"
            path.write_bytes(raw)
            images.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
            return {"type": "input_text", "text": f"[Attached image {len(images)}: {path.name}]"}
        return {key: visit(val) for key, val in item.items()}
    return visit(copy.deepcopy(value)), images


def infer(payload, schema, directory, *, model="gpt-6-astra", effort="medium", timeout=600):
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    rendered, images = materialize_images(payload, directory)
    write_json(directory / "input.json", rendered)
    write_json(directory / "schema.json", schema)
    prompt = ("You are making one decision for a simulated robot. Use only the supplied "
              "instructions, conversation, observations and attached images. Do not access "
              "files, shell, web, other agents or tools. Return only the requested JSON. "
              "The harness executes the action. Image markers correspond to attachments "
              "in numbered order.\n\n" + json.dumps(rendered, ensure_ascii=False))
    (directory / "prompt.txt").write_text(prompt)
    binary = os.environ.get("COMPARE_CODEX_BIN", "/home/zxc/.local/bin/codex")
    command = [binary, "--no-daemon", "exec", "--ignore-user-config", "--ephemeral",
               "--skip-git-repo-check", "--sandbox", "read-only", "--model", model,
               "-c", f'model_reasoning_effort="{effort}"', "-c", 'web_search="disabled"']
    for name in ("shell_tool", "unified_exec", "multi_agent", "apps", "plugins", "hooks",
                 "browser_use", "computer_use", "image_generation", "memories", "view_image",
                 "code_mode", "code_mode_host", "goals"):
        command += ["-c", f"features.{name}=false"]
    for row in images:
        command += ["--image", row["path"]]
    answer = directory / "answer.json"
    command += ["--output-schema", str(directory / "schema.json"),
                "--output-last-message", str(answer), "--json", "-"]
    write_json(directory / "invocation.json", {"command": command, "images": images,
        "model": model, "effort": effort, "transport": "codex-cli-chatgpt-login",
        "provider_snapshot_available": False, "opaque_reasoning_replayed": False})
    environment = {k: v for k, v in os.environ.items() if not k.startswith((
        "PYTHON", "LD_", "CUDA", "CONDA", "VIRTUAL_ENV", "ASTRA", "ROBODOJO", "ROBOICL"))}
    started_epoch = time.time()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="robot-policy-decision-") as cwd, \
            (directory / "events.jsonl").open("w") as stdout, \
            (directory / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.PIPE,
            stdout=stdout, stderr=stderr, text=True, start_new_session=True)
        try:
            process.communicate(prompt, timeout=timeout)
        except BaseException as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            write_json(directory / "execution.json", {"returncode": process.returncode,
                "started_epoch": started_epoch, "ended_epoch": time.time(),
                "wall_seconds": time.monotonic() - started, "error": type(error).__name__})
            raise
    write_json(directory / "execution.json", {"returncode": process.returncode,
        "started_epoch": started_epoch, "ended_epoch": time.time(),
        "wall_seconds": time.monotonic() - started})
    if process.returncode or not answer.is_file():
        raise RuntimeError(f"Codex decision failed; see {directory}")
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines() if line.strip()]
    usage = validate_cli_events(events, directory)
    result = json.loads(answer.read_text())
    import jsonschema
    jsonschema.validate(result, schema)
    return result, {"input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        "input_tokens_details": {"cached_tokens": usage.get("cached_input_tokens", 0)}}
