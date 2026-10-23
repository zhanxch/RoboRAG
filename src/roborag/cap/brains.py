"""Replaceable model entry; no planning/repair supervisor or retrieval agent."""
from pathlib import Path


SCHEMA = {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["program", "finish"]},
    "code": {"type": "string"}, "note": {"type": "string"}},
    "required": ["kind", "code", "note"], "additionalProperties": False}


def model_content(value):
    """Present standard MCP image content as actual model image attachments."""
    if isinstance(value, list):
        return [model_content(x) for x in value]
    if isinstance(value, dict):
        if value.get("type") == "image" and value.get("mimeType") in ("image/jpeg", "image/png"):
            return {"type": "input_image", "image_url": f"data:{value['mimeType']};base64,{value['data']}"}
        return {key: model_content(item) for key, item in value.items()}
    return value


class CodexBrain:
    def __init__(self, output, *, model="gpt-6-astra", effort="xhigh", timeout=600):
        self.output = Path(output)
        self.model, self.effort, self.timeout = model, effort, timeout
        self.calls = 0
        self.metadata = {"kind": "codex-cli", "model": model, "effort": effort,
                         "provider_snapshot_available": False}

    def __call__(self, context):
        from roborag.transports.codex_cli import infer
        self.calls += 1
        result, _ = infer(model_content(context), SCHEMA, self.output / f"call-{self.calls:03d}",
                          model=self.model, effort=self.effort, timeout=self.timeout)
        return result


class FixtureBrain:
    """Scripted integration fixture. Never reported as an LLM experiment."""
    metadata = {"kind": "scripted-fixture"}

    def __call__(self, context):
        obs = context["inputs"][0]["result"]["structuredContent"]
        if obs["objects"]["block"] == obs["objects"]["target"] and not obs["held"]:
            return {"kind": "finish", "code": "", "note": "Fixture observation shows block at target, released."}
        return {"kind": "program", "note": "Observe and retry from current state, then pick and place.",
                "code": '''obs = mcp.call_tool("observe", {})["structuredContent"]
if not obs["held"]:
    mcp.call_tool("gripper", {"arm": "left", "opening": 1})
    block = obs["objects"]["block"]
    for point in [[block[0], block[1], 0.3], block]:
        mcp.call_tool("move", {"arm": "left", "position": point})
    mcp.call_tool("gripper", {"arm": "left", "opening": 0})
obs = mcp.call_tool("observe", {})["structuredContent"]
if obs["held"]:
    target = obs["objects"]["target"]
    current = obs["arms"]["left"]["position"]
    for point in [[current[0], current[1], 0.3], [target[0], target[1], 0.3], target]:
        mcp.call_tool("move", {"arm": "left", "position": point})
    mcp.call_tool("gripper", {"arm": "left", "opening": 1})
'''}
