"""Replaceable model entry; no planning/repair supervisor or retrieval agent."""
from pathlib import Path


SCHEMA = {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["program", "finish"]},
    "code": {"type": "string"}, "note": {"type": "string"}},
    "required": ["kind", "code", "note"], "additionalProperties": False}


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
        result, _ = infer(context, SCHEMA, self.output / f"call-{self.calls:03d}",
                          model=self.model, effort=self.effort, timeout=self.timeout)
        return result


class FixtureBrain:
    """Scripted integration fixture. Never reported as an LLM experiment."""
    metadata = {"kind": "scripted-fixture"}

    def __call__(self, context):
        obs = context["observation"]
        if obs["objects"]["block"] == obs["objects"]["target"] and not obs["held"]:
            return {"kind": "finish", "code": "", "note": "Fixture observation shows block at target, released."}
        return {"kind": "program", "note": "Observe and retry from current state, then pick and place.",
                "code": '''obs = robot.observe()
if not obs["held"]:
    robot.gripper("left", 1)
    block = obs["objects"]["block"]
    for point in [[block[0], block[1], 0.3], block]:
        robot.move("left", point)
    robot.gripper("left", 0)
obs = robot.observe()
if obs["held"]:
    target = obs["objects"]["target"]
    current = obs["arms"]["left"]["position"]
    for point in [[current[0], current[1], 0.3], [target[0], target[1], 0.3], target]:
        robot.move("left", point)
    robot.gripper("left", 1)
'''}
