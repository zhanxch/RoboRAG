"""One brain -> program -> local skills -> feedback -> same brain."""
from copy import deepcopy
import json
from pathlib import Path

from .program import Program


INSTRUCTIONS = """Control the supplied device by writing a short Python program.
Return {kind: 'program', code: '...', note: '...'} or
{kind: 'finish', code: '', note: 'evidence and remaining uncertainty'}.
Use robot.describe(), robot.observe(), robot.move(arm, position),
robot.gripper(arm, opening), robot.stop() exactly as the capability description says.
Supported Python: local variables, JSON literals, indexing, numeric arithmetic,
comparisons, if/else, for over a list or range (max 100), break/continue.
Imports, functions, while, comprehensions and other Python APIs are unavailable.
At most 64 device calls and 5000 interpreter operations per program.
Read feedback before correcting an error: earlier motion is not rolled back.
Use short programs and return for visual reasoning when the scene needs inspection.
Gripper commands do not prove grasp success. Finish is your assessment, not a score.
"""


def run_episode(robot, brain, task, *, output, max_rounds=8):
    if type(max_rounds) is not int or not 1 <= max_rounds <= 80:
        raise ValueError("max_rounds must be an integer in [1, 80]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    history = []

    def log(event):
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    summary = {"status": "round_limit", "task_success": None, "rounds": 0,
               "interface": "roborag-mhs/0.1"}
    try:
        capabilities = robot.describe()
        log({"type": "start", "task": task, "capabilities": capabilities,
             "brain": getattr(brain, "metadata", {"kind": "custom"}), "rag": "off"})
        for turn in range(1, max_rounds + 1):
            summary["rounds"] = turn
            observation = robot.observe()
            context = {"instructions": INSTRUCTIONS, "task": task,
                       "capabilities": capabilities, "observation": observation,
                       "history": deepcopy(history)}
            decision = brain(context)
            if (not isinstance(decision, dict) or decision.get("kind") not in ("program", "finish")
                    or not isinstance(decision.get("code"), str)
                    or not isinstance(decision.get("note"), str)):
                raise ValueError("brain must return kind, code and note")
            log({"type": "decision", "round": turn, "observation": observation, **decision})
            if decision["kind"] == "finish":
                if decision["code"].strip():
                    raise ValueError("finish cannot contain executable code")
                summary.update(status="model_finished", note=decision["note"])
                break
            (output / f"program-{turn:03d}.py").write_text(decision["code"], encoding="utf-8")
            calls = []

            def emit(event):
                calls.append(event)
                log({"type": "device_call", "round": turn, **event})

            feedback = {"decision": decision, "status": "executed", "calls": calls}
            try:
                Program(robot, emit).execute(decision["code"])
            except Exception as exc:
                feedback.update(status="error", error=f"{type(exc).__name__}: {exc}")
            history.append(feedback)
            log({"type": "feedback", "round": turn, **feedback})
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            log({"type": "stop", "result": robot.stop()})
        except Exception as exc:
            summary.update(status="failed", stop_error=f"{type(exc).__name__}: {exc}")
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
