"""Run CaP through an actual MCP stdio connection; synthetic device by default."""
import argparse
import json
from pathlib import Path
import sys

from .brains import CodexBrain, FixtureBrain
from .harness import run_episode
from roborag.mhs.client import MCPDevice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--brain", choices=["fixture", "codex"], default="fixture")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="xhigh", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--reject-first-move", action="store_true")
    parser.add_argument("--server-config", type=Path,
                        help="Trusted JSON with command, args and optional env for a RoboRAG MHS stdio server")
    parser.add_argument("--task", default="Move the block to the target and release it.")
    args = parser.parse_args()
    if args.server_config and (args.brain == "fixture" or args.reject_first_move):
        parser.error("external server requires --brain codex; fault injection only applies to the built-in fixture")
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory already exists; choose a new run")
    if args.server_config:
        server = json.loads(args.server_config.read_text())
    else:
        server = {"command": sys.executable, "args": ["-m", "roborag.mhs.server"],
                  "env": {"PYTHONPATH": str(Path(__file__).resolve().parents[2])}}
        if args.reject_first_move:
            server["args"].append("--reject-first-move")
    brain = (FixtureBrain() if args.brain == "fixture" else
             CodexBrain(output / "model", model=args.model, effort=args.effort))
    with MCPDevice(server["command"], server.get("args", []), env=server.get("env")) as robot:
        summary = run_episode(robot, brain, args.task, output=output, max_rounds=args.max_rounds)
        if not args.server_config:
            # Fixture-only external check, never injected into the model context.
            final = robot.observe()
            summary["synthetic_check"] = (final["objects"]["block"] == final["objects"]["target"]
                                          and not final["held"])
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "model_finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
