"""Run CaP through an actual MCP stdio connection; synthetic device by default."""
import argparse
import json
from pathlib import Path
import sys

from .brains import CodexBrain, FixtureBrain
from .harness import run_episode
from .mcp_client import MCPClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--brain", choices=["fixture", "codex"], default="fixture")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="xhigh", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--reject-first-move", action="store_true")
    parser.add_argument("--server-config", type=Path,
                        help="Trusted MCP stdio config: command, args, optional env/timeout/before_round/on_exit")
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
        server = {"command": sys.executable, "args": ["-m", "roborag.devices.synthetic_server"],
                  "env": {"PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                  "before_round": [{"tool": "observe"}], "on_exit": [{"tool": "stop"}]}
        if args.reject_first_move:
            server["args"].append("--reject-first-move")
    brain = (FixtureBrain() if args.brain == "fixture" else
             CodexBrain(output / "model", model=args.model, effort=args.effort))
    with MCPClient(server["command"], server.get("args", []), env=server.get("env"),
                   timeout=server.get("timeout", 30)) as mcp:
        summary = run_episode(mcp, brain, args.task, output=output, max_rounds=args.max_rounds,
                              before_round=server.get("before_round", []), on_exit=server.get("on_exit", []))
        if not args.server_config:
            # Fixture-only external check, never injected into the model context.
            final = mcp.call_tool("observe")["structuredContent"]
            summary["synthetic_check"] = (final["objects"]["block"] == final["objects"]["target"]
                                          and not final["held"])
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "model_finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
