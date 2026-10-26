"""Single lazy command entry point; --help never imports the simulator."""

import argparse
import importlib
import os
import sys
from pathlib import Path

COMMANDS = {
    "tasks": ("tasks.cli", "List task configurations and selected evaluation evidence"),
    "verify-evidence": (
        "evaluation.audit",
        "Verify selected historical results and replay retrieval",
    ),
    "run": ("runtime.episode", "Run one native episode"),
    "batch": ("runtime.batch", "Run a frozen set of cases"),
    "mcp": ("mcp.server", "Serve tools for an existing episode"),
    "fetch": ("bank.fetch", "Download and verify TRAIN demonstrations"),
    "build-bank": ("bank.build", "Build a recorded experience graph bank"),
    "index-bank": ("bank.index", "Rebuild retrieval indices without model calls"),
    "browse-bank": ("bank.server", "Browse historical evidence locally"),
    "summarize": ("evaluation.episode", "Summarize one episode"),
    "evaluate": ("evaluation.suite", "Validate a declared task suite and matched comparison"),
    "diagnostics": (
        "evaluation.diagnostics",
        "Inspect recorded push_T actions without running policy",
    ),
    "compare-fold": ("evaluation.comparison", "Compare archived matched fold/HexaShell cases"),
    "doctor": ("runtime.doctor", "Check local dependencies without running a model or simulator"),
}


def main():
    parser = argparse.ArgumentParser(
        description="RoboRAG: one framework for evidence, planning and robot execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n"
        + "\n".join(f"  {name:14} {description}" for name, (_, description) in COMMANDS.items()),
    )
    parser.add_argument(
        "--root", type=Path, help="Repository containing local.runtime.json and data"
    )
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.root:
        os.environ["ROBORAG_ROOT"] = str(args.root.resolve())
    module, _ = COMMANDS[args.command]
    sys.argv = [f"roborag {args.command}", *args.arguments]
    importlib.import_module(f"roborag.{module}").main()
