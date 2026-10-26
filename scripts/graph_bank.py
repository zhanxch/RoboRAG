#!/usr/bin/env python3
"""Launch the local graph builder/browser using the project's policy runtime."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"build", "serve", "reindex"}:
        print("Usage: python3 scripts/graph_bank.py {build|serve|reindex} [options]")
        return 2
    runtime = ROOT / "local.runtime.json"
    python = json.loads(runtime.read_text()).get("policy_python", sys.executable) if runtime.exists() else sys.executable
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    commands = {"build": [python, str(ROOT / "scripts/build_graph_bank.py")],
                "reindex": [python, str(ROOT / "scripts/reindex_graph_bank.py")],
                "serve": [python, "-m", "roborag.graph_server"]}
    command = commands[sys.argv[1]]
    try:
        return subprocess.call(command + sys.argv[2:], cwd=ROOT, env=environment)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
