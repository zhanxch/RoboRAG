"""Check local prerequisites without starting a model or simulator."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from roborag.config import DEFAULT_BANK, PROJECT_ROOT, RuntimeConfig


def inspect_runtime(root=PROJECT_ROOT, bank=None):
    root = Path(root)
    checks = []

    def check(name, ok, detail):
        checks.append(dict(name=name, ok=bool(ok), detail=str(detail)))

    try:
        config = RuntimeConfig.load(root)
    except (OSError, ValueError) as exc:
        check("runtime configuration", False, exc)
        return checks
    for name, path in config.as_dict().items():
        check(name, Path(path).exists(), path)
    for relative in (
        "vendor/HexaShell/hexashell/server/core.py",
        "vendor/HexaShell/tasks/fold_clothes/tools/cloth_view/tool.py",
        "vendor/HexaShell/tasks/push_T/tools/planar_contact/tool.py",
        "vendor/HexaShell/tools/table_plane/tool.py",
        "data/hexashell-agent-python",
        "data/Assets/Eval_Layout/RoboDojo/arx_x5/1",
    ):
        check(relative, (root / relative).exists(), root / relative)
    bank = Path(bank) if bank is not None else root / "data/graph-bank/current"
    for name in ("bank.json", "execution-index.json"):
        check(f"bank/{name}", (bank / name).is_file(), bank / name)
    for executable in ("bwrap", "nvidia-smi"):
        location = shutil.which(executable)
        check(executable, location is not None, location or "not on PATH")
    cli = Path(os.environ.get("COMPARE_CODEX_BIN", str(Path.home() / ".local/bin/codex")))
    check("policy CLI", cli.is_file(), cli)
    if config.policy_python.is_file():
        env = dict(os.environ, PYTHONPATH=str(root / ".runtime/mcp"))
        try:
            result = subprocess.run(
                [
                    str(config.policy_python),
                    "-c",
                    "from mcp import Client; from mcp.server import MCPServer; "
                    "import numpy, scipy, h5py, PIL, psutil; print('imports OK')",
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=20,
            )
            check(
                "policy dependencies",
                result.returncode == 0,
                result.stdout.strip() or result.stderr.strip(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            check("policy dependencies", False, exc)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    checks = inspect_runtime(bank=args.bank)
    if args.json:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
    else:
        for item in checks:
            print(f"{'OK' if item['ok'] else 'MISSING'} {item['name']}: {item['detail']}")
    raise SystemExit(0 if all(item["ok"] for item in checks) else 1)
