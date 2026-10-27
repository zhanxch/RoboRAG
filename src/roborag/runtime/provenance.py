"""Hash the actual package, prompt, bank and native controller used by a run."""

import hashlib
from pathlib import Path

from roborag.config import PROJECT_ROOT, RESOURCE_ROOT, RuntimeConfig


def protocol_sources(bank, root=PROJECT_ROOT):
    root, bank = Path(root), Path(bank)
    paths = list(RESOURCE_ROOT.parent.rglob("*.py"))
    paths += [
        p for p in (RESOURCE_ROOT.parent / "tasks").rglob("*") if p.suffix in (".txt", ".json")
    ]
    paths += [p for p in RESOURCE_ROOT.rglob("*") if p.suffix in (".txt", ".json")]
    paths += list(bank.glob("*.json")) + list((bank / "graphs").glob("*.json"))
    paths += list((root / "vendor/HexaShell/tasks/fold_clothes/tools/cloth_view").glob("*.py"))
    paths += [root / "vendor/HexaShell/tools/table_plane/tool.py"]
    paths += [root / "vendor/HexaShell/tasks/push_T/tools/planar_contact/tool.py"]
    paths += [root / "vendor/HexaShell/tasks/pour_balls_into_vase/tools/rim_geometry/tool.py"]
    native_client = RuntimeConfig.load(root).upstream / "third_party/RoboDojo/src/eval_client"
    paths += [native_client / name for name in ("ik_continuity.py", "eval_env.py")]
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
