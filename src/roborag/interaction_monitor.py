"""Read v3's existing plan/action logs without participating in robot execution.

Labels are the model's declared subgoals, not verified physical interaction states.
Run separately: python -m roborag.interaction_monitor RUN_DIRECTORY --follow
"""

import argparse
import json
import time
from pathlib import Path


def snapshot(directory):
    """Read-only projection; never writes to the episode or agent workspace."""
    root = Path(directory)
    unreadable = []

    def read(path):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            # An observation may race an append or a process exit. Do not turn
            # missing evidence into completed motion or a successful grasp.
            unreadable.append(str(path))
            return None

    plan = read(root / "gpt-plan.json")
    progress = read(root / "progress.json")
    result = read(root / "result.json")
    failure = read(root / "exit.json") or read(root / "sim_failure.json")
    workspace = read(root / "agent-workspace.json")
    interaction = None
    if progress:
        interaction = dict(
            label=progress.get("subgoal"),
            source="progress.json",
            status="feedback_available",
            step=progress.get("step_after"),
            feedback=progress.get("result"),
        )
    if workspace and isinstance(workspace.get("path"), str):
        exchange = Path(workspace["path"])
        try:
            requests = sorted((exchange / "inbox").glob("*.json"))
        except OSError:
            unreadable.append(str(exchange / "inbox"))
            requests = []
        if requests:
            latest = requests[-1]
            request = read(latest)
            if request and any(key in request for key in ("actions", "chunk", "measure")):
                receipt = read(exchange / "outbox" / latest.name)
                interaction = dict(
                    label=request.get("subgoal"),
                    expected_feedback=request.get("expected_feedback"),
                    source=str(latest),
                    status="feedback_available" if receipt else "awaiting_feedback",
                    step=(receipt or {}).get("step_after"),
                    feedback=(receipt or {}).get("result"),
                )
    if result and result.get("complete"):
        status = "completed"
    elif failure:
        status = "interrupted"
    elif interaction:
        status = "running"
    elif plan:
        status = "planned"
    else:
        status = "waiting_for_logs"
    return dict(
        schema="roborag.interaction-monitor.v1",
        episode=str(root.resolve()),
        status=status,
        current_interaction=interaction,
        plan=plan,
        native_result=(result or {}).get("native"),
        unreadable=unreadable,
        interpretation="Labels and expected feedback are declared intent, not verified grasp, stage completion or success. No fixed stage taxonomy or plan-step matching is imposed.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    if not args.directory.is_dir():
        parser.error("episode directory does not exist")
    previous = None
    while True:
        current = snapshot(args.directory)
        encoded = json.dumps(current, ensure_ascii=False, sort_keys=True)
        if encoded != previous:
            print(encoded, flush=True)
            previous = encoded
        if not args.follow or current["status"] in ("completed", "interrupted"):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
