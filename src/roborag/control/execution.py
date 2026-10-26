"""Validate and execute the maintained world-frame delta action grammar.

Only robot kinematics/table constraints are used; no task object poses are read.
The native evaluator, not this module, owns completion and success.
"""

from __future__ import annotations

import math

from roborag.control.backend import ActionBackend


def validate_delta_actions(actions):
    if not isinstance(actions, list) or not 1 <= len(actions) <= 24:
        raise ValueError("Supply 1 to 24 action rows")
    for row in actions:
        if not isinstance(row, list) or len(row) != 14 or any(not math.isfinite(v) for v in row):
            raise ValueError("Each action row needs 14 finite numbers")
        for offset in (0, 7):
            if math.sqrt(sum(v * v for v in row[offset : offset + 3])) > 0.0350001:
                raise ValueError("Per-arm translation norm must be <=0.035 m per row")
            if math.sqrt(sum(v * v for v in row[offset + 3 : offset + 6])) > 0.3500001:
                raise ValueError("Per-arm world rotvec norm must be <=0.35 rad per row")
            if not 0 <= row[offset + 6] <= 1:
                raise ValueError("Gripper targets must be in [0,1]")


def expand_delta_actions(actions=None, segments=None):
    """Lossless run-length encoding; fully validate before dispatching any row."""
    if actions and segments:
        raise ValueError("Choose actions or segments, not both")
    if segments:
        actions = []
        for segment in segments:
            repeat = segment.get("repeat")
            if type(repeat) is not int or not 1 <= repeat <= 24:
                raise ValueError("Segment repeat must be an integer from 1 to 24")
            if len(actions) + repeat > 24:
                raise ValueError("Total expanded rows must be <=24")
            actions.extend([segment["delta"][:] for _ in range(repeat)])
    validate_delta_actions(actions)
    return actions


def execute_delta_actions(api: ActionBackend, actions, *, rejection_errors=()):
    """RoboICL world link6 delta grammar, accumulated from the observed start pose."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    validate_delta_actions(actions)
    before, receipts = api.step, []
    rejected, last_command = None, None
    targets = {side: api.arm(side).ee().copy() for side in ("left", "right")}
    for row in actions:
        if api.over:
            break
        command = {}
        for side, offset in (("left", 0), ("right", 7)):
            targets[side][:3, 3] += row[offset : offset + 3]
            targets[side][:3, :3] = (
                Rotation.from_rotvec(row[offset + 3 : offset + 6]).as_matrix()
                @ targets[side][:3, :3]
            )
            command[f"{side}_ee_pose"] = api.geometry.matrix_to_pose(targets[side])
            grip_key = api.gripper_command_key(side)
            command[grip_key] = [row[offset + 6]]
        try:
            api.dispatch_action(command)
        except rejection_errors as exc:
            # Only opt-in, pre-dispatch rejections are recoverable. Other errors
            # still propagate as incomplete episodes. Neither gripper is updated.
            rejected = exc.event
            break
        last_command = command
        api.invalidate_observation()
        error = {}
        for side, offset in (("left", 0), ("right", 7)):
            arm = api.arm(side)
            api.update_arm_targets(side, row[offset + 6])
            error[side] = float(np.linalg.norm(arm.ee()[:3, 3] - targets[side][:3, 3]))
        api.record_action(
            {"step": api.step, "delta_row": row, "command": command, "tracking_error_m": error}
        )
        receipts.append({"step": api.step, "tracking_error_m": error})
        if max(error.values()) > 0.04:
            break
    return {
        "action_space": "dual-arx-x5/link6/ee_delta/world/m/rotvec-rad/25hz",
        "submitted_rows": len(actions),
        "executed_rows": len(receipts),
        "executed_waypoints": 0,
        "action_steps": api.step - before,
        "discarded_rows": len(actions) - len(receipts),
        "last_command": last_command,
        "last_tracking": receipts[-1] if receipts else None,
        "step": api.step,
        "episode_over": api.over,
        **({"rejection": rejected} if rejected is not None else {}),
    }
