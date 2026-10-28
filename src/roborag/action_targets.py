"""Compile GPT's explicit current-scene targets into the existing delta grammar.

This is a coordinate conversion/interpolator, not a task policy. It neither
retrieves demonstrations nor selects objects, grasps or targets. Durations are
only extended when necessary to satisfy the existing per-row motion limits.
"""
from __future__ import annotations

import math

from .rag_execution import validate_delta_actions


TCP_OFFSET_M = .145


def compile_targets(state, segments):
    """Return delta rows, extending too-short target segments as necessary."""
    actions, _ = compile_targets_with_timing(state, segments)
    return actions


def compile_targets_with_timing(state, segments):
    """Interpolate link6 translation and world rotation to each TCP endpoint.

    Each segment explicitly selects steps and optional left/right targets:
    tcp_xyz, quaternion_wxyz, gripper. Omitted fields hold the previous target.
    Each call starts from measured state; later segments use requested targets.
    Both arms share max(requested steps, minimum steps for either arm). Return
    rows and per-segment timing; compiled steps do not imply execution completed.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    if not isinstance(segments, list) or not segments:
        raise ValueError("Supply at least one target segment")
    position, rotation, grip = {}, {}, {}
    for side in ("left", "right"):
        pose = state[f"{side}_ee_pose"]
        position[side] = np.asarray(pose[:3], dtype=float)
        rotation[side] = Rotation.from_quat(np.asarray(pose[3:])[[1,2,3,0]])
        grip[side] = state[f"{side}_ee_joint_state"][0]
    actions, timing = [], []
    for index, segment in enumerate(segments):
        requested = segment.get("steps")
        if type(requested) is not int or requested < 1:
            raise ValueError("Target segment steps must be positive integers")
        # Empty arm targets (or a segment containing only steps) explicitly
        # retain the previous targets and grippers, matching omitted fields.
        motion, minimum = [], 1
        for side in ("left", "right"):
            target = segment.get(side) or {}
            end_rotation = rotation[side]
            quaternion = target.get("quaternion_wxyz")
            if quaternion is not None:
                if len(quaternion) != 4 or not all(math.isfinite(x) for x in quaternion) or np.linalg.norm(quaternion) < 1e-9:
                    raise ValueError("quaternion_wxyz requires a finite nonzero quaternion")
                end_rotation = Rotation.from_quat(np.asarray(quaternion)[[1,2,3,0]])
            tcp = target.get("tcp_xyz")
            if tcp is None:
                tcp = position[side] + rotation[side].apply([TCP_OFFSET_M,0,0])
            if len(tcp) != 3 or not all(math.isfinite(x) for x in tcp):
                raise ValueError("tcp_xyz requires three finite metres")
            end_position = np.asarray(tcp) - end_rotation.apply([TCP_OFFSET_M,0,0])
            value = target.get("gripper")
            if value is not None:
                grip[side] = value
            delta = end_position-position[side]
            rotvec = (end_rotation*rotation[side].inv()).as_rotvec()
            minimum = max(minimum, math.ceil(np.linalg.norm(delta)/.0350001),
                          math.ceil(np.linalg.norm(rotvec)/.3500001))
            motion.append((delta, rotvec, grip[side]))
            position[side], rotation[side] = end_position, end_rotation
        count = max(requested, minimum)
        row = []
        for delta, rotvec, gripper in motion:
            row.extend([*(delta/count), *(rotvec/count), gripper])
        actions.extend([row[:] for _ in range(count)])
        timing.append(dict(segment=index+1, requested_steps=requested,
                           minimum_steps=minimum, compiled_steps=count))
    # Same execution limits as hand-authored rows, checked before any movement.
    validate_delta_actions(actions)
    return actions, timing
