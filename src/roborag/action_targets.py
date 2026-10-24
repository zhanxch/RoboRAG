"""Compile GPT's explicit current-scene targets into the existing delta grammar.

This is a coordinate conversion/interpolator, not a task policy. It neither
retrieves demonstrations nor selects objects, grasps, targets or durations.
"""
from __future__ import annotations

import math

from .rag_execution import validate_delta_actions


TCP_OFFSET_M = .145


def compile_targets(state, segments):
    """Interpolate link6 translation and world rotation to each TCP endpoint.

    Each segment explicitly selects steps and optional left/right targets:
    tcp_xyz, quaternion_wxyz, gripper. Omitted fields hold the previous target.
    Each call starts from measured state; later segments use requested targets.
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
    actions = []
    for segment in segments:
        count = segment.get("steps")
        if type(count) is not int or not 1 <= count <= 24 or len(actions) + count > 24:
            raise ValueError("Target segments need integer steps; total must be 1–24")
        if not any(segment.get(s) for s in ("left", "right")):
            raise ValueError("Specify an arm target or explicit gripper hold")
        row = []
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
            delta = (end_position-position[side])/count
            rotvec = (end_rotation*rotation[side].inv()).as_rotvec()/count
            row.extend([*delta, *rotvec, grip[side]])
            position[side], rotation[side] = end_position, end_rotation
        actions.extend([row[:] for _ in range(count)])
    # Same execution limits as hand-authored rows, checked before any movement.
    validate_delta_actions(actions)
    return actions
