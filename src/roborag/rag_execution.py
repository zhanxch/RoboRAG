"""Execute GPT-authored short TCP chunks with the existing RGB robot adapter.

Only robot kinematics/table constraints are used; no task object poses are read.
The native evaluator, not this module, owns completion and success.
"""
from __future__ import annotations

import math


def validate_chunk(waypoints):
    if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 8:
        raise ValueError("Supply 1 to 8 waypoints")
    for item in waypoints:
        if item.get("arm") not in ("left", "right"):
            raise ValueError("Each waypoint needs left or right arm")
        xyz = item.get("xyz")
        if xyz is not None and (len(xyz) != 3 or any(not math.isfinite(v) for v in xyz)):
            raise ValueError("xyz must be three finite world coordinates in meters")
        if item.get("point") not in (None, "down", "forward", "down45"):
            raise ValueError("Unknown pointing preset")
        if item.get("open_axis", "x") not in ("x", "y", "z"):
            raise ValueError("Unknown finger opening axis")
        grip = item.get("gripper")
        if grip is not None and (not math.isfinite(grip) or not 0 <= grip <= 1):
            raise ValueError("gripper must be in [0,1]")
        hold = item.get("hold_steps", 0)
        if not isinstance(hold, int) or not 0 <= hold <= 25:
            raise ValueError("hold_steps must be an integer from 0 to 25")
        if item.get("home") and (xyz is not None or item.get("point")):
            raise ValueError("home cannot be combined with xyz or point")
        if (item.get("point"), item.get("open_axis", "x")) in (("down", "z"), ("forward", "y")):
            raise ValueError("Finger opening axis must be perpendicular to approach")
        if xyz is None and item.get("point") is None and grip is None and hold == 0 and not item.get("home"):
            raise ValueError("Waypoint has no action")


def execute_chunk(api, waypoints):
    from hexashell.server.core import tool_rotation
    validate_chunk(waypoints)  # reject malformed tails before any movement
    before, receipts = api.step, []
    for item in waypoints:
        if api.over:
            break
        if api.commands_used >= 60:
            api.stop("waypoint_budget")
            break
        arm = api.arm(item["arm"])
        feedback = {"requested": item, "step_before": api.step}
        api.commands_used += 1
        if item.get("home"):
            import numpy as np
            sequence = api.motion.time_path(np.stack([arm.joints(), arm.home_joints]))
            api.run({arm.tag: sequence})
            for _ in range(10):
                if api.over or np.max(np.abs(arm.joints()-arm.home_joints)) <= .01:
                    break
                api.hold(1)
            feedback["home_joint_error_rad"] = float(np.max(np.abs(arm.joints()-arm.home_joints)))
        if item.get("xyz") is not None or item.get("point"):
            target = arm.tcp().copy()
            if item.get("xyz") is not None:
                target[:3, 3] = item["xyz"]
            if item.get("point"):
                target[:3, :3] = tool_rotation(item["point"], item.get("open_axis", "x"), target[:3, :3])
            code = api.move_tcp(arm, target, feedback)
            # Do not continue a stale tail after an infeasible or poorly reached pose.
            if code not in (0, 3) or feedback.get("workspace_limited") or feedback.get("error_m", 0) > .015:
                feedback.update(stopped_tail=True, step_after=api.step)
                receipts.append(feedback)
                if api.commands_used >= 60 and not api.over:
                    api.stop("waypoint_budget")
                break
        if not api.over and item.get("gripper") is not None:
            api.set_gripper(arm, item["gripper"])
            feedback["gripper_command"] = item["gripper"]
        if not api.over and item.get("hold_steps", 0):
            api.hold(item["hold_steps"])
        feedback["step_after"] = api.step
        receipts.append(feedback)
        if api.commands_used >= 60 and not api.over:
            api.stop("waypoint_budget")
    return {"waypoints": receipts, "submitted_waypoints": len(waypoints),
            "executed_waypoints": len(receipts), "action_steps": api.step-before,
            "step": api.step, "episode_over": api.over}


def validate_delta_actions(actions):
    if not isinstance(actions, list) or not 1 <= len(actions) <= 24:
        raise ValueError("Supply 1 to 24 action rows")
    for row in actions:
        if not isinstance(row, list) or len(row) != 14 or any(not math.isfinite(v) for v in row):
            raise ValueError("Each action row needs 14 finite numbers")
        for offset in (0, 7):
            if math.sqrt(sum(v*v for v in row[offset:offset+3])) > .0350001:
                raise ValueError("Per-arm translation norm must be <=0.035 m per row")
            if math.sqrt(sum(v*v for v in row[offset+3:offset+6])) > .3500001:
                raise ValueError("Per-arm world rotvec norm must be <=0.35 rad per row")
            if not 0 <= row[offset+6] <= 1:
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


def execute_delta_actions(api, actions):
    """RoboICL world link6 delta grammar, accumulated from the observed start pose."""
    import json
    import numpy as np
    from scipy.spatial.transform import Rotation
    validate_delta_actions(actions)
    before, receipts = api.step, []
    targets = {side:api.arm(side).ee().copy() for side in ("left", "right")}
    for row in actions:
        if api.over:
            break
        command = {}
        for side, offset in (("left", 0), ("right", 7)):
            targets[side][:3, 3] += row[offset:offset+3]
            targets[side][:3, :3] = Rotation.from_rotvec(row[offset+3:offset+6]).as_matrix() @ targets[side][:3, :3]
            command[f"{side}_ee_pose"] = api.geometry.matrix_to_pose(targets[side])
            grip_key = api._env.robot_manager.process_name(api.arm(side)._robot.gripper_name)
            command[grip_key] = [row[offset+6]]
        api._env.take_action(command)
        api._cached_obs = None
        error = {}
        for side, offset in (("left", 0), ("right", 7)):
            arm = api.arm(side)
            arm.joint_target = arm.joints().copy()
            arm.gripper_target = row[offset+6]
            error[side] = float(np.linalg.norm(arm.ee()[:3, 3]-targets[side][:3, 3]))
        api._action_log.write(json.dumps({"step":api.step,"delta_row":row,"command":command,
                                         "tracking_error_m":error}) + "\n")
        api._action_log.flush()
        receipts.append({"step":api.step,"tracking_error_m":error})
        if max(error.values()) > .04:
            break
    return {"action_space":"dual-arx-x5/link6/ee_delta/world/m/rotvec-rad/25hz",
            "submitted_rows":len(actions), "executed_rows":len(receipts), "executed_waypoints":0,
            "action_steps":api.step-before, "discarded_rows":len(actions)-len(receipts),
            "last_command":command if receipts else None,
            "last_tracking":receipts[-1] if receipts else None,
            "step":api.step, "episode_over":api.over}
