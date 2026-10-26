"""Joint-space homing used by the delta protocol's explicit return_home command.

The receipt shape and dispatch order match the evaluated implementation. The
retired arbitrary TCP waypoint controller is no longer an execution entry point.
"""

import numpy as np


def validate_home(requests):
    if not isinstance(requests, list) or not 1 <= len(requests) <= 2:
        raise ValueError("Return home requires one or two arm requests")
    for item in requests:
        if not isinstance(item, dict) or item.get("arm") not in ("left", "right"):
            raise ValueError("Return home requires an explicit arm")
        if item != dict(arm=item["arm"], home=True, gripper=1):
            raise ValueError("Only return_home with an open gripper is supported")


def execute_home(api, requests):
    validate_home(requests)
    before, receipts = api.step, []
    for item in requests:
        if api.over:
            break
        if api.commands_used >= 60:
            api.stop("waypoint_budget")
            break
        arm = api.arm(item["arm"])
        feedback = {"requested": item, "step_before": api.step}
        api.commands_used += 1
        sequence = api.motion.time_path(np.stack([arm.joints(), arm.home_joints]))
        api.run({arm.tag: sequence})
        for _ in range(10):
            if api.over or np.max(np.abs(arm.joints() - arm.home_joints)) <= 0.01:
                break
            api.hold(1)
        feedback["home_joint_error_rad"] = float(np.max(np.abs(arm.joints() - arm.home_joints)))
        if not api.over:
            api.set_gripper(arm, item["gripper"])
            feedback["gripper_command"] = item["gripper"]
        feedback["step_after"] = api.step
        receipts.append(feedback)
        if api.commands_used >= 60 and not api.over:
            api.stop("waypoint_budget")
    return {
        "waypoints": receipts,
        "submitted_waypoints": len(requests),
        "executed_waypoints": len(receipts),
        "action_steps": api.step - before,
        "step": api.step,
        "episode_over": api.over,
    }
