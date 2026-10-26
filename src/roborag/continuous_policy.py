"""Hybrid waypoint/precision variants of the RoboDojo dialogue policies."""

import math

import numpy as np

from roboicl.policy.dialogue_policy import Model as DialogueModel

from .robot_memory import RetrievalModel


HYBRID_CONTROL_INSTRUCTIONS = """

HYBRID GPT CONTROL: PERSISTENT ROUTE + PRECISION INTERVENTION
There are two GPT action regimes. You choose between them from the latest RGB,
proprioception, and execution receipt; the local controller does not choose task
phases or grasp stages.

Keep the plan across calls. On the first call, select the arm and write a compact
route of 3-6 semantic key positions (for example clear transit point, above the
bottle, pre-grasp), plus `mode=transit; phase=...; next=...; replan_if=...` in
execution_note (maximum 400 characters). The route is your persistent high-level
plan, not a claim that a stage was completed. On later calls, preserve it and
advance only when the latest observation supports reaching the next key position.
Replan only after a tracking rejection, failed grasp/release, moved or occluded
target, obstruction, or evidence that contradicts the route.

WAYPOINT TRANSIT MODE: while the gripper is clearly in free space and the bottle
is still at a distance, return 3-6 waypoint rows toward the next key positions.
Each row is a sparse displacement of up to 0.30m; the local motor adapter
expands it into <=3 cm steps, with at most 32 motor steps dispatched per call.
Keep the same route direction across GPT calls; avoid repeated tiny waypoint
nudges and unnecessary holds. Use RGB and proprioception to decide whether the
next route point has been reached. Do not close the gripper in transit mode.

HIGH-FREQUENCY PRECISION MODE: when the arm is near the bottle (roughly within
15 cm), approaching contact, closing/opening the gripper, checking retention, or
when geometry/tracking is uncertain, switch to short 4-8 action rows with
0.5-1 cm translation steps or smaller. These pass through as individual motor
steps. Reobserve and make a fresh GPT decision after each short chunk. Confirm
grasp and placement from RGB before advancing.

Keep the other arm still unless the task requires a handover. These are GPT
planning choices; the local controller only executes the submitted action rows
at the normal control rate and applies existing bounds/guards.
""".strip()


def expand_waypoint_actions(rows, state, *, max_output_steps=32,
                            max_translation_m=0.03, max_rotation_rad=0.15):
    """Interpolate sparse EE displacement rows into bounded motor commands.

    This is geometric interpolation only. It does not select a route, task
    phase, grasp timing, or next action.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("actions must be a non-empty list of waypoint rows")
    if type(max_output_steps) is not int or max_output_steps < 1:
        raise ValueError("max_output_steps must be a positive integer")

    current_grippers = {}
    for side in ("left", "right"):
        value = np.asarray((state or {}).get(f"{side}_ee_joint_state", []), dtype=np.float64)
        current_grippers[side] = float(value[0]) if value.shape == (1,) and 0 <= value[0] <= 1 else None

    expanded = []
    for row_index, row in enumerate(rows):
        if (not isinstance(row, list) or len(row) != 14
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in row)):
            raise ValueError(f"actions[{row_index}] must contain 14 finite numbers")
        row = [float(value) for value in row]
        if any(not 0 <= row[index] <= 1 for index in (6, 13)):
            raise ValueError(f"actions[{row_index}] gripper targets must be in [0,1]")

        # Refuse implausibly large waypoint jumps before interpolation.
        for side, offset in (("left", 0), ("right", 7)):
            if np.linalg.norm(row[offset:offset + 3]) > 0.30 + 1e-8:
                raise ValueError(f"actions[{row_index}].{side} waypoint exceeds 0.30m")
            if np.linalg.norm(row[offset + 3:offset + 6]) > 0.60 + 1e-8:
                raise ValueError(f"actions[{row_index}].{side} waypoint rotation exceeds 0.60rad")

        substeps = 1
        for offset in (0, 7):
            translation_norm = float(np.linalg.norm(row[offset:offset + 3]))
            rotation_norm = float(np.linalg.norm(row[offset + 3:offset + 6]))
            substeps = max(
                substeps,
                math.ceil(translation_norm / max_translation_m),
                math.ceil(rotation_norm / max_rotation_rad),
            )

        for substep in range(substeps):
            motor_row = [0.0] * 14
            for offset in (0, 7):
                for component in range(6):
                    motor_row[offset + component] = row[offset + component] / substeps
                side = "left" if offset == 0 else "right"
                target_gripper = row[offset + 6]
                previous_gripper = current_grippers[side]
                motor_row[offset + 6] = (
                    target_gripper if substep == substeps - 1 or previous_gripper is None
                    else previous_gripper
                )
            expanded.append(motor_row)
            if len(expanded) >= max_output_steps:
                return expanded

        current_grippers["left"] = row[6]
        current_grippers["right"] = row[13]
    return expanded


class _HybridControlMixin:
    def __init__(self, cfg):
        super().__init__(cfg)
        c = self.dialogue_config
        original_step_contract = (
            f"Per step, translation norm <= {c.max_translation_m}m and "
            f"rotation norm <= {c.max_rotation_rad}rad. "
        )
        waypoint_contract = (
            "In waypoint-transit mode, each action row is a displacement to the "
            "next sparse route point (at most 0.30m and 0.60rad); the local motor "
            "adapter linearly subdivides it so each dispatched step is at most "
            "0.03m and 0.15rad. In precision mode, provide 4-8 separate small "
            "action rows, each no more than 0.01m and 0.15rad. "
        )
        if original_step_contract not in self.prompt:
            raise ValueError("Could not locate the base action-step contract")
        self.prompt = self.prompt.replace(original_step_contract, waypoint_contract, 1)
        self.prompt = self.prompt + "\n\n" + HYBRID_CONTROL_INSTRUCTIONS
        self.base_prompt = self.prompt

    def _convert_tensor(self, rows, state):
        motor_rows = expand_waypoint_actions(
            rows,
            state,
            max_output_steps=self.dialogue_config.predict_horizon,
        )
        return super()._convert_tensor(motor_rows, state)


class ContinuousDialogueModel(_HybridControlMixin, DialogueModel):
    """Continuous planning without long-term retrieval."""


class ContinuousRetrievalModel(_HybridControlMixin, RetrievalModel):
    """Continuous planning with the existing RoboRAG retrieval memory."""


# Keep the earlier profile name working while exposing the intended mixed-mode
# behavior explicitly in new runs as `--profile hybrid`.
HybridDialogueModel = ContinuousDialogueModel
HybridRetrievalModel = ContinuousRetrievalModel
