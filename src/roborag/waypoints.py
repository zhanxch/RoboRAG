"""Bounded expansion of an entire authorized segment, without truncation.

These are geometric commands, not a collision checker or velocity controller.
The caller converts deltas to absolute targets once, then queues those targets.
"""

import math


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def expand_segment(rows, state, *, mode, max_motor_steps=64):
    """Validate a whole segment before producing any motor commands.

    Transit accepts 1..6 sparse rows and preserves both commanded gripper
    openings. Precision accepts 1..8 small rows, including explicit gripper
    changes. An over-capacity segment is rejected in full, never truncated.
    """
    if mode not in ("transit", "precision"):
        raise ValueError("control_mode must be transit or precision")
    if type(max_motor_steps) is not int or not 1 <= max_motor_steps <= 4096:
        raise ValueError("max_motor_steps must be an integer in 1..4096")
    max_rows = 6 if mode == "transit" else 8
    if not isinstance(rows, list) or not 1 <= len(rows) <= max_rows:
        raise ValueError(f"{mode} requires 1..{max_rows} action rows")
    openings = []
    for side in ("left", "right"):
        opening = (state or {}).get(f"{side}_ee_joint_state")
        if (not isinstance(opening, (list, tuple)) or len(opening) != 1
                or not _finite_number(opening[0]) or not 0 <= opening[0] <= 1):
            raise ValueError(f"Missing valid commanded {side} gripper opening")
        openings.append(float(opening[0]))

    validated = []
    count = 0
    for index, row in enumerate(rows):
        if (not isinstance(row, list) or len(row) != 14
                or not all(_finite_number(v) for v in row)):
            raise ValueError(f"actions[{index}] must contain 14 finite numbers")
        row = [float(v) for v in row]
        steps = 1
        for arm, offset in enumerate((0, 7)):
            if not 0 <= row[offset + 6] <= 1:
                raise ValueError(f"actions[{index}] gripper target must be in [0,1]")
            if mode == "transit":
                if abs(row[offset + 6] - openings[arm]) > 1e-6:
                    raise ValueError("transit must preserve commanded gripper openings")
                # Preserve the exact observation, even if the model rounded it.
                row[offset + 6] = openings[arm]
            translation = math.hypot(*row[offset:offset + 3])
            rotation = math.hypot(*row[offset + 3:offset + 6])
            if translation > (0.30 if mode == "transit" else 0.01) + 1e-8:
                raise ValueError(f"actions[{index}] exceeds {mode} translation limit")
            if rotation > (0.60 if mode == "transit" else 0.15) + 1e-8:
                raise ValueError(f"actions[{index}] exceeds {mode} rotation limit")
            steps = max(steps, math.ceil(translation / 0.03), math.ceil(rotation / 0.15))
        count += steps
        validated.append((row, steps))
    if count > max_motor_steps:
        raise ValueError(f"segment needs {count} motor steps; capacity is {max_motor_steps}")

    expanded = []
    for row, steps in validated:
        for part in range(steps):
            command = row.copy()
            for arm, offset in enumerate((0, 7)):
                command[offset:offset + 6] = [v / steps for v in row[offset:offset + 6]]
                if part < steps - 1:
                    command[offset + 6] = openings[arm]
            expanded.append(command)
        openings = [row[6], row[13]]
    return expanded
