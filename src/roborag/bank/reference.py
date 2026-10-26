"""Bounded whole-phase TRAIN evidence, without live perception or motor targets."""

from roborag.bank.chunks import ARMS, state_at


def phase_context(source, phase):
    """Complement a short action window with its actual temporal context.

    Selection uses recorded phase boundaries and robot motion only. It does not
    infer contact, grasp, object poses, success, or a trajectory for a new scene.
    """
    import h5py
    import numpy as np

    with h5py.File(source, "r") as data:
        total = len(data["state/left_ee_poses"])
        start, stop = int(phase["start"]), int(phase["end"])
        if not 0 <= start < stop <= total:
            raise ValueError("Phase interval outside source recording")
        frames = sorted({start, (start + stop - 1) // 2, stop - 1})
        movement = {}
        for arm in ARMS:
            poses = np.asarray(data[f"state/{arm}_ee_poses"][start:stop])
            grippers = np.asarray(data[f"action/{arm}_ee_joint_states"][start:stop, 0])
            movement[arm] = {
                "ee_path_length_m": round(
                    float(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum()), 6
                ),
                "ee_z_range_m": np.round([poses[:, 2].min(), poses[:, 2].max()], 6).tolist(),
                "gripper_command_range": np.round([grippers.min(), grippers.max()], 6).tolist(),
            }
        active = max(ARMS, key=lambda arm: movement[arm]["ee_path_length_m"])
        samples = [dict(frame=frame, recorded_state=state_at(data, frame)) for frame in frames]
        frequency = float(data["additional_info/frequency"][()])
    images = [(frame, "cam_head") for frame in frames]
    # A moving wrist view complements the head view; no claim it sees contact.
    if movement[active]["ee_path_length_m"] >= 0.005:
        images.append((frames[len(frames) // 2], f"cam_{active}_wrist"))
    return dict(
        role="HISTORICAL_PHASE_CONTEXT",
        interval=[start, stop],
        frequency=frequency,
        duration_s=(stop - start) / frequency,
        sampled_states=samples,
        measured_robot_motion=movement,
        semantics="Sparse recorded samples across this phase, not the short transition interval. "
        "EE is link6, not contact/TCP. Gripper values are commands, not proof of holding. "
        "Images and motion do not independently establish success. Adapt from current observations; "
        "do not replay historical coordinates.",
    ), images
