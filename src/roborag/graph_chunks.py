"""Measured transitions attached to graph phases; no generated motor targets.

The action conversion follows vendor/RoboICL/roboicl/policy/reference_data.py:
first target minus observed start pose, then successive target differences.
Source poses use xyz + quaternion wxyz; output rotations are world rotvecs.
"""
from __future__ import annotations

from urllib.parse import urlencode

ARMS = ("left", "right")
ROBOT = "dual-arx-x5"


def action_space(frequency):
    return f"{ROBOT}/link6/ee_delta/world/m/rotvec-rad/{frequency:g}hz"


def state_at(data, frame):
    import numpy as np
    return {f"{arm}_{dest}": np.round(data[f"state/{arm}_{source}"][frame], 6).tolist()
            for arm in ARMS for dest, source in
            (("ee_pose", "ee_poses"), ("ee_joint_state", "ee_joint_states"))}


def tensor_at(data, start, count):
    import numpy as np
    from scipy.spatial.transform import Rotation
    previous = {arm: data[f"state/{arm}_ee_poses"][start] for arm in ARMS}
    rows = []
    for frame in range(start, start + count):
        row = []
        for arm in ARMS:
            target = data[f"action/{arm}_ee_poses"][frame]
            before = previous[arm]
            rotation = (Rotation.from_quat(target[[4, 5, 6, 3]]) *
                        Rotation.from_quat(before[[4, 5, 6, 3]]).inv()).as_rotvec()
            row.extend([*(target[:3] - before[:3]), *rotation,
                        float(data[f"action/{arm}_ee_joint_states"][frame, 0])])
            previous[arm] = target
        rows.append(row)
    return np.round(rows, 6).tolist()


def index_chunks(source, graph, *, horizon=10, stride=5):
    import h5py
    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must be positive")
    chunks = []
    with h5py.File(source, "r") as data:
        total = len(data["state/left_ee_poses"])
        frequency = float(data["additional_info/frequency"][()])
        for phase in (n for n in graph["nodes"] if n["kind"] == "phase"):
            # Reserve an actual observation at end, never invent terminal state.
            stop = min(phase["end_frame"], total - 1)
            for start in range(phase["start_frame"], stop, stride):
                end = min(start + horizon, stop)
                chunks.append({"id": f"{graph['id']}:{phase['id']}:{start}:{end}",
                               "graph_id": graph["id"], "node_id": phase["id"],
                               "start": start, "end": end, "frequency": frequency,
                               "action_space": action_space(frequency),
                               "before_state": state_at(data, start)})
    return chunks


def image_refs(graph, frame):
    return {camera: "/api/frame?" + urlencode({"id": graph["id"], "frame": frame, "camera": camera})
            for camera in graph["source"]["cameras"]}


def load_chunk(source, graph, chunk):
    import h5py
    import numpy as np
    start, end = chunk["start"], chunk["end"]
    with h5py.File(source, "r") as data:
        before, after = state_at(data, start), state_at(data, end)
        actions = tensor_at(data, start, end - start)
    summary = {}
    rows = np.asarray(actions)
    for i, arm in enumerate(ARMS):
        offset = i * 7
        summary[arm] = {
            "observed_xyz_change_m": np.round(np.asarray(after[f"{arm}_ee_pose"][:3]) -
                                              before[f"{arm}_ee_pose"][:3], 6).tolist(),
            "max_command_translation_m": round(float(np.linalg.norm(rows[:, offset:offset+3], axis=1).max()), 6),
            "gripper_command_start_end": [float(rows[0, offset+6]), float(rows[-1, offset+6])],
        }
    return {"id": chunk["id"], "graph_id": graph["id"], "node_id": chunk["node_id"],
            "source_sha256": graph["source"]["sha256"],
            "interval": [start, end], "frequency": chunk["frequency"],
            "action_space": chunk["action_space"], "duration_s": (end-start)/chunk["frequency"],
            "columns": [f"{arm}_{field}" for arm in ARMS for field in
                        ("dx_m", "dy_m", "dz_m", "rx_rad", "ry_rad", "rz_rad", "gripper_target")],
            "before": {"frame": start, "state": before, "images": image_refs(graph, start)},
            "actions": actions,
            "after": {"frame": end, "state": after, "images": image_refs(graph, end)},
            "measured_summary": summary,
            "semantics": "Recorded reference, not an executable plan for the current scene. First delta is from observed start pose; later deltas are from previous command target. Poses: xyz+wxyz; rotations: world-frame rotvec radians. Gripper state is command-side, not grasp/contact evidence.",
            "task_success": None}
