"""Extract local grasp windows without inventing cloth-state observations."""
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    events = []
    with h5py.File(source, "r") as f:
        frames = len(f["state/left_ee_poses"])
        for side in ("left", "right"):
            grip = f[f"action/{side}_ee_joint_states"][:, 0]
            closes = np.where((grip[1:] < .3) & (grip[:-1] >= .3))[0] + 1
            for ordinal, t in enumerate(closes):
                start, end = max(0, int(t) - 20), min(frames, int(t) + 30)
                name = f"{side}_{'sleeve' if ordinal == 0 else 'hem'}"
                pose = f[f"state/{side}_ee_poses"][int(t)].tolist()
                event = dict(target=name, side=side, close_frame=int(t), start=start,
                             end=end, quaternion_wxyz=pose[3:], close_link6_xyz=pose[:3])
                events.append(event)
        actions = {k: f[f"action/{k}"][:] for k in f["action"]}
        states = {k: f[f"state/{k}"][:] for k in f["state"]}
        np.savez_compressed(output / "demonstration.npz", **{f"action_{k}":v for k,v in actions.items()},
                            **{f"state_{k}":v for k,v in states.items()})
        images = output / "demo_frames"
        images.mkdir(exist_ok=True)
        for t in range(0, frames, 3):
            for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                raw = bytes(f[f"vision/{camera}/colors"][t])
                with Image.open(io.BytesIO(raw)) as im:
                    im.convert("RGB").resize((480, 360)).save(images / f"{camera}_{t:04d}.jpg", quality=80)
        frequency = int(f["additional_info/frequency"][()])
    manifest = dict(schema="cloth-grasp.demo.v1", source=str(source),
                    sha256=hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
                    demonstrations=1, frames=frames, frequency_hz=frequency, events=events,
                    frame_stride=3, cameras=["cam_head", "cam_left_wrist", "cam_right_wrist"],
                    provenance="Original TRAIN episode; cloth particle state is absent. No synthetic cloth labels added.")
    (output / "demo.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
