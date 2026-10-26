"""Small synthetic TRAIN bank for portable protocol tests; never a benchmark corpus."""

import io
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from roborag.bank.models import make_graph, publish_bank, write_json
from roborag.bank.retrieval import build_index


def make_test_bank(root):
    root = Path(root)
    source = root / "demo.hdf5"
    frames = 40
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 128, 128)).save(buffer, format="JPEG")
    with h5py.File(source, "w") as data:
        data["additional_info/frequency"] = 25.0
        for side in ("left", "right"):
            poses = np.tile([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0], (frames, 1))
            poses[:, 0] = np.arange(frames) * 0.001
            data[f"state/{side}_ee_poses"] = poses
            data[f"action/{side}_ee_poses"] = poses
            data[f"state/{side}_ee_joint_states"] = np.ones((frames, 1))
            grips = np.ones((frames, 1))
            grips[8:28] = 0
            data[f"action/{side}_ee_joint_states"] = grips
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            data[f"vision/{camera}/colors"] = np.tile(
                np.frombuffer(buffer.getvalue(), dtype=np.uint8), (frames, 1)
            )
    for task in ("stack_blocks", "fold_clothes", "push_T"):
        meta = dict(
            id=task + "-0000",
            task=task,
            episode=0,
            instruction="堆叠三块积木 / 折叠衣服 / 推动T形块",
            path="demo.hdf5",
            sha256="synthetic-fixture",
            frames=frames,
            frequency=25,
            cameras=["cam_head", "cam_left_wrist", "cam_right_wrist"],
        )
        annotation = dict(
            title="观察后操作",
            summary="合成协议测试",
            phases=[
                dict(
                    title="抓取与移动",
                    summary="选择当前目标",
                    objects=["白色块"],
                    start_frame=0,
                    end_frame=23,
                    cue="三块不同纹理的积木分开放在桌面，夹爪空闲",
                    action="右臂接近白色块，闭合并移动",
                    feedback="物体随夹爪移动",
                ),
                dict(
                    title="释放并撤离",
                    summary="完成放置",
                    objects=["白色块"],
                    start_frame=23,
                    end_frame=40,
                    cue="右爪夹着白色块",
                    action="右爪松开并撤回",
                    feedback="物体留在桌面",
                ),
            ],
        )
        graph = make_graph(meta, annotation, "synthetic-test")
        write_json(root / "graphs" / f"{graph['id']}.json", graph)
    publish_bank(root)
    build_index(root, root)
    return root
