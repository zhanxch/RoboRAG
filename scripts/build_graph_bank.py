#!/usr/bin/env python3
"""Build phase-chain / guidance-tree graphs from local HDF5 demonstrations.

Use the configured policy Python (h5py, Pillow, numpy, jsonschema). Completed
graphs and Codex annotations are cached so rerunning resumes the same bank.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from roborag.graph_bank import make_graph, publish_bank, write_json


def annotation_schema():
    properties = {k: {"type": "string"} for k in ["title", "summary", "cue", "action", "feedback"]}
    properties.update({"start_frame": {"type": "integer"}, "end_frame": {"type": "integer"},
                       "objects": {"type": "array", "items": {"type": "string"}}})
    return {"type": "object", "properties": {
        "title": {"type": "string"}, "summary": {"type": "string"},
        "phases": {"type": "array", "items": {"type": "object", "properties": properties,
                                                  "required": list(properties), "additionalProperties": False}}},
        "required": ["title", "summary", "phases"], "additionalProperties": False}


def prepare_episode(source, meta, directory):
    import h5py
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as data:
        count = len(data["state/left_ee_poses"])
        frequency = float(data["additional_info/frequency"][()])
        instruction = data["instruction"][()]
        if isinstance(instruction, bytes):
            instruction = instruction.decode()
        cameras = [c for c in ("cam_head", "cam_left_wrist", "cam_right_wrist") if f"vision/{c}/colors" in data]
        grips = {arm: np.asarray(data[f"state/{arm}_ee_joint_states"]).reshape(count, -1)[:, 0]
                 for arm in ("left", "right")}
        events = []
        for arm, values in grips.items():
            scale = float(np.ptp(values))
            if scale > 1e-6:
                changes = np.flatnonzero(np.abs(np.diff(values)) > scale * .15) + 1
                spaced = []
                for frame in changes:
                    if not spaced or frame - spaced[-1] >= 4:
                        spaced.append(int(frame))
                events.extend({"frame": f, "arm": arm, "gripper_before": round(float(values[f-1]), 4),
                               "gripper_after": round(float(values[f]), 4)} for f in spaced)
        frames = set(int(x) for x in np.linspace(0, count - 1, 24))
        for event in events[:16]:
            frames.update([max(0, event["frame"] - 3), min(count - 1, event["frame"] + 4)])
        frames = sorted(frames)
        if len(frames) > 40:
            frames = [frames[int(i)] for i in np.linspace(0, len(frames)-1, 40)]
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
        except OSError:
            font = ImageFont.load_default()
        attachments = []
        def sheet(items, name):
            width, height, columns = 320, 240, 4
            board = Image.new("RGB", (width * columns, (height + 28) * ((len(items) + columns - 1) // columns)), "#f1f3f1")
            draw = ImageDraw.Draw(board)
            for i, (camera, frame) in enumerate(items):
                image = Image.open(io.BytesIO(bytes(data[f"vision/{camera}/colors"][frame]))).convert("RGB")
                image.thumbnail((width, height))
                x, y = (i % columns) * width, (i // columns) * (height + 28)
                board.paste(image, (x + (width-image.width)//2, y))
                draw.text((x+8, y+height+3), f"{camera.replace('cam_', '')} | f={frame} | {frame/frequency:.2f}s", font=font, fill="#17221b")
            path = directory / name
            board.save(path, quality=85)
            attachments.append(path)
        for start in range(0, len(frames), 16):
            sheet([("cam_head", f) for f in frames[start:start+16]], f"head-{start//16+1}.jpg")
        wrist_frames = sorted(set([min(count-1, e["frame"]+4) for e in events] +
                                  [int(i) for i in np.linspace(0,count-1,6)]))
        if len(wrist_frames) > 8:
            wrist_frames = [wrist_frames[int(i)] for i in np.linspace(0,len(wrist_frames)-1,8)]
        wrist_items = [(camera, f) for f in wrist_frames for camera in cameras if camera != "cam_head"]
        if wrist_items:
            sheet(wrist_items, "wrists.jpg")
        states = []
        for f in frames:
            state = {"frame": f}
            for arm in ("left", "right"):
                state[arm] = {"xyz": np.round(data[f"state/{arm}_ee_poses"][f][:3], 3).tolist(),
                              "gripper": round(float(grips[arm][f]), 4)}
            states.append(state)
        meta.update(frames=count, frequency=frequency, instruction=str(instruction), cameras=cameras)
    write_json(directory / "metadata.json", meta)
    return meta, attachments, {"gripper_change_candidates": events, "sampled_states": states}


def annotate(meta, attachments, measurements, directory, model, effort):
    # Reuse the existing tool-free multimodal transport and its signed-in CLI.
    from roborag.transports.codex_cli import infer
    prompt = f"""这是离线机器人示范整理任务，不是控制机器人。请从完整示范的接触表和状态摘要构建自然语言交互阶段图。
任务：{meta['instruction']}；总帧数 {meta['frames']}；频率 {meta['frequency']} Hz。
图片按 frame 标记，可以跨接触表连续查看。把连续轨迹分成约 5–12 个有意义的阶段，复杂任务允许更多；不要按每个相机采样机械拆分。
参考 TG-RAG 的 chain-of-trees：阶段顺序构成主链，每个阶段的 cue/action/feedback 构成本地指导树。
不要求固定阶段类别。title 用简短中文自然语言描述具体阶段；summary 简述局部目标；objects 填具体对象或角色。
cue 描述此阶段开始时看得到的状态与操作线索；action 描述示范实际采取的操作，可保留对后续执行有用的相对关系；feedback 描述片段末尾可见变化，遮挡处写不确定。
不要虚构恢复分支、隐藏接触、精确物体位姿或未出现的失败。夹爪指令不等于抓持成功。
start_frame 包含，end_frame 不包含，范围 0..{meta['frames']}；按时间排序、无重叠且无空隙，完整覆盖示范。第一段从0开始，后一段start_frame等于前一段end_frame，最后一段end_frame必须等于{meta['frames']}；边界是估计值。
整图 title 描述这条示范的策略，summary 用一两句概括。只返回 schema JSON。
本体状态中的 gripper 是数据原始值，变化只能用于边界候选；结合画面识别开闭与抓持。
测量摘要：{json.dumps(measurements, ensure_ascii=False)}"""
    payload = [{"role": "user", "content": [{"type": "input_text", "text": prompt},
        *[{"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()}
          for p in attachments]]}]
    result, usage = infer(payload, annotation_schema(), directory, model=model, effort=effort, timeout=600)
    return result, usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/graph-bank/v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--require-full-coverage", action="store_true",
                        help="Reject phase annotations with unindexed gaps in a new full bank")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    seen = set()
    for path in args.manifest:
        manifest = json.loads(path.read_text())
        for row in manifest["records"]:
            if row["path"] in seen:
                continue
            seen.add(row["path"])
            entries.append((manifest["task"], row))
    if args.limit:
        entries = entries[:args.limit]
    for index, (task, row) in enumerate(entries):
        graph_id = f"{task}-{row['episode']:04d}"
        destination = root / "graphs" / f"{graph_id}.json"
        if destination.exists():
            print(f"[{index+1}/{len(entries)}] cached {graph_id}", flush=True)
            continue
        source = ROOT / "data" / row["path"]
        if row.get("split") != "train":
            raise ValueError("Use training demonstrations for this bank")
        with source.open("rb") as stream:
            actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_sha256 != row["sha256"]:
            raise ValueError(f"TRAIN source differs from manifest: {source}")
        meta = {"id": graph_id, "task": task, "episode": row["episode"], "path": row["path"],
                "sha256": actual_sha256}
        job = root / "build" / graph_id
        print(f"[{index+1}/{len(entries)}] preparing {graph_id}", flush=True)
        meta, images, measurements = prepare_episode(source, meta, job)
        if args.prepare_only:
            continue
        if (job / "annotation.json").exists():
            annotation = json.loads((job / "annotation.json").read_text())
        else:
            attempt = 1
            while (job / f"codex-{attempt}").exists():
                attempt += 1
            print(f"[{index+1}/{len(entries)}] Codex extracting {graph_id}", flush=True)
            annotation, usage = annotate(meta, images, measurements, job / f"codex-{attempt}", args.model, args.effort)
            write_json(job / "annotation.json", annotation)
            write_json(job / "usage.json", usage)
        graph = make_graph(meta, annotation, args.model)
        if args.require_full_coverage:
            phases = annotation["phases"]
            if (phases[0]["start_frame"] != 0 or phases[-1]["end_frame"] != meta["frames"] or
                    any(a["end_frame"] != b["start_frame"] for a, b in zip(phases, phases[1:]))):
                raise ValueError(f"Incomplete phase coverage in {job / 'annotation.json'}")
        write_json(destination, graph)
        bank = publish_bank(root)
        print(f"[{index+1}/{len(entries)}] saved {graph_id}: {sum(n['kind']=='phase' for n in graph['nodes'])} phases; bank={bank['counts']}", flush=True)
    if not args.prepare_only:
        print(json.dumps(publish_bank(root)["counts"], ensure_ascii=False), flush=True)
        from roborag.graph_retrieval import build_index
        index = build_index(root, ROOT / "data")
        print(json.dumps({"retrieval": index["counts"]}), flush=True)


if __name__ == "__main__":
    main()
