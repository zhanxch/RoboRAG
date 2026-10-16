#!/usr/bin/env python3
"""Build a multi-episode, training-only bank from pinned download manifests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runtime = json.loads((ROOT / "local.runtime.json").read_text())
    sys.path[:0] = [str(ROOT / "src"), runtime["upstream"]]
    os.environ["ROBOICL_DATA_ROOT"] = str(ROOT / "data")
    from roboicl.paths import configure_imports
    configure_imports()
    from roboicl.reference import _observation, _valid_chunk, file_sha256
    from roboicl.policy.reference_data import tensor_at
    from roboicl.policy.astra_policy import CAMERAS
    from roboicl.policy.train_reference_checks import source_cameras
    from roborag.vision import VisualEncoder
    import h5py
    import numpy as np

    manifests = [json.loads(p.read_text()) for p in args.manifest]
    tasks = {m["task"] for m in manifests}
    if len(tasks) != 1:
        raise ValueError("Each pilot bank must have one task")
    task = tasks.pop()
    code = Path(runtime["upstream"])
    horizon = json.loads((code / "configs/adaptive_horizons.json").read_text())["task_horizons"][task]
    rotation = json.loads((code / "configs/tasks" / f"{task}.json").read_text()).get("harness_overrides", {}).get("max_rotation_rad", .15)
    encoder = VisualEncoder()
    episodes, records, seen = [], [], set()
    for manifest in manifests:
        for entry in manifest["records"]:
            if entry["split"] != "train":
                raise ValueError("Evaluation episodes cannot enter the training bank")
            source = ROOT / "data" / entry["path"]
            digest = file_sha256(source)
            if digest != entry["sha256"]:
                raise ValueError(f"Source integrity failed: {source}")
            if digest in seen:
                continue
            seen.add(digest)
            with h5py.File(source, "r") as data:
                if float(data["additional_info/frequency"][()]) != 25:
                    raise ValueError("Training data frequency differs from action interface")
                total = len(data["state/left_ee_poses"])
                initial = _observation(data, 0)
                instruction = data["instruction"][()].decode("utf-8")
                episode = {"source_sha256": digest, "source": entry["path"], "split": "train",
                           "instruction": instruction, "frames": total,
                           "provenance": {"repository": manifest["repository"], "revision": manifest["revision"]},
                           "head_reference": source_cameras(data, 0, np.zeros(3))["cam_head"],
                           "initial": initial, "scene": encoder.scene(initial["head_image"]).tolist(),
                           "outcome": "published demonstration; success not independently replay-verified"}
                starts = set(range(0, total - horizon, args.stride or horizon))
                if total > horizon:
                    starts.add(total - horizon - 1)
                features, count = {}, 0
                for start in sorted(starts):
                    if not _valid_chunk(data, start, horizon, .03, rotation):
                        continue
                    before, after = _observation(data, start), _observation(data, start + horizon)
                    if start not in features:
                        features[start] = encoder.encode(before["prompt_images"], CAMERAS).tolist()
                    records.append({"record_id": f"{digest[:12]}:{start}:{start+horizon}",
                                    "source_sha256": digest, "split": "train", "start": start, "end": start+horizon,
                                    "before": before, "after": after, "actions": tensor_at(data, start, horizon),
                                    "visual": features[start], "outcome": "unlabeled recorded demonstration transition"})
                    count += 1
                if not count:
                    raise ValueError(f"Source has no valid chunks: {source}")
                episode["chunk_count"] = count
                episodes.append(episode)
                print(f"Encoded {source.name}: {total} frames, {count} chunks", flush=True)
    payload = {"schema": "roborag.visual-action-corpus.v2", "task": task, "horizon": horizon,
               "encoder": encoder.name, "scene_encoder": encoder.scene_name, "cameras": list(CAMERAS),
               "action_space": "dual-arx-x5/link6/ee_delta/world/m/rotvec-rad/25hz", "source_split": "train",
               "episodes": episodes, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"output": str(args.output), "episodes": len(episodes), "records": len(records),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
