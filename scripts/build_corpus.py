#!/usr/bin/env python3
"""Build a dense training-only visual/action corpus from a verified RoboDojo demonstration."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="deposit_coin")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = json.loads((ROOT / "local.runtime.json").read_text())
    sys.path[:0] = [str(ROOT / "src"), runtime["upstream"]]
    os.environ["ROBOICL_DATA_ROOT"] = str(ROOT / "data")
    from roboicl.reference import _observation, _valid_chunk, file_sha256
    from roboicl.policy.reference_data import tensor_at
    from roboicl.policy.astra_policy import CAMERAS
    from roboicl.policy.train_reference_checks import verify_training_manifest
    from roborag.vision import VisualEncoder
    import h5py

    code = Path(runtime["upstream"])
    settings = json.loads((code / "configs/references/one_shot_j12_b12.json").read_text())
    locked = next(row for row in settings["one_shot_references"] if row["task"] == args.task)
    source = ROOT / "data" / locked["source"]["install_path"]
    digest = file_sha256(source)
    if digest != locked["source"]["sha256"]:
        raise ValueError("Training source hash differs from the published reference")
    horizon = locked["reference_horizon"]
    reference = ROOT / "data/runtime-data" / args.task / f"reference_1shot_train12_endpoint_h{horizon}"
    verification = json.loads((reference / "verification.json").read_text())
    if not verification["passed"]:
        raise ValueError("Source reference verification failed")
    full_verification = verify_training_manifest(json.loads((reference / "train_reference_bundle.json").read_text()))
    rotation = json.loads((code / "configs/tasks" / f"{args.task}.json").read_text()).get("harness_overrides", {}).get("max_rotation_rad", .15)
    output = (args.output or ROOT / "data/corpora" / args.task / "index.json").resolve()
    if output.exists():
        raise FileExistsError(output)
    encoder = VisualEncoder()
    records = []
    features = {}
    with h5py.File(source, "r") as data:
        total = len(data["state/left_ee_poses"])
        instruction = data["instruction"][()].decode("utf-8")
        starts = list(range(0, total - horizon, args.stride or horizon))
        if total - horizon - 1 not in starts:
            starts.append(total - horizon - 1)
        for start in sorted(starts):
            if not _valid_chunk(data, start, horizon, .03, rotation):
                continue
            before, after = _observation(data, start), _observation(data, start + horizon)
            for obs in (before, after):
                if obs["frame"] not in features:
                    features[obs["frame"]] = encoder.encode(obs["prompt_images"], CAMERAS).tolist()
            records.append({"record_id": f"{digest[:12]}:{start}:{start+horizon}",
                            "source_sha256": digest, "split": "train", "start": start, "end": start+horizon,
                            "before": before, "after": after, "actions": tensor_at(data, start, horizon),
                            "visual": features[start], "outcome": "unlabeled recorded demonstration transition"})
            print(f"Encoded {start}..{start+horizon} ({len(records)} records)", flush=True)
    payload = {"schema": "roborag.visual-action-corpus.v1", "task": args.task, "instruction": instruction,
               "horizon": horizon, "encoder": encoder.name, "cameras": list(CAMERAS),
               "action_space": "dual-arx-x5/link6/ee_delta/world/m/rotvec-rad/25hz",
               "source": str(source), "source_sha256": digest, "source_split": "train",
               "source_reference": locked, "head_reference": full_verification["episodes"][0]["head_reference"],
               "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"output": str(output), "records": len(records), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
