"""Restore the complete frozen v3 bank, retaining all demonstration information.

Copies metadata verbatim and verifies TRAIN HDF5 sources instead of selecting
content based on which windows a previous evaluation happened to request.
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(baseline_episode, data, output):
    config = json.loads((baseline_episode / "run.json").read_text())
    bank = Path(config["args"]["bank"])
    index = json.loads((bank / "execution-index.json").read_text())
    originals = {Path(p): sha for p, sha in config["source_sha256"].items()
                 if Path(p).is_relative_to(bank)}
    assert originals and all(digest(p) == sha for p, sha in originals.items())
    metadata = sorted([*bank.glob("*.json"), *(bank / "graphs").glob("*.json")])
    assert not (bank / "manifest.json").exists(), "Expected the original full bank"
    sources = {}
    for record in index["graphs"]:
        if record["task"] != "fold_clothes":
            continue
        graph = json.loads((bank / "graphs" / (record["id"] + ".json")).read_text())
        source = (data / graph["source"]["path"]).resolve()
        assert source.is_relative_to(data.resolve())
        assert digest(source) == graph["source"]["sha256"], source
        sources[str(source)] = graph["source"]["sha256"]
    assert len(sources) == 3
    # Refuse to overwrite a frozen bank or silently merge with a previous build.
    output.mkdir(parents=True, exist_ok=False)
    copied = {}
    for original in metadata:
        target = output / original.relative_to(bank)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        assert digest(target) == digest(original)
        copied[str(target.relative_to(output))] = digest(target)
    report = dict(schema="roborag.full-v3-evidence.v1", created_at=datetime.now(timezone.utc).isoformat(),
        original_bank=str(bank), original_episode=str(baseline_episode),
        baseline_run_sha256=digest(baseline_episode / "run.json"),
        metadata_sha256=copied, train_source_sha256=sources,
        counts=dict(demonstrations=3,
            phases=sum(p["task"] == "fold_clothes" for p in index["phases"]),
            windows=sum(p["graph_id"].startswith("fold_clothes-") for p in index["chunks"])),
        scope="Complete original metadata and verified TRAIN sources; no evaluation-usage filtering. "
              "Original actions, states and camera frames remain available on demand.")
    (output / "source-inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-episode", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts/fold-v3-full")
    args = parser.parse_args()
    report = restore(args.baseline_episode.resolve(), args.data.resolve(), args.output.resolve())
    print(json.dumps(dict(output=str(args.output), counts=report["counts"],
                         verified_train_sources=len(report["train_source_sha256"]))))


if __name__ == "__main__":
    main()
