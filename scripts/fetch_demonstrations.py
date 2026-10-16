#!/usr/bin/env python3
"""Fetch selected published demonstration files at a fixed revision with LFS SHA256 checks."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="deposit_coin")
    parser.add_argument("--episodes", default="0,1,2,9")
    args = parser.parse_args()
    settings = json.loads((ROOT / "vendor/RoboICL/configs/references/one_shot_j12_b12.json").read_text())
    repo = settings["source_repository"]["repo_id"]
    if not (ROOT / "vendor/RoboICL/configs/tasks" / f"{args.task}.json").is_file():
        raise ValueError("Unknown task")
    row = next((r for r in settings["one_shot_references"] if r["task"] == args.task), None)
    revision = row["source"].get("repository_revision", settings["source_repository"]["revision"]) if row else settings["source_repository"]["revision"]
    directory = row["source"]["repository_path"].rsplit("/", 1)[0] if row else f"data/RoboDojo/{args.task}/arx_x5/data"
    files = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{directory}?limit=1000", timeout=30))
    wanted = {f"episode_{int(i):07d}.hdf5" for i in args.episodes.split(",")}
    selected = [r for r in files if Path(r["path"]).name in wanted]
    if len(selected) != len(wanted):
        raise ValueError("Requested demonstration missing from fixed-revision file listing")
    destination = ROOT / "data/runtime-data" / args.task / "train"
    destination.mkdir(parents=True, exist_ok=True)
    def fetch(item):
        target = destination / Path(item["path"]).name
        expected = item["lfs"]["oid"]
        if not target.exists():
            partial = target.with_suffix(".hdf5.partial")
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{item['path']}"
            subprocess.run(["curl", "-fL", "--retry", "4", "--connect-timeout", "30", "--max-time", "600",
                            "-C", "-", "-o", str(partial), url], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if partial.stat().st_size != item["size"] or sha(partial) != expected:
                raise ValueError(f"Download integrity mismatch: {partial}")
            partial.replace(target)
        if target.stat().st_size != item["size"] or sha(target) != expected:
            raise ValueError(f"Existing file differs from source: {target}")
        print("Verified",target.name,flush=True)
        return {"episode": int(target.stem.split("_")[-1]), "path": str(target.relative_to(ROOT / "data")),
                "repository_path": item["path"], "sha256": expected, "bytes": item["size"], "split": "train"}
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, selected))
    manifest = destination / ("sources-" + "-".join(str(r["episode"]) for r in records) + ".json")
    manifest.write_text(json.dumps({"task": args.task, "repository": repo, "revision": revision,
                                   "records": records}, indent=2) + "\n")
    print(manifest)


if __name__ == "__main__":
    main()
