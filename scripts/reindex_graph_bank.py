#!/usr/bin/env python3
"""Rebuild plan/phase/action indices locally, with no model calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from roborag.graph_retrieval import build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=ROOT / "data/graph-bank/v1")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    index = build_index(args.bank, args.data_root, horizon=args.horizon, stride=args.stride)
    print(json.dumps({"counts": index["counts"], "horizon": index["horizon"], "stride": index["stride"]}))


if __name__ == "__main__":
    main()
