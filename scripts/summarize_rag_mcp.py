#!/usr/bin/env python3
"""Summarize completed and interrupted MCP pilots without inventing usage."""
import argparse
import json
from pathlib import Path


import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from roborag.rag_accounting import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(schema="roborag.mcp-pilot-summary.v1", runs=[summarize(p) for p in args.runs],
        interpretation=["Development integration pilot, not a heldout success-rate study.",
            "Historical RAG data supports GPT; generated plans and live actions are separate records.",
            "MCP chunks, delta rows, waypoints and native control steps are different counts; PhysX integration substeps are not counted here.",
            "Reported token usage covers online policy calls, not offline graph construction or a per-run invoice.",
            "Cache usage is measured, but no causal improvement over a no-RAG policy is established without a controlled comparison.",
            "Interrupted attempts may consume unreported tokens; missing usage is not zero."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
