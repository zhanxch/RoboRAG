"""Recheck selected native artifacts and replay retrieval without policy calls."""

import argparse
import hashlib
from pathlib import Path

from roborag.bank.store import BankStore
from roborag.config import PROJECT_ROOT
from roborag.evaluation.episode import summarize
from roborag.io import read_json, read_jsonl, write_json


def verify_selected(manifest, root=PROJECT_ROOT):
    root = Path(root)
    report = dict(episodes=0, artifact_hashes=0, bank_hashes=0, retrieval_queries=0, failures=[])

    def verify_file(path, expected, counter):
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("SHA256 differs")
            report[counter] += 1
        except (OSError, ValueError) as exc:
            report["failures"].append(f"{path}: {exc}")

    bank = root / manifest["bank"]["directory"]
    for relative, digest in manifest["bank"]["sha256"].items():
        verify_file(bank / relative, digest, "bank_hashes")
    if report["failures"]:
        return report
    retriever = BankStore(bank, root / "data").retriever()
    for task in manifest["tasks"].values():
        for case in task["cases"]:
            for relative, digest in case["files"].items():
                verify_file(root / relative, digest, "artifact_hashes")
            run = root / case["directory"]
            try:
                summary = summarize(run)
                expected = dict(
                    native_control_steps=case["steps"],
                    reported_usage=case["usage"],
                    wall_seconds=case["wall_seconds"],
                    complete=True,
                )
                for key, value in expected.items():
                    if summary[key] != value:
                        raise ValueError(f"{key} changed")
                native = summary["native"]
                if (native["success_rate"] == 1) != case["success"] or native["score"] != case[
                    "progress"
                ]:
                    raise ValueError("Native success/score changed")
                calls, malformed = read_jsonl(run / "mcp-calls.jsonl")
                if malformed:
                    raise ValueError("Malformed MCP log")
                for call in calls:
                    if call["tool"] == "retrieve_evidence":
                        replay = retriever.retrieve(call["request"])
                        if replay["model_context"] != call["result"]["model_context"]:
                            raise ValueError("Model-facing retrieval context changed")
                        report["retrieval_queries"] += 1
                report["episodes"] += 1
            except (OSError, ValueError, KeyError) as exc:
                report["failures"].append(f"{run}: {exc}")
    return report


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "results/selected-evidence.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_selected(read_json(args.manifest))
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["failures"] else 0)
