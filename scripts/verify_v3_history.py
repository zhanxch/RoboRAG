"""Replay archived v3 evidence requests through the extracted MCP tools, without motion."""

import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from mcp import Client
from roborag.episode_files import read_jsonl, write_json
from roborag.rag_mcp import EpisodeMemory, create_server


async def verify(baseline, data, bank_override=None):
    rows = []
    for directory in sorted(baseline.glob("seed1-*-l*")):
        if not directory.is_dir():
            continue
        config = json.loads((directory / "run.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        native = Path(result["native_result"])
        assert hashlib.sha256(native.read_bytes()).hexdigest() == result["native_sha256"]
        bank = Path(config["args"]["bank"])
        bank_files = {
            p: digest
            for p, digest in config["source_sha256"].items()
            if Path(p).is_relative_to(bank)
        }
        assert bank_files
        for p, digest in bank_files.items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest, p
            if bank_override is not None:
                copied = bank_override / Path(p).relative_to(bank)
                assert hashlib.sha256(copied.read_bytes()).hexdigest() == digest, copied
        calls, malformed = read_jsonl(directory / "mcp-calls.jsonl")
        assert not malformed
        counts = dict(retrieve_evidence=0, read_action_reference=0)
        with tempfile.TemporaryDirectory() as tmp:
            memory = EpisodeMemory(bank_override or bank, data, tmp, tmp, "fold_clothes", controller="delta")
            async with Client(create_server(memory)) as client:
                for call in calls:
                    name = call["tool"]
                    if name not in counts:
                        continue
                    request = call["request"]
                    if name == "retrieve_evidence":
                        request = {
                            k: v
                            for k, v in request.items()
                            if k
                            in (
                                "goal",
                                "observation",
                                "mode",
                                "arm",
                                "held_object",
                                "target_object",
                            )
                        }
                    response = await client.call_tool(name, request)
                    assert not response.is_error, (directory.name, name, response)
                    expected = call["result"]
                    if name == "retrieve_evidence":
                        assert response.content[0].text == expected["model_context"], directory.name
                    else:
                        assert json.loads(response.content[0].text) == expected, (
                            directory.name,
                            request,
                        )
                    counts[name] += 1
        rows.append(
            dict(
                case=directory.name,
                native_sha256=result["native_sha256"],
                bank_files_verified=len(bank_files),
                **counts,
            )
        )
    assert len(rows) == 50, f"Expected original full 50 cases, got {len(rows)}"
    return dict(
        schema="roborag.v3-history-replay.v1",
        source_commit="1e9ffa05682fb750bd940938b6b8bd6bf6ac091f",
        baseline=str(baseline),
        replay_bank=str(bank_override) if bank_override else "original per-episode bank",
        passed=True,
        cases=rows,
        retrieval_responses=sum(r["retrieve_evidence"] for r in rows),
        reference_responses=sum(r["read_action_reference"] for r in rows),
        scope="Model-visible evidence text replayed exactly; no model or robot movement. This does not establish new native success rate.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, help="Replay a verified byte-identical copy of the original bank")
    args = parser.parse_args()
    report = asyncio.run(verify(args.baseline.resolve(), args.data.resolve(), args.bank.resolve() if args.bank else None))
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))


if __name__ == "__main__":
    main()
