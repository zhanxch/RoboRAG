"""Replay v3 evidence against a detached SOP package with legacy readers blocked."""

import asyncio
import base64
import copy
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from build_v3_sop_package import BASE, ROOT, lines, normalize_transition
from mcp import Client

from roborag.rag_mcp import EpisodeMemory, create_server
from roborag.sop_store import SOPStore


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_historical_only(value):
    forbidden = {"actions", "current", "query", "observation", "record_plan", "native", "native_score", "success_rate"}
    if isinstance(value, dict):
        assert not (set(value) & forbidden), set(value) & forbidden
        for item in value.values():
            assert_historical_only(item)
    elif isinstance(value, list):
        for item in value:
            assert_historical_only(item)


async def main():
    original = ROOT / "artifacts/fold-v3-sop"
    totals, cases, action_differences = Counter(), [], []
    with tempfile.TemporaryDirectory(prefix="standalone-sop-verification-") as temporary:
        detached = Path(temporary) / "package"
        shutil.copytree(original, detached)
        blocked = AssertionError("Legacy/full-trajectory access is forbidden during SOP replay")
        with patch("h5py.File", side_effect=blocked), \
             patch("roborag.graph_server.BankStore.retriever", side_effect=blocked), \
             patch("roborag.graph_server.BankStore.graph", side_effect=blocked), \
             patch("roborag.graph_server.BankStore.source", side_effect=blocked), \
             patch("roborag.mcp.evidence.frame_bytes", side_effect=blocked), \
             patch("roborag.mcp.evidence.select_reference_window", side_effect=blocked), \
             patch("roborag.graph_retrieval.load_chunk", side_effect=blocked):
            store = SOPStore(detached)
            actual_files = {str(p.relative_to(detached)) for p in detached.rglob("*") if p.is_file()}
            assert actual_files == set(store.manifest["files"]) | {"manifest.json"}
            assert not any(p.is_symlink() for p in detached.rglob("*"))
            assert all(Path(p).suffix in (".json", ".jpg") for p in actual_files)
            for name in actual_files:
                if name.endswith(".json"):
                    assert_historical_only(json.loads((detached / name).read_text()))
            for directory in sorted(p for p in BASE.glob("seed1-*-l*") if p.is_dir()):
                case_dir = Path(temporary) / directory.name
                case_dir.mkdir()
                memory = EpisodeMemory(detached, Path(temporary) / "nonexistent-data", case_dir,
                                       case_dir, "fold_clothes", controller="delta")
                archived_references = []
                for event in lines(directory / "agent-events.jsonl"):
                    item = event.get("item") or {}
                    if event.get("type") == "item.completed" and item.get("tool") == "read_action_reference":
                        assert item["status"] != "failed"
                        archived_references.append(item["result"]["content"])
                count, reference_index = Counter(), 0
                async with Client(create_server(memory)) as client:
                    for call in lines(directory / "mcp-calls.jsonl"):
                        name = call["tool"]
                        if name not in ("retrieve_evidence", "read_action_reference"):
                            continue
                        args = {k: v for k, v in call["request"].items() if k not in ("task", "k", "max_context_chars")}
                        response = await client.call_tool(name, args)
                        assert not response.is_error, (directory.name, args, response)
                        actual = json.loads(response.content[0].text)
                        if name == "retrieve_evidence":
                            expected = store.context(json.loads(call["result"]["model_context"]))
                            mode = args["mode"]
                            if mode == "plan":
                                assert actual == expected, (directory.name, "initial SOP retrieval mismatch")
                            elif actual != expected:
                                action_differences.append(dict(case=directory.name,
                                    original_sources=[m["source"] for m in expected["memories"]],
                                    sop_sources=[m["source"] for m in actual["memories"]]))
                            count[mode+"_retrievals"] += 1
                            count[mode+"_retrievals_equivalent"] += actual == expected
                        else:
                            expected = copy.deepcopy(call["result"])
                            expected["transition"] = normalize_transition(expected["transition"])
                            assert actual == expected, (directory.name, args)
                            before = archived_references[reference_index]
                            reference_index += 1
                            assert len(response.content) == len(before)
                            for old, new in zip(before[1:], response.content[1:]):
                                assert old["type"] == new.type
                                if new.type == "text":
                                    assert old["text"] == new.text
                                else:
                                    assert old["mimeType"] == new.mime_type == "image/jpeg"
                                    raw = base64.b64decode(new.data, validate=True)
                                    assert raw == base64.b64decode(old["data"], validate=True)
                                    count["images_equal"] += 1
                                    count["image_bytes_equal"] += len(raw)
                            count["references_equivalent"] += 1
                assert reference_index == len(archived_references)
                totals.update(count)
                cases.append(dict(case=directory.name, counts=dict(count),
                    mcp_log_sha256=digest(directory / "mcp-calls.jsonl"),
                    events_sha256=digest(directory / "agent-events.jsonl")))
    assert len(cases) == 50 and totals["plan_retrievals_equivalent"] == 50
    assert totals["references_equivalent"] == 241 and totals["images_equal"] == 474
    report = dict(schema="roborag.v3-sop-replay.v1", passed=True,
        generated_at=datetime.now(timezone.utc).isoformat(), totals=dict(totals), cases=cases,
        execution_retrieval_differences=action_differences,
        acceptance_scope="50 initial retrievals and 241 explicit reference expansions. Per user instruction, the three execution-time retrieval rankings are diagnostic only and were not tuned to these queries.",
        allowed_representation_changes=["SOP label and available_sections directory added",
            "Unexpanded initial-image URL becomes null; no frame-0 image is invented",
            "Before/after image references point to packaged head JPEGs; unused wrist-image URLs removed",
            "Unexpanded action-window references explicitly marked unavailable, retaining their historical identifiers and metadata"],
        isolation="Replayed a copied package with a nonexistent data root; HDF5, legacy BankStore and trajectory/window readers were patched to fail on any access.",
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), ROOT / "src/roborag/sop_store.py", ROOT / "src/roborag/mcp/evidence.py",
            ROOT / "src/roborag/mcp/session.py", ROOT / "src/roborag/rag_runner.py", original / "manifest.json")},
        native_test=False,
        limitation="Content/reading-path equivalence only. Package selection used historical evaluation usage; no independent generalization or improved folding outcome is claimed.")
    output = ROOT / "results/v3-sop-replay-20261028.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(passed=True, totals=dict(totals), action_differences=action_differences, output=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
