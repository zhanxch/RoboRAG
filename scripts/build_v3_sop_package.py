"""Extract only historical evidence actually returned in the archived v3 run.

The archive selects content, never supplies live observations, plans or scores.
TRAIN originals are used for build-time verification only. Runtime is standalone.
"""

import argparse
import base64
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from roborag.graph_server import BankStore, frame_bytes
from roborag.sop_store import SCHEMA, SOPStore

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "fold-v3-frozen-baseline/runs/v3-final-seed1-50"
TRANSITION_FIELDS = {"id", "graph_id", "node_id", "source_sha256", "interval", "frequency",
    "action_space", "duration_s", "columns", "before", "after", "measured_summary", "semantics", "task_success"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def put(mapping, key, value):
    assert key not in mapping or mapping[key] == value, (key, "conflicting historical evidence")
    mapping[key] = copy.deepcopy(value)


def image_name(graph, frame):
    return f"images/{graph}_{frame:06d}_cam_head.jpg"


def normalize_transition(transition):
    value = copy.deepcopy(transition)
    assert set(value) == TRANSITION_FIELDS and value["task_success"] is None
    for side in ("before", "after"):
        assert set(value[side]) == {"frame", "state", "images"}
        assert set(value[side]["state"]) == {f"{arm}_{field}" for arm in ("left", "right")
                                           for field in ("ee_pose", "ee_joint_state")}
        value[side]["images"] = dict(cam_head=image_name(value["graph_id"], value[side]["frame"]))
    return value


def chunk_metadata(value):
    graph, node, start, end = value["id"].split(":")
    assert [int(start), int(end)] == value["interval"]
    return dict(id=value["id"], graph_id=graph, node_id=node, start=int(start), end=int(end),
                frequency=value["frequency"], action_space=value["action_space"])


def build(runs, output, data):
    cases = sorted(p for p in runs.glob("seed1-*-l*") if p.is_dir())
    assert len(cases) == 50
    bank = BankStore(Path(read(cases[0] / "run.json")["args"]["bank"]), data)
    index = bank.retriever().index
    graphs = {r["id"]: r for r in index["graphs"] if r["task"] == "fold_clothes"}
    original_phases = {r["id"]: r for r in index["phases"]}
    overviews, texts, windows, selectors, image_bytes, references = {}, {}, {}, {}, {}, {}
    counts, logs, usage = Counter(), [], {}
    for case in cases:
        calls_path, events_path = case / "mcp-calls.jsonl", case / "agent-events.jsonl"
        references_in_case = []
        for number, call in enumerate(lines(calls_path), 1):
            if call["tool"] == "retrieve_evidence":
                mode = call["request"]["mode"]
                counts[mode+"_retrievals"] += 1
                for memory in json.loads(call["result"]["model_context"])["memories"]:
                    source = memory["source"]
                    if mode == "plan":
                        assert set(memory) == {"source", "historical_initial_scene", "episode_trace",
                                               "historical_reported_outcome", "evidence_status", "initial_image"}
                        original = graphs[source]
                        assert memory["historical_initial_scene"] == original["initial_cue"]
                        assert memory["episode_trace"] == original["episode_trace"]
                        assert memory["historical_reported_outcome"] == original["reported_outcome"]
                        overview = {k: v for k, v in memory.items() if k != "initial_image"}
                        put(overviews, source, overview)
                    else:
                        put(texts, source, dict(before=memory["historical_before"],
                            operation=memory["demonstrated_operation"], phase_feedback=memory["historical_phase_feedback"]))
                        put(references, memory["action_reference"]["id"], chunk_metadata(memory["action_reference"]))
            elif call["tool"] == "read_action_reference":
                request, packet = call["request"], call["result"]
                assert not request["include_actions"] and packet["role"] == "HISTORICAL_REFERENCE"
                counts["reference_calls"] += 1
                counts["reference_calls_with_images"] += bool(request["images"])
                source, window = packet["source"], packet["transition"]
                put(texts, source, {key: packet[key] for key in ("before", "operation", "phase_feedback")})
                put(windows, window["id"], window)
                put(selectors, (source, request["section"]), dict(window_id=window["id"],
                    window_selection=packet["window_selection"], phase_interval=packet["phase_interval"]))
                usage.setdefault(window["id"], []).append(dict(case=case.name, mcp_line=number, images=request["images"]))
                references_in_case.append(call)
        seen = 0
        for event in lines(events_path):
            item = event.get("item") or {}
            if event.get("type") != "item.completed" or item.get("tool") != "read_action_reference":
                continue
            assert item["status"] != "failed" and not item.get("error")
            expected = references_in_case[seen]
            content = item["result"]["content"]
            packet = json.loads(content[0]["text"])
            assert packet == expected["result"]
            expected_images = expected["request"]["images"]
            assert len(content) == (5 if expected_images else 1)
            if expected_images:
                t = packet["transition"]
                for offset, side in ((1, "before"), (3, "after")):
                    assert content[offset]["text"] == f"HISTORICAL {packet['source']} {side} frame {t[side]['frame']}"
                    image = content[offset+1]
                    assert image["type"] == "image" and image["mimeType"] == "image/jpeg"
                    put(image_bytes, image_name(t["graph_id"], t[side]["frame"]), base64.b64decode(image["data"], validate=True))
                    counts["image_returns"] += 1
            seen += 1
        assert seen == len(references_in_case)
        logs.append(dict(case=case.name, mcp_sha256=digest(calls_path.read_bytes()),
                         events_sha256=digest(events_path.read_bytes())))
    assert counts == dict(plan_retrievals=50, action_retrievals=3, reference_calls=241,
                          reference_calls_with_images=237, image_returns=474), counts
    assert (len(overviews), len(texts), len(windows), len(image_bytes), len(selectors)) == (3, 16, 19, 37, 19)
    sources = {}
    for graph in overviews:
        source = bank.graph(graph)["source"]
        assert "/train/" in source["path"] and source["sha256"] == digest(bank.source(bank.graph(graph)).read_bytes())
        sources[graph] = dict(split="TRAIN", sha256=source["sha256"])
    for source, text in texts.items():
        original = original_phases[source]
        assert text == dict(before=original["cue"], operation=original["action"], phase_feedback=original["feedback"])
    for window_id, window in windows.items():
        original = bank.retriever().chunk(window_id)
        original.pop("actions")
        assert window == original, window_id
        for side in ("before", "after"):
            frame = window[side]["frame"]
            name = image_name(window["graph_id"], frame)
            assert name in image_bytes
            assert image_bytes[name] == frame_bytes(str(bank.source(bank.graph(window["graph_id"]))), "cam_head", frame)
    # Runtime data below are exclusively whitelisted historical fields, not call envelopes.
    sops = []
    for graph, overview in sorted(overviews.items()):
        phases, experiences = {}, {}
        for source, text in sorted(texts.items()):
            owner, node = source.split(":")
            if owner != graph:
                continue
            intervals = [s["phase_interval"] for (key, _), s in selectors.items() if key == source]
            assert not intervals or all(v == intervals[0] for v in intervals)
            phases[node] = dict(**text, phase_interval=intervals[0] if intervals else None)
        for (source, section), selection in sorted(selectors.items()):
            owner, node = source.split(":")
            if owner == graph:
                experiences.setdefault(node, {})[section] = {k: v for k, v in selection.items() if k != "phase_interval"}
        meta = graphs[graph]
        sops.append(dict(id=graph, overview=overview, phase_texts=phases, experiences=experiences,
            retrieval_metadata={key: meta[key] for key in ("task_name", "title", "instruction")}))
    catalog = dict(task="fold_clothes", sops=sops, windows={},
                   unexpanded_window_refs=[references[key] for key in sorted(references) if key not in windows])
    output.mkdir(parents=True, exist_ok=False)
    (output / "windows").mkdir()
    (output / "images").mkdir()
    for window_id, window in sorted(windows.items()):
        name = "windows/"+window_id.replace(":", "_")+".json"
        (output / name).write_text(json.dumps(normalize_transition(window), ensure_ascii=False, indent=2)+"\n")
        catalog["windows"][window_id] = dict(**chunk_metadata(window), file=name)
    for name, raw in sorted(image_bytes.items()):
        (output / name).write_bytes(raw)
    (output / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+"\n")
    files = {str(p.relative_to(output)): digest(p.read_bytes()) for p in sorted(output.rglob("*")) if p.is_file()}
    manifest = dict(schema=SCHEMA, task="fold_clothes", files=files, demonstration_sources=sources,
        selection_basis="Historical evidence usage in a previously observed v3 evaluation cohort; development package, not independent heldout evidence.",
        counts=dict(sops=3, phase_texts=16, windows=19, images=37, unexpanded_window_refs=len(catalog["unexpanded_window_refs"])),
        excludes=["full trajectories", "raw action arrays", "unused image bytes", "evaluation observations", "GPT plans", "online robot trajectories", "native scores"],
        metadata_note="SOP titles and public task instruction are lightweight v3 retrieval keys. Unexpanded references store identifiers/intervals only, with no transition states or images.")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    SOPStore(output)  # verify the detached package without consulting the original bank
    report = dict(schema="roborag.v3-sop-extraction-audit.v1", generated_at=datetime.now(timezone.utc).isoformat(),
        passed=True, package=str(output), package_manifest_sha256=digest((output / "manifest.json").read_bytes()),
        counts=dict(counts), content_counts=manifest["counts"],
        package_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        files=len(files)+1, source_logs=logs, window_usage=usage,
        historical_sources=sources, source_bank=str(bank.directory), source_bank_modified=False,
        numerical_windows_and_image_bytes_verified_against_train_sources=True,
        builder_sha256=digest(Path(__file__).read_bytes()),
        boundary="Only historical fields were copied. No selection by native success/failure. Runtime package contains no query, current observations, online plans or rollout trajectories. TRAIN originals verified at build time only.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/fold-v3-sop")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--audit", type=Path, default=ROOT / "results/v3-sop-extraction-20261028.json")
    args = parser.parse_args()
    report = build(args.runs.resolve(), args.output.resolve(), args.data)
    args.audit.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("passed", "package", "counts", "content_counts", "package_bytes", "files")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
