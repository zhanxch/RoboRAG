import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import Client

from roborag.rag_mcp import EpisodeMemory, create_server
from roborag.sop_store import (
    SCHEMA,
    SOPStore,
    default_bank,
    evidence_source_paths,
    open_evidence_bank,
    sop_prompt,
)


def package(root):
    """Small synthetic fixture; no original bank or archived run is needed."""
    root = Path(root)
    (root / "windows").mkdir()
    (root / "images").mkdir()
    for frame in (2, 3):
        (root / f"images/{frame}.jpg").write_bytes(b"recorded-image-bytes-"+str(frame).encode())
    transition = dict(id="demo:p01:2:3", graph_id="demo", node_id="p01", interval=[2, 3],
        before=dict(frame=2, state={}, images=dict(cam_head="images/2.jpg")),
        after=dict(frame=3, state={}, images=dict(cam_head="images/3.jpg")), measured_summary={})
    (root / "windows/demo.json").write_text(json.dumps(transition))
    catalog = dict(task="fold_clothes", sops=[dict(id="demo", overview=dict(source="demo",
        historical_initial_scene="衣服平铺", historical_reported_outcome="衣袖折回",
        episode_trace=[dict(id="p01", recorded_step="夹起袖口"), dict(id="p02", recorded_step="折回袖口")]),
        retrieval_metadata=dict(task_name="叠衣", title="折衣", instruction="Fold clothes"),
        phase_texts=dict(p01=dict(before="袖口平铺", operation="夹起袖口", phase_feedback="袖口抬起", phase_interval=[0, 5])),
        experiences=dict(p01=dict(close=dict(window_id=transition["id"], window_selection=dict(section="close")))))],
        windows={transition["id"]: dict(id=transition["id"], graph_id="demo", node_id="p01", start=2, end=3,
            frequency=25, action_space="test-delta", file="windows/demo.json")}, unexpanded_window_refs=[])
    (root / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False))
    manifest = dict(schema=SCHEMA, files={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in root.rglob("*") if p.is_file()})
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


class SOPStoreTests(unittest.TestCase):
    def test_detached_read_has_no_search_or_archive_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = package(tmp)
            store = open_evidence_bank(root, Path(tmp) / "missing-data")
            self.assertIsInstance(store, SOPStore)
            with patch.object(store, "retriever", side_effect=AssertionError("No repeat search")), \
                 patch("roborag.graph_server.BankStore.source", side_effect=AssertionError("No archive")):
                packet, images = store.reference("demo:p01", "close")
                self.assertEqual(packet["transition"]["id"], "demo:p01:2:3")
                self.assertEqual(len(images), 2)
                self.assertEqual(images[0][1], b"recorded-image-bytes-2")
                self.assertEqual(store.reference("demo:p01", "close", images=False)[1], [])

    def test_missing_sections_and_raw_actions_are_explicit_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SOPStore(package(tmp))
            for source, section in (("demo:p01", "start"), ("demo:p02", "close")):
                with self.assertRaisesRegex(ValueError, "not stored.*No archive fallback"):
                    store.reference(source, section)
            with self.assertRaisesRegex(ValueError, "Raw action arrays"):
                store.reference("demo:p01", "close", include_actions=True)

    def test_corrupted_package_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = package(tmp)
            (root / "images/2.jpg").write_bytes(b"changed")
            with patch("roborag.graph_server.BankStore", side_effect=AssertionError("No fallback")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    open_evidence_bank(root, root)

    def test_manifest_cannot_escape_the_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = package(tmp)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["files"]["../outside.json"] = "unused"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "stay inside"):
                SOPStore(root)

    def test_directory_and_source_manifest_are_standalone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = package(tmp)
            store = SOPStore(root)
            result = store.retriever().retrieve(dict(mode="plan", task="fold_clothes", goal="叠衣", observation="衣服平铺"))
            memory = json.loads(result["model_context"])["memories"][0]
            self.assertEqual(memory["historical_initial_scene"], "衣服平铺")
            self.assertEqual(memory["episode_trace"][0]["available_sections"], ["close"])
            self.assertEqual(memory["episode_trace"][1]["available_sections"], [])
            self.assertIsNone(memory["initial_image"])
            paths = evidence_source_paths(root)
            self.assertEqual(set(paths), {p for p in root.rglob("*") if p.is_file()})
            self.assertEqual(default_bank(root, "fold_clothes"), root / "artifacts/fold-v3-sop")
            self.assertEqual(default_bank(root, "stack_blocks"), root / "data/graph-bank/v1")

    def test_normalization_does_not_mutate_historical_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SOPStore(package(tmp))
            packet = dict(memories=[copy.deepcopy(store.catalog["sops"][0]["overview"])])
            original = copy.deepcopy(packet)
            store.context(packet)
            self.assertEqual(packet, original)

    def test_sop_protocol_only_changes_evidence_capability_description(self):
        text = ("Retrieve historical evidence in plan mode. Then record YOUR new plan. "
                "use include_actions=true only if exact rows are needed.")
        adapted = sop_prompt(text)
        self.assertIn("Then record YOUR new plan.", adapted)
        self.assertIn("without raw action arrays", adapted)
        self.assertNotIn("include_actions=true", adapted)


class SOPProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_keeps_plan_and_motion_contracts_and_gates_only_source_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = root / "sop"
            bank.mkdir()
            package(bank)
            legacy = EpisodeMemory(root, root, root, root, "fold_clothes", controller="delta")
            memory = EpisodeMemory(bank, root / "missing", root, root, "fold_clothes", controller="delta")
            memory.state = lambda: dict(step=0, episode_over=False)
            async with Client(create_server(legacy)) as client:
                old_tools = {t.name: t.model_dump(mode="json") for t in (await client.list_tools()).tools}
            async with Client(create_server(memory)) as client:
                new_tools = {t.name: t.model_dump(mode="json") for t in (await client.list_tools()).tools}
                for name in ("record_plan", "execute_chunk", "observe", "measure_scene"):
                    self.assertEqual(new_tools[name], old_tools[name])
                before = await client.call_tool("read_action_reference", dict(source="demo:p01", section="close"))
                self.assertTrue(before.is_error)
                retrieved = await client.call_tool("retrieve_evidence", dict(goal="叠衣", observation="衣服平铺"))
                self.assertFalse(retrieved.is_error)
                with patch.object(memory.bank, "retriever", side_effect=AssertionError("No repeat search")):
                    read = await client.call_tool("read_action_reference", dict(source="demo:p01", section="close"))
                self.assertFalse(read.is_error)
                missing = await client.call_tool("read_action_reference", dict(source="demo:p01", section="start"))
                self.assertTrue(missing.is_error)
                self.assertIn("No archive fallback", missing.content[0].text)
                arrays = await client.call_tool("read_action_reference", dict(source="demo:p01", section="close", include_actions=True))
                self.assertTrue(arrays.is_error)
                plan = await client.call_tool("record_plan", dict(current_scene="自选当前观察", adaptation="自主调整", steps=[
                    dict(subgoal="自选步骤，不按示范阶段", current_objects="当前袖口", completion_observation="查看反馈", evidence_refs=["demo"])]))
                self.assertFalse(plan.is_error)
                self.assertEqual(memory.plan["steps"][0]["subgoal"], "自选步骤，不按示范阶段")


if __name__ == "__main__":
    unittest.main()
