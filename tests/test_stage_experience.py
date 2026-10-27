"""Protect established instructions, historical direction and frozen TRAIN evidence."""

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roborag.bank.store import BankStore
from roborag.experience import StageLibrary, bootstrap, load_sop, persistent_guidance
from roborag.experience.library import SUPPORTED_TASKS, _fragments
from roborag.tasks.registry import TASK_ROOT


def synthetic_bank(root):
    """Small lexical corpus, no private TRAIN files, HDF5, images or simulator.

    Keep published stage-to-phase IDs so every association is exercised. Text is
    deliberately synthetic, derived only from public SOP stage query labels.
    The real frozen-bank content/hash check is a separate local evidence record.
    """
    root.mkdir()
    (root / "graphs").mkdir()
    sops = {task: load_sop(task) for task in SUPPORTED_TASKS}
    index = {"method": "synthetic fixture lexical index", "graphs": [], "phases": [], "chunks": []}
    for task, sop in sops.items():
        by_source = {}
        for stage in sop["stages"]:
            for source in stage["experience_refs"]:
                by_source.setdefault(source, []).append(stage["query"])
        for graph_id in sop["experience_source"]["graphs"]:
            sources = sorted(source for source in by_source if source.startswith(graph_id + ":"))
            graph = {
                "id": graph_id,
                "task": task,
                "source": {
                    "path": "synthetic-no-motor-data.hdf5",
                    "sha256": "synthetic-test-only",
                    "frames": len(sources) * 20,
                    "frequency": 25,
                    "cameras": [],
                },
            }
            (root / "graphs" / f"{graph_id}.json").write_text(json.dumps(graph))
            index["graphs"].append({"id": graph_id, "graph_id": graph_id, "task": task})
            for number, source in enumerate(sources):
                node_id = source.split(":", 1)[1]
                title = " ".join(by_source[source])

                def neighbor(offset):
                    target = number + offset
                    if not 0 <= target < len(sources):
                        return None
                    other = sources[target]
                    return {"id": other.split(":", 1)[1], "title": " ".join(by_source[other])}

                phase = dict(
                    id=source,
                    graph_id=graph_id,
                    task=task,
                    task_name=task,
                    node_id=node_id,
                    title=title,
                    summary="合成测试操作，非真实示范",
                    start=number * 20,
                    end=(number + 1) * 20,
                    cue="当前物体在桌面上，左手或右手持杯在瓶口附近",
                    action=title,
                    feedback="合成观察：具体结果未知",
                    held_object=None,
                    support_object=None,
                    arms=[],
                    previous=neighbor(-1),
                    next=neighbor(1),
                )
                index["phases"].append(phase)
                index["chunks"].append(
                    dict(
                        id=source + ":c00",
                        graph_id=graph_id,
                        node_id=node_id,
                        start=phase["start"],
                        end=phase["end"],
                        frequency=25,
                        action_space="synthetic-no-action-data",
                    )
                )
    (root / "execution-index.json").write_text(json.dumps(index))
    (root / "bank.json").write_text(json.dumps({"graphs": index["graphs"]}))
    for sop in sops.values():
        sop["experience_source"]["execution_index_sha256"] = hashlib.sha256(
            (root / "execution-index.json").read_bytes()
        ).hexdigest()
        for graph_id in sop["experience_source"]["graphs"]:
            sop["experience_source"]["graphs"][graph_id] = hashlib.sha256(
                (root / "graphs" / f"{graph_id}.json").read_bytes()
            ).hexdigest()
    return sops


class StageExperienceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bank = self.root / "bank"
        self.fixture_sops = None

    def library(self, task="pour_balls_into_vase"):
        if self.fixture_sops is None:
            self.fixture_sops = synthetic_bank(self.bank)
        with patch(
            "roborag.experience.library.load_sop",
            return_value=copy.deepcopy(self.fixture_sops[task]),
        ):
            return StageLibrary(BankStore(self.bank, self.root), task)

    def test_every_existing_instruction_has_an_exact_global_or_stage_source(self):
        for task in SUPPORTED_TASKS:
            with self.subTest(task=task):
                sop = load_sop(task)
                text = (TASK_ROOT / task / "prompt.txt").read_text()
                fragments = _fragments(text)
                refs = set(sop["global_rule_refs"])
                for stage in sop["stages"]:
                    refs.update(stage["rule_refs"])
                self.assertEqual(refs, set(fragments))
                # Removing blank separators is the only permitted transformation.
                self.assertEqual(
                    [line for line in text.splitlines() if line],
                    [line for fragment in fragments.values() for line in fragment.splitlines()],
                )
                for rule in bootstrap(task)["global_rules"]:
                    self.assertEqual(rule["text"], fragments[rule["id"]])
                self.assertEqual(
                    sop["source_prompt"]["sha256"], hashlib.sha256(text.encode()).hexdigest()
                )

    def test_overview_is_bounded_and_phase_detail_is_progressive(self):
        library = self.library()
        overview = library.overview()
        self.assertNotIn("rule_refs", json.dumps(overview))
        self.assertNotIn("p_world", json.dumps(overview))
        self.assertNotIn("p_world", persistent_guidance("pour_balls_into_vase"))
        prepared = library.prepare("controlled_pour", observation="左手持杯在瓶口上方")
        self.assertIn("p_world = tcp_world + R_world @ p_local", json.dumps(prepared))
        grasp = next(stage for stage in overview["stages"] if stage["id"] == "acquire_upright_cup")
        self.assertIn("controlled_pour", grasp["lookahead"])
        self.assertIn("No lifting the block", persistent_guidance("push_T"))

    def test_stage_associations_prevent_reverse_pour_despite_conflicting_words(self):
        library = self.library()
        pouring = library.prepare(
            "controlled_pour", goal="扶正杯子 杯口朝上 恢复直立", observation="右手持杯"
        )
        sources = {example["source"] for example in pouring["historical_examples"]}
        self.assertTrue(sources)
        self.assertLessEqual(
            sources,
            {
                "pour_balls_into_vase-0000:p06",
                "pour_balls_into_vase-0001:p06",
                "pour_balls_into_vase-0002:p05",
            },
        )
        restored = library.prepare(
            "restore_place", goal="杯口下倾 倾倒小球", observation="左手持杯"
        )
        self.assertTrue(restored["historical_examples"])
        self.assertFalse(
            sources & {example["source"] for example in restored["historical_examples"]}
        )
        # The opposite direction may be inspected as temporal context, never
        # passed off as the primary stage-matched example.
        for neighbor in pouring["temporal_neighbors"]:
            self.assertIn("not a stage match", neighbor["semantics"])
            self.assertIn(neighbor["source"], pouring["allowed_refs"])

    def test_missing_recovery_keeps_empty_evidence_and_unknown(self):
        result = self.library().prepare(
            "controlled_pour", observation="杯子滑落，未抓住，发生失败", refresh=True
        )
        self.assertEqual(result["historical_examples"], [])
        self.assertEqual(result["allowed_refs"], [])
        self.assertTrue(result["warnings"])
        self.assertIn("unknown", result["use"])
        measure = self.library("fold_clothes").prepare("measure_bind")
        self.assertEqual(measure["historical_examples"], [])
        self.assertTrue(measure["rules"])

    def test_cache_requires_same_query_and_cannot_be_mutated_by_caller(self):
        library = self.library()
        first = library.prepare("controlled_pour", observation="左手 持杯")
        refs = first["allowed_refs"][:]
        self.assertFalse(first["diagnostics"]["cache_hit"])
        first["rules"].clear()
        second = library.prepare("controlled_pour", observation=" 左手   持杯 ", known_refs=refs)
        self.assertTrue(second["diagnostics"]["cache_hit"])
        self.assertFalse(second["diagnostics"]["search_performed"])
        self.assertTrue(second["rules"])
        self.assertEqual(second["diagnostics"]["known_refs_reused"], sorted(refs))
        changed = library.prepare("controlled_pour", observation="杯子抓持发生变化")
        self.assertFalse(changed["diagnostics"]["cache_hit"])
        forced = library.prepare("controlled_pour", observation="左手 持杯", refresh=True)
        self.assertFalse(forced["diagnostics"]["cache_hit"])
        self.assertTrue(forced["diagnostics"]["refresh_requested"])

    def test_sidecar_keeps_source_intervals_chunks_and_bank_unchanged(self):
        self.library()
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.bank.rglob("*.json")
        }
        for task in SUPPORTED_TASKS:
            library = self.library(task)
            sidecar = library.export_index()
            self.assertEqual(sidecar["version"], library.version)
            chunks = library.bank.retriever().chunks
            for association in sidecar["associations"]:
                start, end = association["historical_interval"]
                self.assertLess(start, end)
                self.assertTrue(association["chunk_ids"])
                for chunk_id in association["chunk_ids"]:
                    chunk = chunks[chunk_id]
                    self.assertGreaterEqual(chunk["start"], start)
                    self.assertLessEqual(chunk["end"], end)
            for stage in library.overview()["stage_ids"]:
                prepared = library.prepare(stage, observation="当前物体在桌面上")
                self.assertTrue(
                    all(
                        example["source"].startswith(task + "-")
                        and example["role"] == "recorded_experience"
                        for example in prepared["historical_examples"]
                    )
                )
        self.assertEqual(
            before,
            {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.bank.rglob("*.json")
            },
        )

    def test_cached_context_requires_the_unchanged_source_version(self):
        library = self.library()
        library.prepare("controlled_pour", observation="左手持杯")
        index = self.bank / "execution-index.json"
        index.write_text(index.read_text() + "\n")
        for operation in (library.validate_version, library.overview, library.bootstrap):
            with self.assertRaisesRegex(ValueError, "Source bank changed"):
                operation()
        with self.assertRaisesRegex(ValueError, "Source bank changed"):
            library.prepare("controlled_pour", observation="左手持杯")

    def test_prompt_and_bank_mismatch_require_a_new_version(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / "fold_clothes"
            task.mkdir()
            for name in ("prompt.txt", "sop.json"):
                shutil.copyfile(TASK_ROOT / "fold_clothes" / name, task / name)
            (task / "prompt.txt").write_text((task / "prompt.txt").read_text() + "changed")
            with patch("roborag.experience.library.TASK_ROOT", root):
                with self.assertRaisesRegex(ValueError, "publish a new SOP version"):
                    load_sop("fold_clothes")
            bank = root / "bank"
            bank.mkdir()
            (bank / "execution-index.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "declared execution-index version"):
                StageLibrary(BankStore(bank, root), "fold_clothes")


if __name__ == "__main__":
    unittest.main()
