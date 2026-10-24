"""Retrieval behavior and measured command/observation alignment."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from roborag.graph_bank import make_graph, publish_bank, write_json
from roborag.graph_chunks import action_space, load_chunk
from roborag.graph_retrieval import build_index, held_object
from roborag.graph_server import BankStore, make_handler


def make_source(path, frames=40):
    with h5py.File(path, "w") as data:
        data["additional_info/frequency"] = 25.
        for arm in ("left", "right"):
            positions = np.zeros((frames, 3))
            positions[:, 0] = np.arange(frames)*.003
            rotation = Rotation.from_rotvec(np.stack([np.zeros(frames), np.zeros(frames), np.arange(frames)*.01], axis=1))
            quaternion = rotation.as_quat()[:, [3, 0, 1, 2]]
            target = np.concatenate([positions, quaternion], axis=1)
            observed = target.copy()
            observed[:, 1] -= .012  # Tracking residual: do not integrate per-step observed deltas.
            data[f"state/{arm}_ee_poses"] = observed
            data[f"action/{arm}_ee_poses"] = target
            data[f"state/{arm}_ee_joint_states"] = np.full((frames, 1), .5)
            data[f"action/{arm}_ee_joint_states"] = np.linspace(.4, .6, frames)[:, None]


def graph(name, held="白色", support="黑色", cue=None, feedback="两块保持堆叠"):
    meta = dict(id=name, task="stack_blocks", episode=0, instruction="Stack three blocks",
                path="demo.hdf5", sha256="synthetic-source", frames=40, frequency=25, cameras=["cam_head"])
    annotation = dict(title=f"{held}块放到{support}底块上", summary="两阶段示范", phases=[
        dict(title=f"移动{held}块", summary="对齐目标顶部", objects=[held, support], start_frame=0, end_frame=23,
             cue=cue or f"右爪夹着{held}块，{support}底块已经摆好。", action=f"右臂抬起{held}块，对准{support}底块上方。", feedback=feedback),
        dict(title="放置后撤离", summary="完成叠放", objects=[held, support], start_frame=23, end_frame=40,
             cue=f"{held}块已来到{support}底块上方，右爪仍环绕块体。", action="右爪松开并撤回。", feedback="夹爪移开")])
    return make_graph(meta, annotation, "test")


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        make_source(self.root / "demo.hdf5")
        for g in (graph("white-on-black"), graph("black-on-white", "黑色", "白色"), graph("another-white-on-black")):
            write_json(self.root / "graphs" / f"{g['id']}.json", g)
        publish_bank(self.root)
        self.index = build_index(self.root, self.root)
        self.store = BankStore(self.root, self.root)
        self.engine = self.store.retriever()

    def test_plan_returns_whole_chains_not_top_phase_fragments(self):
        result = self.engine.retrieve({"mode": "plan", "task": "stack_blocks", "goal": "stack blocks"})
        packet = json.loads(result["model_context"])
        self.assertTrue(packet["memories"])
        self.assertEqual(packet["purpose"], "evidence_for_gpt_planning")
        self.assertEqual(len(packet["memories"][0]["episode_trace"]), 2)
        self.assertNotIn("plan", packet["memories"][0])
        self.assertIsNone(result["results"][0]["node_id"])

    def test_roles_not_just_same_words_and_one_result_per_episode(self):
        result = self.engine.retrieve({"mode": "action", "task": "stack_blocks", "goal": "白色块放到黑色底块上",
                                      "observation": "右爪夹着白色块", "held_object": "白色块", "target_object": "深色底块", "arm": "right", "k": 5})
        ids = [r["graph_id"] for r in result["results"]]
        self.assertEqual(len(ids), 2)
        self.assertNotIn("black-on-white", ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_failure_has_no_misleading_success_reference(self):
        result = self.engine.retrieve({"task": "stack_blocks", "observation": "抓取失败，积木在桌上"})
        self.assertEqual(result["results"], [])
        self.assertTrue(result["warnings"])

    def test_natural_observation_distinguishes_already_held_from_failed_grasp(self):
        self.assertIsNone(held_object("右爪尚未抓住白色块"))
        result = self.engine.retrieve({"goal": "放到黑色底块上",
                                      "observation": "右夹爪已经抓住白色块"})
        self.assertTrue(result["results"])
        self.assertNotIn("black-on-white", [r["graph_id"] for r in result["results"]])

    def test_feedback_is_not_a_pre_action_search_key(self):
        graphs = [graph("feedback-only", cue="普通情境", feedback="unique_signal"),
                  graph("cue-match", cue="unique_signal", feedback="普通反馈")]
        for g in graphs:
            write_json(self.root / "graphs" / f"{g['id']}.json", g)
        publish_bank(self.root)
        build_index(self.root, self.root)
        result = self.store.retriever().retrieve({"observation": "unique_signal"})
        self.assertEqual([r["graph_id"] for r in result["results"]], ["cue-match"])

    def test_same_interface_state_selects_window_not_phase_start(self):
        candidate = next(c for c in self.index["chunks"] if c["graph_id"] == "white-on-black" and c["start"] == 10)
        query = {"observation": "右爪夹着白色块", "goal": "移动白色块", "held_object": "白色块",
                 "action_space": action_space(25), "state": candidate["before_state"], "arm": "right"}
        result = self.engine.retrieve(query)
        self.assertEqual(result["results"][0]["chunk"]["interval"][0], 10)
        query["action_space"] = "other-robot"
        self.assertEqual(self.engine.retrieve(query)["results"], [])
        query.pop("action_space")
        result = self.engine.retrieve(query)
        self.assertEqual(result["results"][0]["chunk"]["interval"][0], 0)
        self.assertTrue(result["warnings"])

    def test_numeric_rows_reconstruct_original_targets_and_use_real_end_observation(self):
        for chunk in self.index["chunks"]:
            phases = [n for n in self.store.graph(chunk["graph_id"])["nodes"] if n["kind"] == "phase"]
            phase = next(p for p in phases if p["id"] == chunk["node_id"])
            self.assertLessEqual(chunk["end"], phase["end_frame"])
            self.assertLess(chunk["end"], 40)
        ref = next(c for c in self.index["chunks"] if c["start"] == 5)
        transition = self.engine.chunk(ref["id"])
        with h5py.File(self.root / "demo.hdf5", "r") as data:
            for side, offset in (("left", 0), ("right", 7)):
                pose = data[f"state/{side}_ee_poses"][ref["start"]]
                xyz = pose[:3].copy()
                rotation = Rotation.from_quat(pose[[4, 5, 6, 3]])
                for t, row in enumerate(transition["actions"]):
                    xyz += row[offset:offset+3]
                    rotation = Rotation.from_rotvec(row[offset+3:offset+6]) * rotation
                    target = data[f"action/{side}_ee_poses"][ref["start"]+t]
                    np.testing.assert_allclose(xyz, target[:3], atol=1e-5)
                    error = (rotation * Rotation.from_quat(target[[4, 5, 6, 3]]).inv()).magnitude()
                    self.assertLess(error, 1e-5)
                    self.assertAlmostEqual(row[offset+6], data[f"action/{side}_ee_joint_states"][ref["start"]+t, 0], places=5)
                np.testing.assert_allclose(transition["after"]["state"][f"{side}_ee_pose"], data[f"state/{side}_ee_poses"][ref["end"]], atol=1e-6)
        self.assertIsNone(transition["task_success"])

    def test_budget_omits_whole_transitions_and_keeps_refs(self):
        result = self.engine.retrieve({"observation": "右爪夹着白色块", "include_actions": True,
                                      "k": 3, "max_context_chars": 1600})
        self.assertLessEqual(len(result["model_context"]), 1600)
        packet = json.loads(result["model_context"])
        self.assertTrue(packet["memories"])
        self.assertIn("action_reference", packet["memories"][0])
        self.assertNotIn("transition", packet["memories"][0])
        self.assertEqual(result["diagnostics"]["online_model_calls"], 0)

    def test_example_query_excludes_source_and_post_endpoint_returns_context(self):
        query = self.engine.example_query("white-on-black", "p01")
        result = self.engine.retrieve(query)
        self.assertTrue(result["results"])
        self.assertNotIn("white-on-black", [r["graph_id"] for r in result["results"]])
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            request = Request(base + "/api/retrieve", data=json.dumps(query).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request) as response:
                result = json.load(response)
            self.assertTrue(json.loads(result["model_context"])["memories"])
            with urlopen(base + result["results"][0]["chunk"]["url"]) as response:
                self.assertEqual(len(json.load(response)["actions"][0]), 14)
            request = Request(base + "/api/retrieve", data=b'{"state":{"right_ee_pose":[1]}}')
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
