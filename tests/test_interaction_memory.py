"""Role/phase retrieval must preserve uncertainty and source transitions."""

import json
import tempfile
import unittest
from pathlib import Path

from test_graph_retrieval import graph, make_source

from roborag.bank.interactions import query_conditions
from roborag.bank.models import publish_bank, write_json
from roborag.bank.retrieval import build_index
from roborag.bank.store import BankStore


class ConditionsTest(unittest.TestCase):
    def test_role_order_and_relative_relation(self):
        observations = [
            ("桌面左侧有红色T形积木，右侧有薄灰色T形目标垫", "left", "right"),
            ("桌面中央偏右有蓝紫色T形积木，左侧有灰色T形目标垫", "right", "left"),
            ("红色T形块位于桌面左侧，灰色T形垫位于桌面右侧", "left", "right"),
        ]
        for text, side, direction in observations:
            conditions = query_conditions(text, "")
            self.assertEqual(conditions["object_side"], side)
            self.assertEqual(conditions["target_direction"], direction)
        self.assertEqual(
            query_conditions("蓝色T形积木在灰色T形薄垫的左下方", ""), {"target_direction": "right"}
        )

    def test_never_infer_location_from_arm_heading_goal_or_ambiguity(self):
        for text in ("右臂在旁边，长杆向左", "左侧有红色T形块，右侧有蓝色T形块", "物体被遮挡"):
            self.assertNotIn("object_side", query_conditions(text, "向右移动物块"))
        self.assertNotIn("stage", query_conditions("刚完成换手", "继续移动", mode="action"))
        self.assertEqual(
            query_conditions("", "换手后再精确对齐", mode="action")["stage"], "handoff"
        )


class InteractionRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        make_source(self.root / "demo.hdf5")
        for gid, first, second in (
            ("left-to-right", "left", "right"),
            ("right-to-left", "right", "left"),
        ):
            g = graph(gid)
            g["task"] = "push_T"
            g["instruction"] = "推动T形块到目标垫"
            g["interaction"] = dict(
                object_side=first,
                target_direction=second,
                starting_arm=first,
                plan_phases=["p01", "p02"],
                preparation_limit="没有独立接触标签",
            )
            phases = [n for n in g["nodes"] if n["kind"] == "phase"]
            phases[0]["interaction"] = dict(stage="transport", active_arms=[first])
            phases[1]["interaction"] = dict(
                stage="handoff", active_arms=[first, second], handoff_to=second
            )
            for node in g["nodes"]:
                if node["kind"] == "cue":
                    node["text"] = "物块在桌上，仍有偏差。"
                if node["kind"] == "action":
                    node["text"] = "推动后改变末端位置。"
                if node["kind"] == "feedback":
                    node["text"] = "物块移动，精确接触不确定。"
            write_json(self.root / "graphs" / f"{gid}.json", g)
        publish_bank(self.root)
        build_index(self.root, self.root)
        self.retriever = BankStore(self.root, self.root).retriever()

    def test_plan_opposite_direction_and_source_linked_decisions(self):
        for side, other, expected in (("左", "右", "left-to-right"), ("右", "左", "right-to-left")):
            result = self.retriever.retrieve(
                dict(
                    mode="plan",
                    task="push_T",
                    goal="推动T形块到目标垫",
                    observation=f"{side}侧有红色T形块，{other}侧有灰色T形垫",
                )
            )
            packet = json.loads(result["model_context"])
            memory = packet["memories"][0]
            self.assertEqual(memory["source"], expected)
            card = memory["interaction_evidence"]["decisions"][0]
            self.assertEqual(card["recorded_next_choice"]["source"], expected + ":p02")
            self.assertEqual(card["interval"], [0, 23])
            self.assertIn("不确定", card["observed_after"])
            self.assertNotIn("actions", card)

    def test_handoff_matches_receiving_arm_not_standby_mentions(self):
        for arm, gid in (("left", "right-to-left"), ("right", "left-to-right")):
            result = self.retriever.retrieve(
                dict(
                    mode="action",
                    task="push_T",
                    goal="换手接手",
                    observation="物块在桌上，仍有偏差。",
                    arm=arm,
                )
            )
            self.assertEqual(
                [(r["graph_id"], r["node_id"]) for r in result["results"]], [(gid, "p02")]
            )

    def test_unknown_phase_or_failure_does_not_masquerade_as_recovery(self):
        for goal, observation in (("小幅修正残差", "物块在桌上"), ("换手", "夹持失败，物块滑落")):
            result = self.retriever.retrieve(
                dict(mode="action", task="push_T", goal=goal, observation=observation)
            )
            packet = json.loads(result["model_context"])
            self.assertEqual(packet["memories"], [])
            self.assertTrue(packet["retrieval_limits"])

    def test_small_budget_preserves_whole_cards_or_reports_omission(self):
        result = self.retriever.retrieve(dict(mode="plan", task="push_T", max_context_chars=1200))
        self.assertLessEqual(len(result["model_context"]), 1200)
        packet = json.loads(result["model_context"])
        for memory in packet["memories"]:
            self.assertEqual(len(memory["interaction_evidence"]["decisions"]), 2)


if __name__ == "__main__":
    unittest.main()
