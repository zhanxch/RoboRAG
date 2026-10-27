"""Execution contracts: stable plans, live evidence and unchanged robot grammar."""

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from mcp import Client
from PIL import Image

from roborag.execution.models import GraphPhase
from roborag.io import write_json
from roborag.mcp.server import create_server
from roborag.mcp.session import EpisodeSession


class FixtureLibrary:
    """No benchmark data: focus these tests on the episode state machine."""

    def __init__(self):
        self.calls = []

    def overview(self):
        return {
            "version": {"id": "fixture-v1"},
            "stages": [{"id": "fold_sleeve"}, {"id": "release_inspect_home"}],
        }

    def bootstrap(self):
        return self.overview()

    def export_index(self):
        return {"version": {"id": "fixture-v1"}}

    def validate_version(self):
        return None

    def prepare(self, stage, **kwargs):
        self.calls.append((stage, kwargs))
        return {
            "sop_stage": stage,
            "version_id": "fixture-v1",
            "rules": [{"text": "Use the current visible cloth"}],
            "allowed_refs": ["demo:phase-1"],
            "historical_examples": [{"source": "demo:phase-1"}],
            "diagnostics": {"search_performed": True, "cache_hit": False},
        }


def phase(stage_id="fold", depends_on=(), sop_stage="fold_sleeve"):
    return GraphPhase(
        id=stage_id,
        sop_stage=sop_stage,
        goal="Fold the current sleeve inward",
        object_bindings={"cloth": "visible striped sleeve"},
        depends_on=list(depends_on),
        expected_effects=["sleeve rests inward"],
        experience=[
            dict(
                source="demo:phase-1",
                adopted="lift before inward motion",
                adaptation="use current measured sleeve position",
            )
        ],
    ).model_dump()


class StagedExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        (self.path / "obs").mkdir()
        for camera in ("head", "wrist_l", "wrist_r"):
            Image.new("RGB", (8, 8)).save(self.path / "obs" / f"{camera}.png")
        self.state = {"step": 0, "episode_over": False}
        self.memory = EpisodeSession(
            self.path, self.path, self.path, self.path, "fold_clothes", architecture="staged-v1"
        )
        self.memory.state = lambda: copy.deepcopy(self.state)
        self.execution = self.memory.execution
        self.library = FixtureLibrary()
        self.execution._library = self.library
        self.execution.observation(self.state, ["head"])

    def plan(self, phases=None):
        self.execution.bootstrap()
        self.execution.prepare("fold_sleeve", observation="当前衣袖", goal="折衣袖")
        return self.execution.record_plan(
            "one visible cloth",
            {"cloth": "striped cloth"},
            phases or [phase()],
            "current geometry",
            "fixture-v1",
        )

    def check(self, evidence=None, status="met"):
        return {
            "effect": "sleeve rests inward",
            "status": status,
            "evidence_ids": evidence if evidence is not None else [self.execution.last_observation],
            "observation": "Current head RGB shows the sleeve resting across the body",
        }

    def finish(self, stage_id):
        self.execution.assess(stage_id, [self.check()], "complete", "visible condition", [])

    def test_prepare_then_enter_reuses_context_and_does_not_plan_again(self):
        self.plan([phase(), phase("other_sleeve")])
        calls = len(self.library.calls)
        result = self.execution.enter("fold", "different current scene wording", "entry")
        self.assertTrue(result["context"]["session_reuse"])
        self.assertEqual(len(self.library.calls), calls)
        self.finish("fold")
        self.execution.enter("other_sleeve", "current opposite sleeve", "entry")
        self.assertEqual(len(self.library.calls), calls)
        self.assertEqual(self.execution.plan["version"], 1)
        self.execution.prepare("release_inspect_home", observation="preview")
        self.assertEqual(self.execution.active, "other_sleeve")
        self.assertEqual(self.execution.plan["version"], 1)
        with self.assertRaisesRegex(ValueError, "persistent plan"):
            self.execution.record_plan(
                "scene", {"cloth": "cloth"}, [phase()], "again", "fixture-v1"
            )

    def test_graph_dependencies_and_evidence_do_not_become_historical_facts(self):
        self.execution.bootstrap()
        self.execution.prepare("fold_sleeve")
        with self.assertRaisesRegex(ValueError, "acyclic"):
            self.execution.record_plan(
                "scene",
                {"cloth": "cloth"},
                [phase("a", ["b"]), phase("b", ["a"])],
                "",
                "fixture-v1",
            )
        self.plan([phase(), phase("home", ["fold"], "release_inspect_home")])
        with self.assertRaisesRegex(ValueError, "dependencies"):
            self.execution.enter("home", "", "entry")
        self.execution.enter("fold", "", "entry")
        for evidence in (["demo:phase-1"], ["action-00001"], []):
            with self.assertRaises(ValueError):
                self.execution.assess("fold", [self.check(evidence)], "complete", "claim", [])
        with self.assertRaisesRegex(ValueError, "every expected effect"):
            self.execution.assess("fold", [self.check([], "unknown")], "complete", "occluded", [])
        result = self.execution.assess(
            "fold", [self.check([], "unknown")], "continue", "occluded", []
        )
        self.assertEqual(result["execution"]["status"], "active")
        self.finish("fold")
        self.assertIn("unknown", self.execution.status()["task_success"])

    async def test_mcp_actions_keep_same_schema_and_bind_receipt_to_active_stage(self):
        self.plan()
        async with Client(create_server(self.memory)) as client:
            tool = next(t for t in (await client.list_tools()).tools if t.name == "execute_chunk")
            legacy = EpisodeSession(self.path, self.path, self.path, self.path, "fold_clothes")
            async with Client(create_server(legacy)) as old_client:
                old_tool = next(
                    t for t in (await old_client.list_tools()).tools if t.name == "execute_chunk"
                )
            self.assertEqual(tool.model_dump(), old_tool.model_dump())
            row = [0, 0, 0, 0, 0, 0, 1, 0.01, 0, 0, 0, 0, 0, 1]
            arguments = dict(
                subgoal="fold",
                expected_feedback="sleeve follows",
                segments=[dict(delta=row, repeat=2)],
            )
            rejected = await client.call_tool("execute_chunk", arguments)
            self.assertTrue(rejected.is_error)
            await client.call_tool("enter_stage", {"stage_id": "fold"})
            old_observation = self.execution.last_observation

            async def submit(request):
                self.state["step"] += len(request["actions"])
                return {
                    "step_before": 0,
                    "step_after": self.state["step"],
                    "result": {
                        "step": self.state["step"],
                        "executed_rows": 2,
                        "discarded_rows": 0,
                        "episode_over": False,
                    },
                }

            self.memory.submit = AsyncMock(side_effect=submit)
            result = await client.call_tool("execute_chunk", arguments)
            self.assertFalse(result.is_error)
            packet = json.loads(result.content[0].text)
            self.assertEqual(packet["stage_execution"]["stage_id"], "fold")
            request = self.memory.submit.call_args.args[0]
            self.assertEqual(request["actions"], [row, row])
            self.assertEqual(request["execution_context"]["plan_version"], 1)
            self.assertEqual(self.execution.last_action["observation_id"], packet["observation_id"])
            self.assertEqual(self.execution.plan["version"], 1)
            with self.assertRaisesRegex(ValueError, "predates"):
                self.execution.assess(
                    "fold", [self.check([old_observation])], "complete", "old", []
                )
            result = await client.call_tool(
                "assess_stage",
                {
                    "stage_id": "fold",
                    "checks": [self.check()],
                    "decision": "complete",
                    "rationale": "current visible position",
                    "experience": [
                        {
                            "source": "demo:phase-1",
                            "adopted": "raise before inward motion",
                            "adaptation": "smaller lift after observed tracking error",
                        }
                    ],
                },
            )
            self.assertFalse(result.is_error)
            self.assertEqual(self.execution.plan["version"], 1)
            self.assertEqual(len(self.execution.progress["fold"]["experience_adaptations"]), 1)
            saved = json.loads((self.path / "episode-context.json").read_text())
            self.assertEqual(saved["last_action"]["id"], "action-00001")
            ledger = [
                json.loads(line)
                for line in (self.path / "episode-ledger.jsonl").read_text().splitlines()
            ]
            action = next(e for e in ledger if e["kind"] == "action_received")
            self.assertEqual(action["action"]["stage_id"], "fold")

    async def test_real_exchange_hook_exposes_measurement_evidence_without_planning(self):
        (self.path / "inbox").mkdir()
        (self.path / "outbox").mkdir()

        async def backend():
            for _ in range(1000):
                pending = list((self.path / "inbox").glob("*.json"))
                if pending:
                    request = json.loads(pending[0].read_text())
                    self.assertIn("measure", request)
                    write_json(
                        self.path / "outbox" / pending[0].name,
                        {
                            "step_before": 0,
                            "step_after": 0,
                            "result": {
                                "plan_ok": True,
                                "measured_at_step": 0,
                                "geometry": {"plane_z": 0.7},
                            },
                        },
                    )
                    return
                await asyncio.sleep(0.001)
            self.fail("measurement never reached the exchange")

        responder = asyncio.create_task(backend())
        async with Client(create_server(self.memory)) as client:
            result = await client.call_tool("measure_scene", {})
            self.assertFalse(result.is_error)
            packet = json.loads(result.content[0].text)
            self.assertEqual(self.execution.evidence[packet["evidence_id"]]["kind"], "measurement")
            self.assertEqual(packet["observation_id"], self.execution.last_observation)
            self.assertIsNone(self.execution.plan)
        await responder

    async def test_failed_assessment_and_transition_block_dispatch_until_valid_entry(self):
        self.plan([phase(), phase("home", ["fold"], "release_inspect_home")])
        receipt = {"result": {"plan_ok": True, "measured_at_step": 0}}
        self.execution.measurement({"measure": {}}, receipt, self.state)
        old_measurement = receipt["result"]["evidence_id"]
        self.state["step"] = 1
        current = self.execution.observation(self.state, ["head"])
        self.execution.enter("fold", "current sleeve visible", "entry")
        original_plan = copy.deepcopy(self.execution.plan)
        arguments = dict(
            subgoal="clear the arms after folding",
            expected_feedback="hands clear of the cloth",
            segments=[dict(delta=[0, 0, 0, 0, 0, 0, 1] * 2, repeat=1)],
        )

        async def submit(request):
            before = self.state["step"]
            self.state["step"] += len(request["actions"])
            return {
                "step_before": before,
                "step_after": self.state["step"],
                "result": {"step": self.state["step"], "executed_rows": 1},
            }

        self.memory.submit = AsyncMock(side_effect=submit)
        async with Client(create_server(self.memory)) as client:
            bad = await client.call_tool(
                "assess_stage",
                dict(
                    stage_id="fold",
                    checks=[self.check([old_measurement, current])],
                    decision="complete",
                    rationale="combine a pre-visit fit with the current view",
                ),
            )
            self.assertTrue(bad.is_error)
            self.assertIsNone(self.execution.dispatch_gate)
            rejected = await client.call_tool("enter_stage", {"stage_id": "home"})
            self.assertTrue(rejected.is_error)
            self.assertEqual(self.execution.active, "fold")
            self.assertEqual(self.execution.plan, original_plan)
            gate = copy.deepcopy(self.execution.dispatch_gate)
            self.assertEqual(gate["target_stage_id"], "home")
            self.assertEqual(gate["from_stage_id"], "fold")
            for filename in ("execution-state.json", "episode-context.json"):
                self.assertEqual(
                    json.loads((self.path / filename).read_text())["dispatch_gate"], gate
                )
            blocked = await client.call_tool("execute_chunk", arguments)
            self.assertTrue(blocked.is_error)
            self.memory.submit.assert_not_awaited()
            self.assertEqual(self.execution.actions, 0)
            self.assertIsNone(self.execution.in_flight)
            corrected = await client.call_tool(
                "assess_stage",
                dict(
                    stage_id="fold",
                    checks=[self.check([current])],
                    decision="complete",
                    rationale="use evidence from this stage visit",
                ),
            )
            self.assertFalse(corrected.is_error)
            self.assertEqual(self.execution.dispatch_gate, gate)
            entered = await client.call_tool("enter_stage", {"stage_id": "home"})
            self.assertFalse(entered.is_error)
            self.assertIsNone(self.execution.dispatch_gate)
            result = await client.call_tool("execute_chunk", arguments)
            self.assertFalse(result.is_error)
            self.memory.submit.assert_awaited_once()
            self.assertEqual(self.execution.last_action["stage_id"], "home")
            self.assertEqual(self.execution.plan, original_plan)

        ledger = [
            json.loads(line)
            for line in (self.path / "episode-ledger.jsonl").read_text().splitlines()
        ]
        rejected = next(e for e in ledger if e["kind"] == "stage_transition_rejected")
        blocked = next(e for e in ledger if e["kind"] == "action_dispatch_blocked")
        resolved = next(e for e in ledger if e["kind"] == "stage_transition_resolved")
        self.assertEqual(rejected["id"], gate["rejected_event_id"])
        self.assertNotIn("action_id", blocked)
        self.assertEqual(resolved["rejected_event_id"], rejected["id"])
        self.assertEqual(resolved["resolution"], "entered_target")
        self.assertEqual(resolved["entered_stage_id"], "home")
        self.assertEqual(resolved["observation_id"], current)
        self.assertEqual(resolved["observed_at_step"], 1)

    def test_explicit_recovery_requires_current_presented_observation_and_allows_actions(self):
        self.plan([phase(), phase("home", ["fold"], "release_inspect_home")])
        self.execution.enter("fold", "", "entry")
        with self.assertRaisesRegex(ValueError, "dependencies"):
            self.execution.enter("home", "", "entry")
        gate = copy.deepcopy(self.execution.dispatch_gate)
        with self.assertRaisesRegex(ValueError, "Unknown current plan phase"):
            self.execution.enter("typo", "", "entry")
        self.assertEqual(self.execution.dispatch_gate, gate)
        for observation, reason in (("", "entry"), ("need recovery", "entry"), ("", "revisit")):
            with self.assertRaisesRegex(ValueError, "explicitly resume"):
                self.execution.enter("fold", observation, reason)
        self.state["step"] = 1
        with self.assertRaisesRegex(ValueError, "current observed step"):
            self.execution.enter("fold", "need recovery", "revisit")
        self.execution.observation(self.state, [])
        with self.assertRaisesRegex(ValueError, "presented RGB"):
            self.execution.enter("fold", "need recovery", "exception")
        current = self.execution.observation(self.state, ["head"])
        self.execution.enter(
            "fold", "sleeve still raised; lower before retrying completion", "revisit"
        )
        self.assertIsNone(self.execution.dispatch_gate)
        self.assertEqual(self.execution.plan["version"], 1)
        action = self.execution.start_action(
            dict(subgoal="lower sleeve", expected_feedback="sleeve supported")
        )
        self.assertEqual(action, "action-00001")
        ledger = [
            json.loads(line)
            for line in (self.path / "episode-ledger.jsonl").read_text().splitlines()
        ]
        resolved = next(e for e in ledger if e["kind"] == "stage_transition_resolved")
        self.assertEqual(resolved["resolution"], "resumed_current")
        self.assertEqual(resolved["observation_id"], current)
        self.assertEqual(resolved["observed_at_step"], 1)
        self.assertEqual(resolved["entered_stage_id"], "fold")

    def test_corrected_stage_name_resolves_gate_with_original_target_preserved(self):
        self.plan([phase(), phase("home", ["fold"], "release_inspect_home")])
        self.execution.enter("fold", "", "entry")
        self.finish("fold")
        with self.assertRaisesRegex(ValueError, "Unknown current plan phase"):
            self.execution.enter("hoem", "", "entry")
        self.execution.enter("home", "", "entry")
        self.assertIsNone(self.execution.dispatch_gate)
        self.assertEqual(self.execution.active, "home")
        self.assertEqual(self.execution.plan["version"], 1)
        ledger = [
            json.loads(line)
            for line in (self.path / "episode-ledger.jsonl").read_text().splitlines()
        ]
        resolved = next(e for e in ledger if e["kind"] == "stage_transition_resolved")
        self.assertEqual(resolved["target_stage_id"], "hoem")
        self.assertEqual(resolved["entered_stage_id"], "home")
        self.assertEqual(resolved["resolution"], "entered_target")

    def test_local_revision_preserves_completed_history_and_versions(self):
        self.plan([phase(), phase("home", ["fold"], "release_inspect_home")])
        self.execution.enter("fold", "", "entry")
        self.finish("fold")
        previous = copy.deepcopy(self.execution.plan)
        replacement = phase("home", ["fold"], "release_inspect_home")
        replacement["goal"] = "Clear another arm after object position changed"
        event = {
            "kind": "sop_inapplicable",
            "description": "old clearance path obstructed",
            "evidence_ids": [self.execution.last_observation],
        }
        with self.assertRaisesRegex(ValueError, "completed phase history"):
            self.execution.revise(1, event, [phase()], [], "must retain history")
        self.execution.revise(1, event, [replacement], [], "new arm clearance path")
        self.assertEqual(self.execution.plan["previous_version"], 1)
        self.assertEqual(self.execution.plan["phases"][0], previous["phases"][0])
        self.assertEqual(self.execution.progress["fold"]["status"], "complete")
        self.assertEqual(json.loads((self.path / "plan-versions/v001.json").read_text()), previous)
        self.assertTrue((self.path / "plan-versions/v002.json").exists())

    def test_revisit_invalidates_completed_descendants_but_not_independent_phase(self):
        self.plan([phase(), phase("independent"), phase("home", ["fold"], "release_inspect_home")])
        for name in ("fold", "independent", "home"):
            self.execution.enter(name, "", "entry")
            self.finish(name)
        self.execution.enter("fold", "sleeve shifted; correction needed", "revisit")
        self.assertEqual(self.execution.plan["version"], 1)
        self.assertEqual(self.execution.progress["home"]["status"], "pending")
        self.assertTrue(self.execution.progress["home"]["needs_revalidation"])
        self.assertEqual(self.execution.progress["independent"]["status"], "complete")

    def test_compound_effect_combines_stage_process_evidence_with_current_outcome(self):
        compound = phase()
        compound["expected_effects"] = ["sleeve follows lift and remains inward after release"]
        self.plan([compound])
        before_visit = self.execution.last_observation
        self.state["step"] = 1
        self.execution.observation(self.state, ["head"])
        self.execution.enter("fold", "current sleeve visible", "entry")
        observed = []
        for subgoal in ("lift sleeve", "release sleeve"):
            request = {"subgoal": subgoal, "expected_feedback": subgoal}
            action_id = self.execution.start_action(request)
            self.state["step"] += 1
            self.execution.finish_action(
                action_id,
                request,
                {"result": {"step": self.state["step"], "executed_rows": 1}},
                self.state,
            )
            observed.append(self.execution.observation(self.state, ["head"]))
        check = {
            "effect": compound["expected_effects"][0],
            "status": "met",
            "evidence_ids": [observed[0]],
            "observation": "Earlier lift showed following; latest view shows released sleeve inward",
        }
        with self.assertRaisesRegex(ValueError, "latest action"):
            self.execution.assess("fold", [check], "complete", "old evidence alone", [])
        check["evidence_ids"] = [before_visit, observed[1]]
        with self.assertRaisesRegex(ValueError, "current visit"):
            self.execution.assess("fold", [check], "complete", "cross-visit evidence", [])
        check["evidence_ids"] = observed
        result = self.execution.assess(
            "fold", [check], "complete", "process plus current outcome", []
        )
        self.assertEqual(result["execution"]["status"], "complete")
        self.assertEqual(self.execution.plan["version"], 1)

    def test_failed_measurement_unknown_response_and_no_rag_are_explicit(self):
        self.plan()
        self.execution.enter("fold", "", "entry")
        receipt = {"result": {"plan_ok": False, "reason": "occluded", "measured_at_step": 0}}
        self.execution.measurement({"measure": {}}, receipt, self.state)
        evidence = [receipt["result"]["evidence_id"]]
        with self.assertRaisesRegex(ValueError, "failed measurement"):
            self.execution.assess("fold", [self.check(evidence)], "complete", "claim", [])
        self.execution.assess("fold", [self.check(evidence, "unknown")], "continue", "occluded", [])
        request = {"subgoal": "fold", "expected_feedback": "follow"}
        action_id = self.execution.start_action(request)
        self.execution.fail_action(action_id, TimeoutError("response missing"))
        with self.assertRaisesRegex(ValueError, "pending action"):
            self.execution.start_action(request)
        self.assertEqual(self.execution.last_action["receipt_status"], "unknown")
        with self.assertRaisesRegex(ValueError, "not supported"):
            EpisodeSession(
                self.path,
                self.path,
                self.path,
                self.path,
                "fold_clothes",
                architecture="staged-v1",
                no_rag=True,
            )


if __name__ == "__main__":
    unittest.main()
