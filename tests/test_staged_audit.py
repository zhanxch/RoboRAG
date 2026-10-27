"""Adversarial saved-evidence checks; never invoke a simulator or model."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from roborag.evaluation.staged import audit_structure, evaluate
from roborag.io import read_json, read_jsonl, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ledger_write(path, events):
    for number, event in enumerate(events, 1):
        event["id"] = f"event-{number:05d}"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def fixture(root, name="candidate", architecture="staged-v1", layout=3):
    path = root / name
    path.mkdir()
    bank = root / "bank"
    version = dict(
        architecture="staged-v1",
        sop_id="push_T/stages",
        sop_version="v1",
        sop_sha256="sop-hash",
        prompt_sha256="task-prompt-hash",
        experience_index_sha256="index-hash",
        graph_sha256={"train": "graph-hash"},
    )
    version["id"] = hashlib.sha256(json.dumps(version, sort_keys=True).encode()).hexdigest()
    config = dict(
        task="push_T",
        model="same-model",
        layout_sha256=f"scene-{layout}",
        prompt_sha256="delivered-prompt",
        source_sha256={
            str(root / "src/roborag/tasks/push_T/sop.json"): "sop-hash",
            str(root / "src/roborag/tasks/push_T/prompt.txt"): "task-prompt-hash",
            str(bank / "execution-index.json"): "index-hash",
            str(bank / "graphs/train.json"): "graph-hash",
        },
        args=dict(
            task="push_T",
            seed=1,
            variant="standard",
            layout=layout,
            effort="xhigh",
            controller="delta",
            no_rag=False,
            architecture=architecture,
            max_wall_seconds=2160,
            bank=str(bank),
        ),
    )
    write_json(path / "run.json", config)
    native = dict(eval_time=1, success_rate=1, score=100)
    write_json(path / "native.json", native)
    write_json(
        path / "result.json",
        dict(
            complete=True,
            native=native,
            native_result=str(path / "native.json"),
            native_sha256=sha(path / "native.json"),
            wall_seconds=120,
        ),
    )
    write_json(
        path / "agent-exit.json",
        dict(
            wall_seconds=100,
            completed_turns=1,
            accounting=dict(
                reported_usage=dict(
                    input_tokens=1000,
                    cached_input_tokens=900,
                    output_tokens=60,
                    reasoning_output_tokens=20,
                )
            ),
        ),
    )
    write_json(path / "episode_end.json", dict(step=1))
    write_json(
        path / "sop-index.json", dict(schema="roborag.sop-experience-index.v1", version=version)
    )
    phase = dict(
        id="align",
        sop_stage="align",
        goal="Align the visible block",
        object_bindings={"block": "visible T"},
        expected_effects=["block aligned"],
        depends_on=[],
        experience=[
            dict(
                source="train:p1",
                adopted="short contact push",
                adaptation="use measured current position",
            )
        ],
    )
    plan = dict(
        schema="roborag.graph-plan.v1",
        role="GPT_FUTURE_PLAN",
        version=1,
        previous_version=None,
        sop_version=version["id"],
        source_versions=version,
        object_bindings={"block": "visible T"},
        current_scene="block left of goal",
        phases=[phase],
    )
    write_json(path / "gpt-plan.json", plan)
    write_json(path / "plan-versions/v001.json", plan)
    snapshots = []
    for step in (0, 1):
        snap = path / "observations" / str(step)
        write_json(snap / "state.json", dict(step=step))
        (snap / "head.png").write_bytes(b"fixture-image")
        snapshots.append(
            dict(
                id=f"obs-{step}",
                kind="observation",
                role="CURRENT_OBSERVATION",
                step=step,
                cameras=["head"] if step == 0 else [],
                snapshot=str(snap),
                sha256={p.name: sha(p) for p in snap.iterdir()},
            )
        )
    receipt = dict(step=1, executed_rows=1)
    attribution = dict(action_id="action-00001", stage_id="align", plan_version=1)
    action = dict(
        id="action-00001",
        stage_id="align",
        plan_version=1,
        expected_feedback="block aligned",
        receipt=receipt,
        observation_id="obs-1",
        effect_status="unknown; assess effects separately",
    )
    events = []

    def event(kind, plan_version=None, stage_id=None, **details):
        events.append(
            dict(
                role="CURRENT_EXECUTION_RECORD",
                kind=kind,
                plan_version=plan_version,
                stage_id=stage_id,
                **details,
            )
        )

    event("observation_recorded", evidence=snapshots[0])
    event("sop_loaded", version=version)
    event(
        "stage_prepared",
        sop_stage="align",
        allowed_refs=["train:p1"],
        version_id=version["id"],
        diagnostics=dict(cache_hit=False, search_performed=True),
    )
    event("plan_created", plan_version=1, phase_ids=["align"])
    event(
        "stage_entered", plan_version=1, stage_id="align", context_reused=False, entered_at_step=0
    )
    event(
        "action_submitted",
        **attribution,
        subgoal="align",
        expected_feedback="block aligned",
        step_before=0,
        observation_id="obs-0",
    )
    event("observation_recorded", plan_version=1, stage_id="align", evidence=snapshots[1])
    event("action_received", plan_version=1, stage_id="align", action=action)
    event(
        "observation_presented",
        plan_version=1,
        stage_id="align",
        evidence_id="obs-1",
        cameras=["head"],
    )
    event(
        "stage_assessed",
        plan_version=1,
        stage_id="align",
        checks=[
            dict(
                effect="block aligned",
                status="met",
                evidence_ids=["obs-1"],
                observation="visible block overlaps target",
            )
        ],
        decision="complete",
        rationale="current view supports alignment",
        experience_adaptations=[],
    )
    ledger_write(path / "episode-ledger.jsonl", events)
    write_json(
        path / "execution-state.json",
        dict(
            role="CURRENT_EXECUTION_STATE",
            plan_version=1,
            active_stage="align",
            in_flight=None,
            stages=[dict(**phase, execution=dict(status="complete"))],
        ),
    )
    command = dict(
        request=dict(actions=[[0] * 14], execution_context=attribution),
        result=receipt,
        step_after=1,
    )
    (path / "commands.jsonl").write_text(json.dumps(command) + "\n")
    (path / "mcp-calls.jsonl").write_text(
        json.dumps(dict(tool="read_action_reference", result={})) + "\n"
    )
    return path


class StagedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = fixture(self.root)
        self.cases = [dict(seed=1, variant="standard", layout=3)]

    def mutate(self, change):
        path = self.path / "episode-ledger.jsonl"
        events, _ = read_jsonl(path)
        change(events)
        ledger_write(path, events)

    def test_coherent_native_run_and_scoped_cost_gate(self):
        structure = audit_structure(self.path)
        self.assertTrue(structure["passed"], structure["errors"])
        baseline = fixture(self.root, "baseline", "legacy")
        report = evaluate([self.path], "push_T", self.cases, [baseline])
        self.assertTrue(report["gate"]["passed"], report["gate"])
        self.assertFalse(report["gate"]["default_promotion_allowed"])
        self.assertEqual(report["overall"]["uncached_input_tokens"], 100)
        self.assertEqual(report["overall"]["usage"]["output_tokens"], 60)
        self.assertEqual(report["structure_cost"]["stage_searches"], 1)
        self.assertIsNone(report["overall"]["model_request_count"])

    def test_wrong_receipt_phase_and_native_unowned_dispatch_fail(self):
        self.mutate(
            lambda events: next(e for e in events if e["kind"] == "action_received")[
                "action"
            ].update(stage_id="other")
        )
        errors = audit_structure(self.path)["errors"]
        self.assertTrue(any("attribution" in e for e in errors), errors)
        command = read_jsonl(self.path / "commands.jsonl")[0][0]
        command["request"]["execution_context"]["action_id"] = "unlogged"
        (self.path / "commands.jsonl").write_text(json.dumps(command) + "\n")
        self.assertFalse(audit_structure(self.path)["passed"])

    def test_lost_receipt_stays_unresolved_even_with_native_success(self):
        def change(events):
            event = next(e for e in events if e["kind"] == "action_received")
            event.pop("action")
            event.update(kind="action_response_unknown", action_id="action-00001", error="timeout")

        self.mutate(change)
        report = evaluate([self.path], "push_T", self.cases)
        self.assertEqual(report["overall"]["successes"], 1)
        self.assertFalse(report["gate"]["passed"])
        self.assertEqual(report["cases"][0]["structure"]["unresolved_actions"], ["action-00001"])

    def test_missing_native_result_is_recorded_unscored_and_keeps_cost(self):
        (self.path / "native.json").unlink()
        report = evaluate([self.path], "push_T", self.cases)
        self.assertEqual(report["overall"]["recorded"], 1)
        self.assertEqual(report["overall"]["scored"], 0)
        self.assertIsNone(report["overall"]["success_rate"])
        self.assertEqual(report["overall"]["sum_episode_wall_seconds"], 120)
        self.assertFalse(report["gate"]["passed"])

    def test_budget_and_bank_differences_block_pairing(self):
        baseline = fixture(self.root, "baseline", "legacy")
        original = read_json(baseline / "run.json")
        for change, expected in (
            (lambda c: c["args"].update(max_wall_seconds=3600), "max_wall_seconds"),
            (
                lambda c: c["source_sha256"].update(
                    {str(self.root / "bank/execution-index.json"): "changed"}
                ),
                "bank_sources",
            ),
        ):
            config = copy.deepcopy(original)
            change(config)
            write_json(baseline / "run.json", config)
            report = evaluate([self.path], "push_T", self.cases, [baseline])
            self.assertFalse(report["gate"]["passed"])
            self.assertTrue(any(expected in e for e in report["gate"]["comparability_errors"]))

    def test_missing_token_and_cache_evidence_remains_unknown(self):
        (self.path / "agent-exit.json").unlink()
        self.mutate(
            lambda events: next(e for e in events if e["kind"] == "stage_prepared").update(
                diagnostics={}
            )
        )
        report = evaluate([self.path], "push_T", self.cases)
        self.assertIsNone(report["overall"]["uncached_input_tokens"])
        self.assertIsNone(report["overall"]["usage"]["input_tokens"])
        self.assertIsNone(report["structure_cost"]["stage_cache_hits"])
        self.assertFalse(report["gate"]["passed"])

    def test_ordinary_feedback_cannot_advance_plan_version(self):
        self.mutate(
            lambda events: next(e for e in events if e["kind"] == "stage_assessed").update(
                plan_version=2
            )
        )
        errors = audit_structure(self.path)["errors"]
        self.assertIn("Ordinary execution event changed the persistent plan version", errors)

    def test_compound_effect_uses_stage_history_and_latest_anchor_but_not_history_alone(self):
        snapshot = self.path / "observations/2"
        write_json(snapshot / "state.json", dict(step=2))
        (snapshot / "head.png").write_bytes(b"final-state-image")
        record = dict(
            id="obs-2",
            kind="observation",
            role="CURRENT_OBSERVATION",
            step=2,
            cameras=["head"],
            snapshot=str(snapshot),
            sha256={p.name: sha(p) for p in snapshot.iterdir()},
        )
        receipt = dict(step=2, executed_rows=1)
        attribution = dict(action_id="action-00002", stage_id="align", plan_version=1)
        additions = [
            dict(
                kind="action_submitted",
                **attribution,
                subgoal="release and inspect",
                expected_feedback="block aligned",
                step_before=1,
                observation_id="obs-1",
            ),
            dict(kind="observation_recorded", evidence=record),
            dict(
                kind="action_received",
                action=dict(
                    id="action-00002",
                    stage_id="align",
                    plan_version=1,
                    expected_feedback="block aligned",
                    receipt=receipt,
                    observation_id="obs-2",
                    effect_status="unknown",
                ),
            ),
        ]

        def change(events):
            assessment = events.pop()
            assessment["checks"][0].update(
                evidence_ids=["obs-1", "obs-2"],
                observation="movement milestone followed by released aligned block",
            )
            events.extend(
                dict(role="CURRENT_EXECUTION_RECORD", plan_version=1, stage_id="align") | item
                for item in additions
            )
            events.append(assessment)

        self.mutate(change)
        with (self.path / "commands.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        request=dict(actions=[[0] * 14], execution_context=attribution),
                        result=receipt,
                        step_after=2,
                    )
                )
                + "\n"
            )
        write_json(self.path / "episode_end.json", dict(step=2))
        audited = audit_structure(self.path)
        self.assertTrue(audited["passed"], audited["errors"])
        self.mutate(lambda events: events[-1]["checks"][0].update(evidence_ids=["obs-1"]))
        self.assertTrue(
            any("current result anchor" in error for error in audit_structure(self.path)["errors"])
        )

    def test_local_revision_requires_invalidation_and_parent_evidence(self):
        initial = read_json(self.path / "gpt-plan.json")
        changed = copy.deepcopy(initial)
        changed.update(version=2, previous_version=1)
        changed["phases"][0]["goal"] = "Use alternate contact after obstruction"
        revision = dict(
            event=dict(
                kind="sop_inapplicable",
                description="Current path obstructed",
                evidence_ids=["obs-0"],
            ),
            rationale="Use reachable side",
            affected_phase_ids=["align"],
        )
        changed["revision"] = revision
        write_json(self.path / "plan-versions/v002.json", changed)
        write_json(self.path / "gpt-plan.json", changed)
        state = read_json(self.path / "execution-state.json")
        state["plan_version"] = 2
        write_json(self.path / "execution-state.json", state)
        command = read_jsonl(self.path / "commands.jsonl")[0][0]
        command["request"]["execution_context"]["plan_version"] = 2
        (self.path / "commands.jsonl").write_text(json.dumps(command) + "\n")

        def change(events):
            insertion = next(i for i, e in enumerate(events) if e["kind"] == "stage_entered")
            for event in events[insertion:]:
                event["plan_version"] = 2
                if event["kind"] == "action_received":
                    event["action"]["plan_version"] = 2
            events.insert(
                insertion,
                dict(
                    role="CURRENT_EXECUTION_RECORD",
                    kind="plan_revised",
                    stage_id=None,
                    plan_version=2,
                    previous_version=1,
                    revision=revision,
                ),
            )

        self.mutate(change)
        structure = audit_structure(self.path)
        self.assertTrue(structure["passed"], structure["errors"])
        self.mutate(
            lambda events: next(e for e in events if e["kind"] == "plan_revised")["revision"][
                "event"
            ].update(kind="ordinary_feedback")
        )
        self.assertTrue(any("invalidating" in e for e in audit_structure(self.path)["errors"]))

    def test_native_failure_cannot_pass_baseline_success_or_lower_cost_gate(self):
        baseline = fixture(self.root, "baseline", "legacy")
        native = read_json(self.path / "native.json")
        native.update(success_rate=0, score=0)
        write_json(self.path / "native.json", native)
        result = read_json(self.path / "result.json")
        result.update(native=native, native_sha256=sha(self.path / "native.json"), wall_seconds=1)
        write_json(self.path / "result.json", result)
        report = evaluate([self.path], "push_T", self.cases, [baseline])
        self.assertTrue(report["gate"]["coherent_structure"])
        self.assertFalse(report["gate"]["effectiveness_passed"])
        self.assertEqual(report["paired"]["jointly_successful"]["count"], 0)

    def test_baseline_audit_file_resolves_selected_case_directories(self):
        baseline = fixture(self.root, "baseline", "legacy")
        audit = self.root / "baseline-audit.json"
        write_json(
            audit,
            dict(
                proposed_paired_regression=dict(
                    cases=[dict(task="push_T", baseline_directory=str(baseline))]
                )
            ),
        )
        self.assertTrue(evaluate([self.path], "push_T", self.cases, [audit])["gate"]["passed"])

    def test_unpresented_or_changed_observation_cannot_support_completion(self):
        self.mutate(
            lambda events: events.remove(
                next(e for e in events if e["kind"] == "observation_presented")
            )
        )
        self.assertTrue(any("never presented" in e for e in audit_structure(self.path)["errors"]))
        (self.path / "observations/1/head.png").write_bytes(b"replaced")
        self.assertTrue(any("snapshot hashes" in e for e in audit_structure(self.path)["errors"]))

    def test_missing_structure_is_not_passed_by_native_success(self):
        (self.path / "episode-ledger.jsonl").unlink()
        report = evaluate([self.path], "push_T", self.cases)
        self.assertFalse(report["gate"]["passed"])
        self.assertIsNone(report["structure_cost"]["stage_cache_hits"])

    def test_cli_preserves_partial_report_and_pending_cases(self):
        (self.path / "result.json").unlink()
        cases_path, output = self.root / "cases.json", self.root / "evaluation.json"
        write_json(cases_path, self.cases + [dict(seed=1, variant="standard", layout=4)])
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "roborag",
                "evaluate-staged",
                "--task",
                "push_T",
                "--runs",
                str(self.path),
                "--cases",
                str(cases_path),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        report = read_json(output)
        self.assertEqual(len(report["pending"]), 1)
        self.assertEqual(report["overall"]["recorded"], 1)
        self.assertFalse(report["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
