"""Persistent future plan and append-only current execution evidence.

This module does not move the robot, infer object truth or decide task success.
Model assessments are attributable claims about public observations. Mechanical
receipts establish command execution only, never completion of a task effect.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

from roborag.io import write_json


class StagedEpisode:
    def __init__(self, memory):
        self.memory = memory
        self.plan = None
        self.active = None
        self.progress = {}
        self.evidence = {}
        self.prepared = {}
        self.prepared_sop = {}
        self.events = 0
        self.actions = 0
        self.in_flight = None
        self.last_action = None
        self.last_observation = None
        self.dispatch_gate = None
        self.sop_seen = False
        self._library = None

    @property
    def library(self):
        if self._library is None:
            from roborag.experience import StageLibrary

            self._library = StageLibrary(self.memory.bank, self.memory.task)
        return self._library

    def event(self, kind, **details):
        self.events += 1
        record = dict(
            id=f"event-{self.events:05d}",
            role="CURRENT_EXECUTION_RECORD",
            kind=kind,
            plan_version=self.plan["version"] if self.plan else None,
            stage_id=self.active,
        )
        record.update(details)
        self.memory.output.mkdir(parents=True, exist_ok=True)
        with (self.memory.output / "episode-ledger.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return record["id"]

    def status(self):
        phases = self.plan["phases"] if self.plan else []
        return dict(
            role="CURRENT_EXECUTION_STATE",
            architecture="staged-v1",
            plan_version=self.plan["version"] if self.plan else None,
            sop_version=self.plan["sop_version"] if self.plan else None,
            current_scene=self.plan["current_scene"] if self.plan else None,
            object_bindings=self.plan["object_bindings"] if self.plan else {},
            active_stage=self.active,
            stages=[dict(**p, execution=self.progress[p["id"]]) for p in phases],
            last_observation=self.last_observation,
            recent_evidence=list(self.evidence.values())[-8:],
            last_action=self.last_action,
            in_flight=self.in_flight,
            dispatch_gate=copy.deepcopy(self.dispatch_gate),
            task_success="unknown: native evaluation is separate and unavailable to this policy",
            recovery_file="episode-context.json",
        )

    def persist(self):
        state = self.status()
        write_json(self.memory.output / "execution-state.json", state)
        write_json(self.memory.workspace / "episode-context.json", state)

    def bootstrap(self):
        result = self.library.bootstrap()
        self.sop_seen = True
        write_json(self.memory.output / "sop-index.json", self.library.export_index())
        self.event("sop_loaded", version=result.get("version"))
        self.persist()
        return result

    def prepare(
        self,
        sop_stage,
        *,
        observation="",
        goal="",
        arm="",
        held_object="",
        target_object="",
        reason="preview",
    ):
        result = self.library.prepare(
            sop_stage,
            goal=goal,
            observation=observation,
            arm=arm,
            held_object=held_object,
            target_object=target_object,
            known_refs=self.memory.references,
            refresh=reason in ("gap", "state_change", "exception"),
        )
        self.memory.references.update(result.get("allowed_refs", []))
        self.memory.retrieved = True
        self.prepared_sop[sop_stage] = copy.deepcopy(result)
        self.event(
            "stage_prepared",
            sop_stage=sop_stage,
            reason=reason,
            diagnostics=result.get("diagnostics", {}),
            allowed_refs=result.get("allowed_refs", []),
            version_id=result.get("version_id"),
        )
        self.persist()
        return result

    def _idle(self):
        if self.in_flight is not None:
            raise ValueError("Wait for the pending action receipt before changing execution state")

    def _phase(self, stage_id):
        if self.plan is None:
            raise ValueError("record_graph_plan must precede stage execution")
        return next((p for p in self.plan["phases"] if p["id"] == stage_id), None)

    def _validate_graph(self, phases):
        ids = [p["id"] for p in phases]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Plan requires nonempty, unique phase IDs")
        # prepare() validates the SOP stage without adding this future plan as fact.
        overview = self.library.overview()
        stages = overview.get("stages", [])
        sop_ids = {p["id"] if isinstance(p, dict) else p for p in stages}
        for phase in phases:
            if sop_ids and phase["sop_stage"] not in sop_ids:
                raise ValueError("Unknown SOP stage: " + phase["sop_stage"])
            if any(d not in ids for d in phase["depends_on"]):
                raise ValueError("Every dependency must name a current plan phase")
            effects = phase["expected_effects"]
            if (
                not effects
                or any(not e.strip() for e in effects)
                or len(set(effects)) != len(effects)
            ):
                raise ValueError("Expected effects must be nonempty, distinct observable criteria")
            if any(use["source"] not in self.memory.references for use in phase["experience"]):
                raise ValueError("Adopt only retrieved experience source IDs")
            if not phase["object_bindings"] or any(
                not str(v).strip() for v in phase["object_bindings"].values()
            ):
                raise ValueError("Each phase needs current object bindings")
        remaining = {p["id"]: set(p["depends_on"]) for p in phases}
        while remaining:
            ready = {key for key, deps in remaining.items() if not deps}
            if not ready:
                raise ValueError(
                    "Plan dependencies must be acyclic; historical order is not a dependency"
                )
            remaining = {key: deps - ready for key, deps in remaining.items() if key not in ready}

    def _save_plan(self):
        self.memory.plan = self.plan
        write_json(
            self.memory.output / "plan-versions" / f"v{self.plan['version']:03d}.json", self.plan
        )
        write_json(self.memory.output / "gpt-plan.json", self.plan)
        self.persist()

    def record_plan(self, current_scene, object_bindings, phases, adaptation, sop_version):
        self._idle()
        if self.plan is not None:
            raise ValueError(
                "A persistent plan already exists; use revise_plan only if it is inapplicable"
            )
        if not self.sop_seen:
            raise ValueError("get_task_sop must precede graph planning")
        self._live_evidence([self.last_observation], current=True)
        version = self.library.overview()["version"]
        expected_version = version["id"]
        if sop_version != expected_version:
            raise ValueError(f"Use the retrieved SOP version: {expected_version}")
        if not current_scene.strip() or not object_bindings:
            raise ValueError("Bind the plan to the current scene and objects")
        self._validate_graph(phases)
        self.plan = dict(
            schema="roborag.graph-plan.v1",
            role="GPT_FUTURE_PLAN",
            version=1,
            previous_version=None,
            sop_version=sop_version,
            source_versions=version,
            current_scene=current_scene,
            object_bindings=object_bindings,
            phases=copy.deepcopy(phases),
            adaptation=adaptation,
            observed_at_step=self.memory.state()["step"],
            observation_id=self.last_observation,
        )
        self.progress = {p["id"]: {"status": "pending", "checks": []} for p in phases}
        self.event("plan_created", phase_ids=[p["id"] for p in phases])
        self._save_plan()
        return {"recorded": True, "plan_version": 1, "phase_ids": list(self.progress)}

    def _live_evidence(self, ids, *, current=False, after_step=None):
        if not ids:
            raise ValueError(
                "An observed claim requires actual observation/measurement evidence IDs"
            )
        for evidence_id in ids:
            evidence = self.evidence.get(evidence_id)
            if evidence is None or evidence["kind"] not in ("observation", "measurement"):
                raise ValueError(
                    "Use live observation/measurement IDs, never historical references or command receipts"
                )
            if evidence["kind"] == "observation" and not evidence["cameras"]:
                raise ValueError(
                    "Use a presented RGB observation or a measurement, not an unseen snapshot"
                )
            if current and evidence["step"] != self.memory.state()["step"]:
                raise ValueError("This decision requires evidence from the current observed step")
            if after_step is not None and evidence["step"] < after_step:
                raise ValueError("Effect evidence predates this stage's current visit")

    def revise(self, revision_of, event, replace_phases, remove_phase_ids, rationale):
        self._idle()
        if self.plan is None or revision_of != self.plan["version"]:
            raise ValueError("revise_plan must name the current plan version")
        self._live_evidence(event["evidence_ids"], current=True)
        if not rationale.strip() or not (replace_phases or remove_phase_ids):
            raise ValueError("Explain the inapplicability event and submit a local phase patch")
        replacements = {p["id"]: p for p in replace_phases}
        if len(replacements) != len(replace_phases):
            raise ValueError("Duplicate phase replacement IDs")
        old_ids = set(self.progress)
        removed = set(remove_phase_ids)
        if removed - old_ids or removed & set(replacements):
            raise ValueError(
                "Remove only existing phases and do not replace them in the same patch"
            )
        affected = removed | set(replacements)
        if any(self.progress.get(key, {}).get("status") == "complete" for key in affected):
            raise ValueError("Preserve completed phase history; add a recovery phase instead")
        phases = [
            replacements.pop(p["id"], p) for p in self.plan["phases"] if p["id"] not in removed
        ]
        phases += list(replacements.values())
        self._validate_graph(phases)
        if phases == self.plan["phases"]:
            raise ValueError("No plan change: ordinary feedback/progress does not revise the plan")
        previous = self.plan["version"]
        self.plan = {
            **self.plan,
            "version": previous + 1,
            "previous_version": previous,
            "phases": copy.deepcopy(phases),
            "revision": dict(event=event, rationale=rationale, affected_phase_ids=sorted(affected)),
        }
        for key in removed:
            self.progress.pop(key)
            self.prepared.pop(key, None)
        for phase in replace_phases:
            self.progress[phase["id"]] = {"status": "pending", "checks": []}
            self.prepared.pop(phase["id"], None)
        if self.active in affected:
            self.active = None
        self.event("plan_revised", previous_version=previous, revision=self.plan["revision"])
        self._save_plan()
        return {
            "revised": True,
            "plan_version": self.plan["version"],
            "affected_phase_ids": sorted(affected),
        }

    def enter(self, stage_id, observation, reason):
        try:
            return self._enter(stage_id, observation, reason)
        except ValueError as exc:
            if self.plan is not None and stage_id != self.active:
                if self.dispatch_gate is None:
                    self.dispatch_gate = dict(
                        target_stage_id=stage_id,
                        from_stage_id=self.active,
                        rejected_event_id=f"event-{self.events + 1:05d}",
                        observed_at_step=self.memory.state()["step"],
                        error=str(exc),
                    )
                self.event(
                    "stage_transition_rejected",
                    attempted_stage_id=stage_id,
                    requested_reason=reason,
                    current_observation=observation,
                    error=str(exc),
                    observed_at_step=self.memory.state()["step"],
                    dispatch_gate=copy.deepcopy(self.dispatch_gate),
                )
                self.persist()
            raise

    def _enter(self, stage_id, observation, reason):
        self._idle()
        self.library.validate_version()
        phase = self._phase(stage_id)
        if phase is None:
            raise ValueError("Unknown current plan phase")
        gate = self.dispatch_gate
        resuming_current = stage_id == self.active
        if gate is not None:
            if resuming_current and (
                reason not in ("exception", "revisit") or not observation.strip()
            ):
                raise ValueError(
                    "A rejected stage transition blocks movement; enter a ready phase or "
                    "explicitly resume the current phase with reason=exception/revisit "
                    "and an observed recovery need"
                )
            self._live_evidence([self.last_observation], current=True)
        revisit = self.progress[stage_id]["status"] == "complete"
        if revisit and (
            reason not in ("revisit", "state_change", "exception") or not observation.strip()
        ):
            raise ValueError(
                "Revisit a completed phase only with an explicit observed change/recovery reason"
            )
        if any(self.progress[d]["status"] != "complete" for d in phase["depends_on"]):
            raise ValueError("Complete this phase's explicit dependencies first")
        if (
            self.active
            and self.active != stage_id
            and self.progress[self.active]["status"] == "active"
        ):
            raise ValueError("Assess the active phase as complete or blocked before switching")
        cached = self.prepared.get(stage_id) or self.prepared_sop.get(phase["sop_stage"])
        if cached and reason in ("entry", "revisit"):
            context = copy.deepcopy(cached)
            context["session_reuse"] = True
            context["applicability_check"] = (
                "Reused historical context; assess its conditions against the current observation. "
                "It does not establish current geometry, contact or grasp. Request gap/state_change/exception if needed."
            )
            self.prepared[stage_id] = copy.deepcopy(context)
        else:
            context = self.prepare(
                phase["sop_stage"], observation=observation, goal=phase["goal"], reason=reason
            )
            self.prepared[stage_id] = copy.deepcopy(context)
        self.active = stage_id
        progress = self.progress[stage_id]
        if revisit:
            progress["checks"] = []
            progress["visits"] = progress.get("visits", 1) + 1
            progress.pop("last_action_step", None)
            progress["entered_at_step"] = self.memory.state()["step"]
            affected = {stage_id}
            while True:
                descendants = {
                    p["id"] for p in self.plan["phases"] if set(p["depends_on"]) & affected
                }
                if descendants <= affected:
                    break
                affected.update(descendants)
            for key in affected - {stage_id}:
                previous = self.progress[key]
                if previous["status"] == "complete":
                    self.event(
                        "stage_progress_invalidated",
                        stage_id=key,
                        cause_stage=stage_id,
                        previous_assessment=copy.deepcopy(previous),
                    )
                    self.progress[key] = {
                        "status": "pending",
                        "checks": [],
                        "needs_revalidation": True,
                    }
        progress["status"] = "active"
        progress.setdefault("entered_at_step", self.memory.state()["step"])
        self.event(
            "stage_entered",
            context_reused=context.get("session_reuse", False),
            revisit=revisit,
            reason=reason,
            current_observation=observation,
            entered_at_step=progress["entered_at_step"],
        )
        if gate is not None:
            self.dispatch_gate = None
            self.event(
                "stage_transition_resolved",
                resolution="resumed_current" if resuming_current else "entered_target",
                rejected_event_id=gate["rejected_event_id"],
                target_stage_id=gate["target_stage_id"],
                entered_stage_id=stage_id,
                from_stage_id=gate["from_stage_id"],
                observation_id=self.last_observation,
                observed_at_step=self.memory.state()["step"],
                requested_reason=reason,
                current_observation=observation,
            )
        self.persist()
        return dict(
            stage_id=stage_id, plan_version=self.plan["version"], phase=phase, context=context
        )

    def assess(self, stage_id, checks, decision, rationale, experience):
        self._idle()
        phase = self._phase(stage_id)
        if phase is None or stage_id != self.active:
            raise ValueError("Assess the current active phase")
        effects = [c["effect"] for c in checks]
        if len(effects) != len(set(effects)) or set(effects) - set(phase["expected_effects"]):
            raise ValueError("Each check must identify one distinct expected effect verbatim")
        progress = self.progress[stage_id]
        after_step = progress.get("last_action_step", progress.get("entered_at_step"))
        for check in checks:
            if check["status"] != "unknown" or check["evidence_ids"]:
                self._live_evidence(
                    check["evidence_ids"], after_step=progress.get("entered_at_step")
                )
            if check["status"] != "unknown" and not any(
                self.evidence[e]["step"] >= after_step for e in check["evidence_ids"]
            ):
                raise ValueError(
                    "Effect evidence predates this stage's latest action; include a current "
                    "observation/measurement alongside earlier process evidence"
                )
            if check["status"] == "met" and all(
                self.evidence[e]["kind"] == "measurement"
                and self.evidence[e]["result"].get("plan_ok") is False
                for e in check["evidence_ids"]
            ):
                raise ValueError("A failed measurement cannot by itself establish a met effect")
        if decision == "complete" and (
            set(effects) != set(phase["expected_effects"])
            or any(c["status"] != "met" for c in checks)
        ):
            raise ValueError(
                "Complete only when every expected effect has met, evidence-backed assessment"
            )
        for use in experience:
            if use["source"] not in self.memory.references:
                raise ValueError("Adopt only retrieved experience source IDs")
        progress.update(
            status="complete"
            if decision == "complete"
            else "blocked"
            if decision == "blocked"
            else "active",
            checks=checks,
            decision=decision,
            rationale=rationale,
            assessed_at_step=self.memory.state()["step"],
        )
        if experience:
            progress.setdefault("experience_adaptations", []).extend(experience)
        self.event(
            "stage_assessed",
            checks=checks,
            decision=decision,
            rationale=rationale,
            experience_adaptations=experience,
        )
        self.persist()
        return dict(
            stage_id=stage_id,
            execution=progress,
            plan_version=self.plan["version"],
            task_success="unknown; stage completion is a model claim, not native success",
        )

    def observation(self, state, cameras, snapshot=None):
        if self.last_observation is not None:
            previous = self.evidence[self.last_observation]
            if previous["step"] == state["step"]:
                presented = set(cameras) - set(previous["cameras"])
                previous["cameras"] = sorted(set(previous["cameras"]) | set(cameras))
                if presented:
                    self.event(
                        "observation_presented",
                        evidence_id=self.last_observation,
                        cameras=list(cameras),
                    )
                    self.persist()
                return self.last_observation
        evidence_id = f"observation-{len(self.evidence) + 1:05d}"
        directory = (
            Path(snapshot) if snapshot else self.memory.output / "stage-observations" / evidence_id
        )
        if not snapshot:
            write_json(directory / "state.json", state)
            for camera in ("head", "wrist_l", "wrist_r"):
                source = self.memory.workspace / "obs" / f"{camera}.png"
                if source.exists():
                    shutil.copyfile(source, directory / f"{camera}.png")
        record = dict(
            id=evidence_id,
            kind="observation",
            role="CURRENT_OBSERVATION",
            step=state["step"],
            cameras=list(cameras),
            snapshot=str(directory),
            sha256={
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.glob("*")
                if path.is_file()
            },
        )
        self.evidence[evidence_id] = record
        self.last_observation = evidence_id
        self.event("observation_recorded", evidence=record)
        self.persist()
        return evidence_id

    def measurement(self, request, receipt, state, snapshot=None):
        observation_id = self.observation(state, [], snapshot)
        evidence_id = f"measurement-{len(self.evidence) + 1:05d}"
        record = dict(
            id=evidence_id,
            kind="measurement",
            role="CURRENT_MEASUREMENT",
            step=state["step"],
            observation_id=observation_id,
            request=request,
            result=copy.deepcopy(receipt["result"]),
        )
        self.evidence[evidence_id] = record
        self.event("measurement_recorded", evidence=record)
        receipt["result"]["evidence_id"] = evidence_id
        receipt["result"]["observation_id"] = observation_id
        self.persist()

    def start_action(self, request):
        self._idle()
        if self.dispatch_gate is not None:
            self.event(
                "action_dispatch_blocked",
                subgoal=request["subgoal"],
                expected_feedback=request["expected_feedback"],
                dispatch_gate=copy.deepcopy(self.dispatch_gate),
                observed_at_step=self.memory.state()["step"],
            )
            self.persist()
            raise ValueError(
                "Movement blocked after rejected stage transition; successfully enter a ready "
                "phase, or explicitly resume the current phase with reason=exception/revisit "
                "and current observation before issuing recovery actions"
            )
        if self.active is None or self.progress[self.active]["status"] != "active":
            raise ValueError("enter_stage must select an active graph-plan phase before movement")
        self.actions += 1
        self.in_flight = f"action-{self.actions:05d}"
        attribution = dict(
            action_id=self.in_flight, stage_id=self.active, plan_version=self.plan["version"]
        )
        request["execution_context"] = attribution
        self.event(
            "action_submitted",
            **attribution,
            subgoal=request["subgoal"],
            expected_feedback=request["expected_feedback"],
            step_before=self.memory.state()["step"],
            observation_id=self.last_observation,
        )
        self.persist()
        return self.in_flight

    def finish_action(self, action_id, request, receipt, state):
        observation_id = self.observation(state, [])
        self.last_action = dict(
            id=action_id,
            stage_id=self.active,
            plan_version=self.plan["version"],
            expected_feedback=request["expected_feedback"],
            receipt=receipt["result"],
            observation_id=observation_id,
            effect_status="unknown; assess observed effects separately",
        )
        self.progress[self.active]["last_action_step"] = state["step"]
        self.event("action_received", action=self.last_action)
        self.in_flight = None
        self.persist()

    def fail_action(self, action_id, error):
        self.event("action_response_unknown", action_id=action_id, error=str(error))
        self.last_action = dict(
            id=action_id, stage_id=self.active, receipt_status="unknown", error=str(error)
        )
        # Keep the pending action visible: do not allow a blind motion retry.
        self.persist()
