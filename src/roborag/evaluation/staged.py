"""Offline staged-run audit and limited-scope paired gate; never drives the policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from roborag.evaluation.comparison import aggregate, candidate_rows
from roborag.evaluation.episode import control_accounting
from roborag.evaluation.suite import case_key, paired, suite
from roborag.io import read_json, read_jsonl, write_json

INVALIDATIONS = {"object_unavailable", "goal_changed", "dependency_invalidated", "sop_inapplicable"}
TASKS = ("fold_clothes", "push_T", "pour_balls_into_vase")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, errors, default=None):
    try:
        return read_json(path, default)
    except (OSError, ValueError) as exc:
        errors.append(f"Unreadable {Path(path).name}: {exc}")
        return default


def _bank_sources(config):
    """Compare bank content independently of checkout-specific absolute paths."""
    bank = config.get("args", {}).get("bank")
    prefix = str(Path(bank).resolve()) + "/" if bank else None
    result = {}
    for path, value in config.get("source_sha256", {}).items():
        if prefix and path.startswith(prefix):
            result[path[len(prefix) :]] = value
        elif "/graph-bank/" in path:
            tail = path.split("/graph-bank/", 1)[1]
            if "/" in tail:
                result[tail.split("/", 1)[1]] = value
    return result


def _version_errors(index, config, task):
    errors = []
    version = index.get("version", {})
    if index.get("schema") != "roborag.sop-experience-index.v1" or not isinstance(version, dict):
        return ["Missing or invalid SOP experience index"], {}
    expected_id = hashlib.sha256(
        json.dumps({k: v for k, v in version.items() if k != "id"}, sort_keys=True).encode()
    ).hexdigest()
    if version.get("id") != expected_id:
        errors.append("SOP/index version content hash differs")
    sources = config.get("source_sha256", {})
    for field, suffix in (
        ("sop_sha256", f"/tasks/{task}/sop.json"),
        ("prompt_sha256", f"/tasks/{task}/prompt.txt"),
    ):
        hashes = {value for path, value in sources.items() if path.endswith(suffix)}
        if not version.get(field) or hashes != {version[field]}:
            errors.append(f"SOP version {field} lacks matching run source evidence")
    bank = _bank_sources(config)
    if not version.get("experience_index_sha256") or (
        bank.get("execution-index.json") != version["experience_index_sha256"]
    ):
        errors.append("SOP experience index differs from frozen run bank")
    graphs = version.get("graph_sha256", {})
    if not graphs or any(bank.get(f"graphs/{key}.json") != value for key, value in graphs.items()):
        errors.append("SOP graph sources lack matching frozen bank evidence")
    return errors, version


def audit_structure(path):
    """Replay provenance and attribution invariants, not model semantic correctness."""
    path = Path(path)
    errors = []
    config = _json(path / "run.json", errors, {})
    task = config.get("args", {}).get("task", config.get("task"))
    index = _json(path / "sop-index.json", errors, {})
    version_errors, source_version = _version_errors(index, config, task)
    errors.extend(version_errors)
    ledger, malformed = read_jsonl(path / "episode-ledger.jsonl")
    if not ledger or malformed:
        errors.append("Missing or malformed execution ledger")
    versions = {}
    for plan_path in sorted((path / "plan-versions").glob("v*.json")):
        plan = _json(plan_path, errors, {})
        number = plan.get("version")
        if not isinstance(number, int) or number < 1 or number in versions:
            errors.append("Invalid or duplicate archived plan version")
            continue
        versions[number] = plan
        if plan_path.name != f"v{number:03d}.json":
            errors.append("Archived plan version filename differs")
        if plan.get("schema") != "roborag.graph-plan.v1" or plan.get("role") != "GPT_FUTURE_PLAN":
            errors.append("Plan role/schema must distinguish future plans from observations")
        if plan.get("previous_version") != (number - 1 if number > 1 else None):
            errors.append("Plan revision parent is not the previous archived version")
        if plan.get("sop_version") != source_version.get("id") or not plan.get("sop_version"):
            errors.append("Plan uses an unbound SOP version")
        if plan.get("source_versions") != source_version:
            errors.append("Plan source versions differ from its SOP index")
        phases = plan.get("phases", [])
        ids = [p.get("id") for p in phases]
        if not ids or len(ids) != len(set(ids)) or None in ids:
            errors.append("Plan phase IDs are missing or duplicated")
        pending = {p.get("id"): set(p.get("depends_on", [])) for p in phases}
        while pending:
            ready = {key for key, deps in pending.items() if not deps}
            if not ready:
                errors.append("Plan dependencies are cyclic or reference missing phases")
                break
            pending = {key: deps - ready for key, deps in pending.items() if key not in ready}
        for phase in phases:
            if (
                not phase.get("object_bindings")
                or not phase.get("goal")
                or not phase.get("expected_effects")
            ):
                errors.append("Plan phase lacks objects, goal or expected effects")
    if not versions or sorted(versions) != list(range(1, max(versions, default=0) + 1)):
        errors.append("Missing initial or intermediate archived plan version")
    latest = _json(path / "gpt-plan.json", errors, {})
    if not versions or latest != versions.get(max(versions, default=0)):
        errors.append("Current plan differs from its archived version")

    evidence, submitted, received, unknown = {}, {}, {}, set()
    active, current, sop_seen = None, None, False
    progress, last_step, known_refs, prepared = {}, {}, set(), set()
    entered_steps = {}
    counts = Counter()

    def live(ids, minimum=None, presented=False):
        if not ids or any(key not in evidence for key in ids):
            errors.append("Claim or revision lacks prior live evidence")
            return False
        if minimum is not None and any(evidence[key].get("step", -1) < minimum for key in ids):
            errors.append("Effect evidence predates the stage action")
            return False
        if presented and any(
            evidence[key].get("kind") == "observation" and not evidence[key].get("cameras")
            for key in ids
        ):
            errors.append("Model claim cites an observation never presented to the policy")
            return False
        return True

    def adoption(uses):
        for use in uses:
            if not all(use.get(key) for key in ("source", "adopted", "adaptation")):
                errors.append("Experience adoption lacks adopted operation or adaptation")
            if use.get("source") not in known_refs:
                errors.append("Experience adoption references an unretrieved source")

    for sequence, event in enumerate(ledger, 1):
        kind = event.get("kind")
        counts[kind] += 1
        if (
            event.get("id") != f"event-{sequence:05d}"
            or event.get("role") != "CURRENT_EXECUTION_RECORD"
        ):
            errors.append("Ledger event IDs/order or semantic role are invalid")
        number, stage = event.get("plan_version"), event.get("stage_id")
        if kind in ("plan_created", "plan_revised"):
            if kind == "plan_created":
                if current is not None or number != 1 or not sop_seen or not evidence:
                    errors.append(
                        "Initial graph plan must follow SOP and current observation exactly once"
                    )
            else:
                revision = event.get("revision", {})
                invalidation = revision.get("event", {})
                if (
                    current is None
                    or number != current + 1
                    or event.get("previous_version") != current
                ):
                    errors.append("Plan revision has wrong parent/version")
                if invalidation.get("kind") not in INVALIDATIONS or not invalidation.get(
                    "description"
                ):
                    errors.append("Plan revision lacks an explicit invalidating event")
                live(
                    invalidation.get("evidence_ids", []),
                    max((e.get("step", 0) for e in evidence.values()), default=0),
                    presented=True,
                )
                if not revision.get("rationale") or not revision.get("affected_phase_ids"):
                    errors.append("Plan revision lacks rationale or affected phases")
                old = {p["id"]: p for p in versions.get(current, {}).get("phases", [])}
                new = {p["id"]: p for p in versions.get(number, {}).get("phases", [])}
                changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
                affected = set(revision.get("affected_phase_ids", []))
                if (
                    not changed
                    or not changed <= affected
                    or any(progress.get(key) == "complete" for key in affected)
                ):
                    errors.append(
                        "Revision rewrites completed/unaffected phases or makes no plan change"
                    )
                if versions.get(number, {}).get("revision") != revision:
                    errors.append("Revision ledger differs from archived plan")
                old_plan, new_plan = versions.get(current, {}), versions.get(number, {})
                mutable = {"version", "previous_version", "phases", "revision"}
                if any(
                    old_plan.get(key) != new_plan.get(key)
                    for key in old_plan.keys() | new_plan.keys()
                    if key not in mutable
                ):
                    errors.append("Revision changes data outside its declared phase patch")
                for key in affected:
                    progress.pop(key, None)
                    last_step.pop(key, None)
                    entered_steps.pop(key, None)
                if active in affected:
                    active = None
            current = number
            if number not in versions:
                errors.append("Ledger plan version has no immutable archive")
            for phase in versions.get(number, {}).get("phases", []):
                adoption(phase.get("experience", []))
        elif number != current:
            errors.append("Ordinary execution event changed the persistent plan version")
        phases = {p["id"]: p for p in versions.get(current, {}).get("phases", [])}
        if kind == "sop_loaded":
            sop_seen = True
            if event.get("version") != source_version:
                errors.append("SOP loaded event differs from saved index version")
        elif kind == "stage_prepared":
            prepared.add(event.get("sop_stage"))
            known_refs.update(event.get("allowed_refs", []))
            if event.get("version_id") != source_version.get("id"):
                errors.append("Prepared context lacks matching version evidence")
            diagnostics = event.get("diagnostics", {})
            if not isinstance(diagnostics.get("cache_hit"), bool) or not isinstance(
                diagnostics.get("search_performed"), bool
            ):
                errors.append("Prepared context lacks retrieval/cache receipt")
        elif kind in ("observation_recorded", "measurement_recorded"):
            record = event.get("evidence", {})
            key = record.get("id")
            if not key or key in evidence or not isinstance(record.get("step"), int):
                errors.append("Live evidence identity/step is invalid")
            if kind == "observation_recorded":
                snapshot = Path(record.get("snapshot", ""))
                state = _json(snapshot / "state.json", errors, {})
                if state.get("step") != record.get("step"):
                    errors.append("Observation snapshot missing or disagrees with evidence step")
                hashes = record.get("sha256", {})
                if not hashes or any(
                    not (snapshot / name).is_file() or _hash(snapshot / name) != value
                    for name, value in hashes.items()
                ):
                    errors.append("Observation snapshot hashes are missing or differ")
            elif record.get("observation_id") not in evidence:
                errors.append("Measurement lacks its preceding observation")
            evidence[key] = record
        elif kind == "observation_presented":
            record = evidence.get(event.get("evidence_id"))
            if not record or record.get("kind") != "observation":
                errors.append("Presented observation lacks preceding evidence")
            else:
                record["cameras"] = sorted(
                    set(record.get("cameras", [])) | set(event.get("cameras", []))
                )
        elif kind == "stage_entered":
            phase = phases.get(stage, {})
            if not phase or phase.get("sop_stage") not in prepared:
                errors.append("Entered phase lacks a plan or prepared SOP context")
            if any(progress.get(dep) != "complete" for dep in phase.get("depends_on", [])):
                errors.append("Stage entered before its declared dependencies completed")
            if active and active != stage and progress.get(active) == "active":
                errors.append("Stage switched without assessing previous active phase")
            active = stage
            progress[stage] = "active"
            entry_step = event.get("entered_at_step")
            if not isinstance(entry_step, int):
                errors.append("Stage entry lacks its observation-step boundary")
            else:
                entered_steps[stage] = entry_step
            if event.get("revisit"):
                last_step.pop(stage, None)
        elif kind == "stage_progress_invalidated":
            cause = event.get("cause_stage")
            ancestors = set(phases.get(stage, {}).get("depends_on", []))
            while True:
                expanded = ancestors | {
                    dep for key in ancestors for dep in phases.get(key, {}).get("depends_on", [])
                }
                if expanded == ancestors:
                    break
                ancestors = expanded
            if (
                cause not in ancestors
                or progress.get(stage) != "complete"
                or event.get("previous_assessment", {}).get("status") != "complete"
            ):
                errors.append("Progress invalidation lacks a completed descendant relationship")
            progress[stage] = "pending"
            last_step.pop(stage, None)
            entered_steps.pop(stage, None)
        elif kind == "action_submitted":
            key = event.get("action_id")
            if not key or key in submitted or stage != active or progress.get(stage) != "active":
                errors.append("Action has missing, duplicate or wrong active-phase ownership")
            if not event.get("subgoal") or not event.get("expected_feedback"):
                errors.append("Action lacks its intended observable effect")
            live([event.get("observation_id")], event.get("step_before"))
            if any(key not in received for key in submitted):
                errors.append("New action submitted while previous response is unresolved")
            submitted[key] = event
        elif kind == "action_received":
            action = event.get("action", {})
            key = action.get("id")
            before = submitted.get(key, {})
            if (
                not before
                or key in received
                or any(
                    action.get(k) != before.get(k)
                    for k in ("stage_id", "plan_version", "expected_feedback")
                )
            ):
                errors.append("Action receipt has missing, duplicate or wrong attribution")
            obs = action.get("observation_id")
            live([obs], before.get("step_before"))
            receipt = action.get("receipt")
            if not isinstance(receipt, dict) or not str(action.get("effect_status", "")).startswith(
                "unknown"
            ):
                errors.append("Action receipt incorrectly claims observed effect completion")
            if (
                isinstance(receipt, dict)
                and obs in evidence
                and receipt.get("step") != evidence[obs].get("step")
            ):
                errors.append("Receipt step differs from its actual observation")
            received[key] = action
            last_step[stage] = evidence.get(obs, {}).get("step", 0)
        elif kind == "action_response_unknown":
            key = event.get("action_id")
            if key not in submitted:
                errors.append("Unknown response has no submitted action")
            unknown.add(key)
        elif kind == "stage_assessed":
            phase = phases.get(stage, {})
            checks = event.get("checks", [])
            if stage != active or not phase or not event.get("rationale"):
                errors.append("Stage assessment lacks active ownership or rationale")
            effects = [check.get("effect") for check in checks]
            if len(effects) != len(set(effects)) or not set(effects) <= set(
                phase.get("expected_effects", [])
            ):
                errors.append("Assessment checks do not name distinct plan effects")
            for check in checks:
                if check.get("status") not in ("met", "unmet", "unknown") or not check.get(
                    "observation"
                ):
                    errors.append("Assessment lacks observed status/description")
                if check.get("status") != "unknown" or check.get("evidence_ids"):
                    live(check.get("evidence_ids", []), entered_steps.get(stage), presented=True)
                if check.get("status") in ("met", "unmet"):
                    anchor = last_step.get(stage, entered_steps.get(stage))
                    if anchor is None or not any(
                        evidence.get(key, {}).get("step", -1) >= anchor
                        for key in check.get("evidence_ids", [])
                    ):
                        errors.append(
                            "Observed effect lacks a current result anchor after the latest stage action"
                        )
                if (
                    check.get("status") == "met"
                    and check.get("evidence_ids")
                    and all(
                        evidence.get(key, {}).get("kind") == "measurement"
                        and evidence[key].get("result", {}).get("plan_ok") is False
                        for key in check["evidence_ids"]
                    )
                ):
                    errors.append("Failed measurement alone cannot support a met effect")
            decision = event.get("decision")
            if decision == "complete" and (
                set(effects) != set(phase.get("expected_effects", []))
                or any(c.get("status") != "met" for c in checks)
            ):
                errors.append("Phase completed without evidence-backed met effects")
            if decision not in ("continue", "complete", "blocked"):
                errors.append("Unknown stage decision")
            progress[stage] = "active" if decision == "continue" else decision
            adoption(event.get("experience_adaptations", []))

    unresolved = sorted(set(submitted) - set(received))
    if unresolved or unknown:
        errors.append("Unresolved/lost action receipt: effect and dispatch result remain unknown")
    if counts["plan_created"] != 1 or counts["plan_revised"] != max(len(versions) - 1, 0):
        errors.append("Plan archive count differs from explicit creation/revision events")
    if not submitted or not counts["stage_assessed"]:
        errors.append("Missing action or stage-assessment execution evidence")
    commands, bad_commands = read_jsonl(path / "commands.jsonl")
    native_ids = []
    for command in commands:
        request = command.get("request", {})
        if not any(key in request for key in ("actions", "chunk")):
            continue
        attribution = request.get("execution_context", {})
        key = attribution.get("action_id")
        native_ids.append(key)
        if key not in submitted or any(
            attribution.get(k) != submitted.get(key, {}).get(k)
            for k in ("stage_id", "plan_version")
        ):
            errors.append("Native dispatched command lacks matching ledger attribution")
        if key in received and command.get("result") != received[key].get("receipt"):
            errors.append("Ledger action receipt differs from native command evidence")
    if bad_commands or len(native_ids) != len(set(native_ids)) or set(native_ids) != set(received):
        errors.append("Native command receipts are missing, duplicated or unaccounted")
    state = _json(path / "execution-state.json", errors, {})
    if (
        state.get("role") != "CURRENT_EXECUTION_STATE"
        or state.get("plan_version") != current
        or state.get("active_stage") != active
    ):
        errors.append("Execution state summary differs from ledger")
    if state.get("in_flight") is not None:
        errors.append("Execution state retains an unresolved action")
    for entry in state.get("stages", []):
        if entry.get("execution", {}).get("status") != progress.get(entry.get("id"), "pending"):
            errors.append("Execution stage status differs from ledger assessments")
    if {entry.get("id") for entry in state.get("stages", [])} != {
        phase.get("id") for phase in latest.get("phases", [])
    }:
        errors.append("Execution state omits or invents plan phases")
    preparations = [e.get("diagnostics", {}) for e in ledger if e.get("kind") == "stage_prepared"]
    calls, bad_calls = read_jsonl(path / "mcp-calls.jsonl")
    if bad_calls:
        errors.append("Malformed MCP audit records")
    tool_counts = Counter(c.get("tool") for c in calls)
    cost = {
        "initial_plans": counts["plan_created"],
        "plan_revisions": counts["plan_revised"],
        "stage_preparations": len(preparations),
        "stage_searches": sum(d.get("search_performed") is True for d in preparations),
        "stage_cache_hits": sum(d.get("cache_hit") is True for d in preparations),
        "stage_context_reuses": sum(e.get("context_reused") is True for e in ledger),
        "legacy_retrieval_calls": tool_counts["retrieve_evidence"]
        if (path / "mcp-calls.jsonl").exists() and not bad_calls
        else None,
        "reference_expansions": tool_counts["read_action_reference"]
        if (path / "mcp-calls.jsonl").exists() and not bad_calls
        else None,
        "model_request_count": None,
    }
    if not ledger or malformed:
        cost = {key: None for key in cost}
    elif any(
        not isinstance(d.get("cache_hit"), bool) or not isinstance(d.get("search_performed"), bool)
        for d in preparations
    ):
        cost["stage_searches"] = cost["stage_cache_hits"] = None
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "unresolved_actions": unresolved,
        "event_counts": dict(counts),
        "cost": cost,
        "interpretation": "Coherent recorded attribution only; model claims and references are not independently proven correct or native task success. Counts describe durable records, not model API requests.",
    }


def _run_paths(paths, task):
    result = []
    for value in paths:
        path = Path(value)
        if path.is_file():
            document = read_json(path)
            entries = document.get("proposed_paired_regression", {}).get("cases")
            if entries is None:
                entries = document.get("cases", document.get("finished", []))
            for entry in entries:
                if entry.get("task", document.get("task", task)) == task:
                    directory = entry.get(
                        "baseline_directory", entry.get("directory", entry.get("output"))
                    )
                    if directory:
                        result.append(Path(directory))
        elif (path / "run.json").is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(p for p in path.iterdir() if (p / "run.json").is_file()))
        else:
            raise ValueError(f"Missing runs/evidence path: {path}")
    if len(result) != len(set(p.resolve() for p in result)):
        raise ValueError("Duplicate run inputs; no best-retry selection")
    return result


def _row(path, task):
    """Malformed or partial evidence remains a recorded unscored attempt with costs."""
    errors = []
    config = _json(path / "run.json", errors, {})
    try:
        row = candidate_rows([path], task=task)[0]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
        args = config.get("args", {})
        agent = _json(path / "agent-exit.json", errors, {})
        result = _json(path / "result.json", errors, {})
        exit_record = _json(path / "exit.json", errors, {})
        try:
            accounting = control_accounting(path, [])
        except (KeyError, ValueError, TypeError):
            accounting = {}
        row = dict(
            seed=args.get("seed"),
            variant=args.get("variant", "standard"),
            layout=args.get("layout"),
            complete=False,
            success=None,
            progress=None,
            steps=accounting.get("native_control_steps"),
            steps_complete=accounting.get("native_control_steps_complete", False),
            wall_seconds=result.get("wall_seconds", exit_record.get("wall_seconds")),
            agent_wall_seconds=agent.get("wall_seconds"),
            usage=agent.get("accounting", {}).get("reported_usage", {}),
            model=config.get("model"),
            effort=args.get("effort"),
            directory=str(path),
            layout_sha256=config.get("layout_sha256"),
            protocol={
                key: args.get(key)
                for key in ("controller", "no_rag", "architecture", "max_wall_seconds")
            },
            failure=exit_record or None,
        )
    row["protocol"]["bank_sources"] = _bank_sources(config)
    row["evidence_errors"] = errors
    return row


def evaluate(runs, task, cases, baseline=None):
    rows = []
    for path in _run_paths(runs, task):
        row = _row(path, task)
        try:
            row["structure"] = audit_structure(path)
        except (KeyError, ValueError, TypeError, OSError) as exc:
            row["structure"] = {
                "passed": False,
                "errors": [f"Malformed staged evidence: {exc}"],
                "cost": {},
            }
        if row["protocol"].get("architecture") != "staged-v1":
            row["structure"]["passed"] = False
            row["structure"]["errors"].append("Candidate architecture is not staged-v1")
        rows.append(row)
    comparisons = []
    try:
        inventory = suite(rows, cases)
    except ValueError as exc:
        comparisons.append(str(exc))
        overall = aggregate(rows, max(len(rows), len(cases)))
        overall.update(planned=len(cases), success_rate=None)
        inventory = dict(
            overall=overall,
            cases=rows,
            by_variant={},
            pending=[c for c in cases if case_key(c) not in {case_key(r) for r in rows}],
        )
    report = dict(schema="roborag.staged-evaluation.v1", task=task, **inventory)
    coherent = len(rows) == len(cases) and all(
        r["structure"]["passed"] and not r["evidence_errors"] for r in rows
    )
    protocols = {json.dumps(r["protocol"], sort_keys=True) for r in rows}
    if len(protocols) > 1:
        comparisons.append("Mixed candidate protocols; frozen revisions must remain separate")
    scored = report["overall"]["scored"] == len(cases)
    effectiveness = None
    if baseline:
        planned = {case_key(c) for c in cases}
        before = [_row(path, task) for path in _run_paths(baseline, task)]
        before = [row for row in before if case_key(row) in planned]
        if len(before) != len(cases) or any(
            not r["complete"] or r["evidence_errors"] for r in before
        ):
            comparisons.append("Baseline lacks complete verified evidence for every declared case")
        try:
            report["paired"] = paired(before, rows)
            for pair in report["paired"]["pairs"]:
                first, after = pair["baseline"], pair["candidate"]
                for field in ("max_wall_seconds", "bank_sources"):
                    a, b = first["protocol"].get(field), after["protocol"].get(field)
                    if a is None or not a or a != b:
                        comparisons.append(
                            f"Paired {field} differs or is unknown: {case_key(after)}"
                        )
            effectiveness = (
                scored
                and not comparisons
                and all(
                    not pair["baseline"]["success"] or pair["candidate"]["success"]
                    for pair in report["paired"]["pairs"]
                )
                and report["overall"]["successes"]
                >= sum(r["success"] for r in before if r["complete"])
            )
        except (KeyError, ValueError) as exc:
            comparisons.append(str(exc))
            effectiveness = False
    report["gate"] = {
        "scope": "declared historical paired regression"
        if baseline
        else "declared development evidence only",
        "declared_cases": len(cases),
        "all_scored": scored,
        "coherent_structure": coherent,
        "comparability_errors": comparisons,
        "effectiveness_passed": effectiveness,
        "passed": scored and coherent and not comparisons and (effectiveness if baseline else True),
        "default_promotion_allowed": False,
        "interpretation": "Limited declared scope only; legacy remains default. No baseline means no effectiveness/non-regression claim.",
    }
    report["structure_cost"] = {
        key: sum(r["structure"]["cost"][key] for r in rows)
        if rows and all(isinstance(r["structure"].get("cost", {}).get(key), int) for r in rows)
        else None
        for key in (
            "initial_plans",
            "plan_revisions",
            "stage_preparations",
            "stage_searches",
            "stage_cache_hits",
            "stage_context_reuses",
            "legacy_retrieval_calls",
            "reference_expansions",
        )
    }
    report["interpretation"] = [
        "Native score is authoritative offline; partial or missing native evidence is unscored, never passed.",
        "All attempted cases and partial costs remain recorded; declared unrun cases remain pending.",
        "Cached input is part of input, reasoning output part of output; billing and API request counts unknown.",
        "Cost improvements require comparable jointly successful pairs, not early failures or shorter logs.",
        "Structure checks prove consistency of saved claims, not visual correctness, generalized success or a causal RAG gain.",
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.runs, args.task, read_json(args.cases), args.baseline)
    report["cases_file"] = str(args.cases.resolve())
    report["cases_sha256"] = _hash(args.cases)
    write_json(args.output, report)
    print(
        json.dumps(
            {"overall": report["overall"], "gate": report["gate"]}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
