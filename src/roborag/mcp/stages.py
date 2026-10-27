"""Opt-in tools: future graph plans, prepared SOP stages and observed progress."""

from typing import Literal

from mcp.server.mcpserver.exceptions import ToolError

from roborag.execution.models import EffectCheck, ExperienceUse, GraphPhase, PlanInvalidation


def register_tools(server, memory, read_only):
    execution = memory.execution

    def invoke(name, function, *args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except (ValueError, KeyError) as exc:
            raise ToolError(str(exc)) from exc
        memory.audit(name, dict(args=args, **kwargs), result)
        return result

    @server.tool(
        annotations=read_only,
        description="Load versioned task SOP overview before graph planning. This is operational guidance, distinct from historical demonstrations and current observations. Detailed guidance arrives when a stage is prepared. No movement.",
    )
    def get_task_sop() -> dict:
        return invoke("get_task_sop", execution.bootstrap)

    @server.tool(
        annotations=read_only,
        description="Prepare one SOP stage before planning or look ahead without changing the active phase. Returns stage guidance and relevant historical examples. Reuse applicable references; use reason=gap/state_change/exception only for a concrete need to refresh retrieval. Use arm=both for a bimanual interaction; it applies no single-arm filter. Historical results are not current facts. No motion or replanning.",
    )
    def prepare_stage(
        sop_stage: str,
        observation: str = "",
        goal: str = "",
        arm: Literal["", "left", "right", "both"] = "",
        held_object: str = "",
        target_object: str = "",
        reason: Literal["preview", "entry", "gap", "state_change", "exception"] = "preview",
    ) -> dict:
        return invoke(
            "prepare_stage",
            execution.prepare,
            sop_stage,
            observation=observation,
            goal=goal,
            arm=arm,
            held_object=held_object,
            target_object=target_object,
            reason=reason,
        )

    @server.tool(
        description="Create the persistent graph plan once, after observing and get_task_sop. Bind current objects; give unique phase IDs, SOP stage IDs, observable expected_effects and explicit logical depends_on edges. Historical order is not a required dependency. experience records what was adopted and how adapted, using retrieved sources. Ordinary actions, feedback, retrieval and phase changes do not revise this plan."
    )
    def record_graph_plan(
        current_scene: str,
        object_bindings: dict[str, str],
        phases: list[GraphPhase],
        adaptation: str,
        sop_version: str,
    ) -> dict:
        return invoke(
            "record_graph_plan",
            execution.record_plan,
            current_scene,
            object_bindings,
            [p.model_dump() for p in phases],
            adaptation,
            sop_version,
        )

    @server.tool(
        description="Patch only affected future/current phases when an explicit observed event makes the plan inapplicable. Cite current observation/measurement evidence IDs and the current version. Geometry adjustment, ordinary feedback, reference adoption and stage progress do not need revision. Completed phase history is immutable; add recovery phases when needed. Omitted phases persist."
    )
    def revise_plan(
        revision_of: int,
        event: PlanInvalidation,
        replace_phases: list[GraphPhase],
        rationale: str,
        remove_phase_ids: list[str] = [],
    ) -> dict:
        return invoke(
            "revise_plan",
            execution.revise,
            revision_of,
            event.model_dump(),
            [p.model_dump() for p in replace_phases],
            remove_phase_ids,
            rationale,
        )

    @server.tool(
        description="Enter one ready phase of the current graph plan and prepare its SOP guidance/experience. Reuses previous applicable preparation. Explicit dependencies must be complete. Assess a previous active phase before switching. A rejected switch blocks movement until a ready phase is entered or the current phase is explicitly resumed with reason=exception/revisit and an observed recovery need; both require a presented observation at the current step. reason=revisit reopens a completed phase without replanning. Use prepare_stage for lookahead; it does not change action ownership."
    )
    def enter_stage(
        stage_id: str,
        observation: str = "",
        reason: Literal["entry", "gap", "state_change", "exception", "revisit"] = "entry",
    ) -> dict:
        return invoke("enter_stage", execution.enter, stage_id, observation, reason)

    @server.tool(
        description="Assess current phase expected effects using actual observation/measurement evidence IDs and describe what was seen. Checks identify effect text verbatim; status is met/unmet/unknown. Evidence may combine earlier process observations within this stage visit with a current outcome; met/unmet requires at least one observation/measurement after the latest phase action. Complete requires all effects met. Command completion, small tracking error, historical feedback and episode_over do not prove an effect or native task success. Record newly adopted experience and adaptation here without revising the plan. Continue or blocked may retain unknowns."
    )
    def assess_stage(
        stage_id: str,
        checks: list[EffectCheck],
        decision: Literal["continue", "complete", "blocked"],
        rationale: str,
        experience: list[ExperienceUse] = [],
    ) -> dict:
        return invoke(
            "assess_stage",
            execution.assess,
            stage_id,
            [c.model_dump() for c in checks],
            decision,
            rationale,
            [e.model_dump() for e in experience],
        )

    @server.tool(
        annotations=read_only,
        description="Recover compact persistent plan, active phase, progress, recent live evidence and latest action after context compaction. Same state is saved in episode-context.json. Does not observe, move, retrieve or replan. Native task success is unavailable to the online policy.",
    )
    def execution_status() -> dict:
        return invoke("execution_status", execution.status)
