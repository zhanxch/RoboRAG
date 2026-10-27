"""Unchanged v3 model-authored step plan; no graph execution or stage gates."""

from mcp.server.mcpserver.exceptions import ToolError

from roborag.episode_files import write_json

from .schemas import PlanStep


def register_tools(server, memory, read_only):
    @server.tool()
    def record_plan(current_scene: str, steps: list[PlanStep], adaptation: str) -> dict:
        """Record YOUR new plan based on current scene + retrieved evidence, before movement.

        Explain what is reused, skipped or changed from examples. Cite adopted source IDs;
        steps can differ from historical graph nodes. Call again only if your plan changes.
        """
        if not memory.retrieved:
            raise ToolError("Retrieve planning evidence before recording the new plan")
        if not steps:
            raise ToolError("Plan requires at least one step")
        for step in steps:
            if any(ref not in memory.references for ref in step.evidence_refs):
                raise ToolError("Cite only retrieved source IDs")
        memory.plan = dict(role="gpt_generated_current_plan", current_scene=current_scene,
                           steps=[s.model_dump() for s in steps], adaptation=adaptation,
                           observed_at_step=memory.state()["step"])
        memory.audit("record_plan", memory.plan, {"recorded": True})
        write_json(memory.output / "gpt-plan.json", memory.plan)
        return {"recorded": True, "steps": len(steps), "role": memory.plan["role"]}
