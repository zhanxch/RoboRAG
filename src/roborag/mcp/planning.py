"""Register planning tools for the current episode."""

from mcp.server.mcpserver.exceptions import ToolError

from roborag.io import write_json
from roborag.mcp.descriptions import tool_description
from roborag.mcp.schemas import PlanStep


def register_tools(server, memory, read_only):
    @server.tool(description=tool_description("record_plan"))
    def record_plan(current_scene: str, steps: list[PlanStep], adaptation: str) -> dict:
        if not memory.retrieved:
            raise ToolError("Retrieve planning evidence before recording the new plan")
        if not steps:
            raise ToolError("Plan requires at least one step")
        for step in steps:
            if any(ref not in memory.references for ref in step.evidence_refs):
                raise ToolError("Cite only retrieved source IDs")
        memory.plan = dict(
            role="gpt_generated_current_plan",
            current_scene=current_scene,
            steps=[s.model_dump() for s in steps],
            adaptation=adaptation,
            observed_at_step=memory.state()["step"],
        )
        memory.audit("record_plan", memory.plan, {"recorded": True})
        write_json(memory.output / "gpt-plan.json", memory.plan)
        return {"recorded": True, "steps": len(steps), "role": memory.plan["role"]}
