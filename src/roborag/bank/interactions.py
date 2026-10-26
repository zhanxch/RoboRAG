"""Optional, source-bound interaction records; no perception or motor policy.

An annotated bank can expose the recorded choice AFTER a phase's feedback and
rank by explicit roles/stages. Old banks retain their original retrieval path.
Relations describe image left/right, never world coordinates. Missing or
ambiguous query relations remain unknown rather than being guessed.
"""

from __future__ import annotations

import re

STAGES = {
    "approach": r"接近|靠近|准备接触|approach",
    "transport": r"长距离|跨桌|大幅平移|推移到另一侧|持续推|transport",
    "handoff": r"换手|换臂|接手|交接|handoff|hand.?over|switch arm",
    "reposition": r"换.*推点|改变接触|换接触|重新接触|重新夹|换位|regrip|reposition",
    "align": r"微调|修正|小幅|残差|精确对齐|fine.?align|residual|correction",
    "retreat": r"撤回|归位|回原|撤离|retreat|return.*home",
}
OBJECT = r"(?:[红蓝紫]色|蓝紫色)?T形(?:积木|块)|(?:红色?|蓝色?|蓝紫色?)块|积木|物块"
TARGET = r"(?:薄?灰色?T形(?:目标)?(?:薄垫|垫)|灰垫|灰色目标|目标垫|目标区域)"


def _side(text, entity):
    # Only clauses explicitly associating a side with this entity. A wrist,
    # arm, object heading ('长杆向右') or a future goal is not its location.
    found = set()
    for clause in re.split(r"[，。；,;]", text):
        patterns = [
            rf"(?:{entity})(?:位于|在)(?:桌面|木桌|中央)?(?:偏)?([左右])",
            rf"(?:桌面|木桌上|桌上|中央)?(?:偏)?([左右])(?:侧|边|方|前方|后方|半部|下方|上方)?(?:有|是|放着)(?:薄)?(?:{entity})",
        ]
        for pattern in patterns:
            found.update(re.findall(pattern, clause))
    return {"左": "left", "右": "right"}[next(iter(found))] if len(found) == 1 else None


def query_conditions(observation, goal, arm="", mode="plan"):
    result = {}
    source_side, target_side = _side(observation, OBJECT), _side(observation, TARGET)
    if source_side:
        result["object_side"] = source_side
    if source_side and target_side and source_side != target_side:
        result["target_direction"] = target_side
    # Relative entity relations also work away from the table centre.
    relative = re.findall(rf"(?:{OBJECT})(?:在|位于)(?:{TARGET})(?:的)?([左右])", observation)
    if len(set(relative)) == 1:
        result["target_direction"] = "right" if relative[0] == "左" else "left"
    if arm:
        result["arm"] = arm
    if mode == "action":
        # Specific intent wins over general approach/alignment words. Current
        # observation is deliberately not a future-action search field.
        for stage in ("handoff", "reposition", "retreat", "align", "transport", "approach"):
            if re.search(STAGES[stage], goal, re.I):
                result["stage"] = stage
                break
    return result


def condition_score(record, conditions, mode):
    metadata = record.get("interaction", {})
    reasons, score = [], 0.0
    keys = ("object_side", "target_direction") if mode == "plan" else ("stage",)
    for key in keys:
        wanted, actual = conditions.get(key), metadata.get(key)
        if wanted and actual:
            match = wanted == actual
            score += 0.9 if match else -0.9
            reasons.append(f"交互条件 {key}: {'一致' if match else '不同'} ({actual})")
    arm = conditions.get("arm")
    if mode == "plan" and arm and metadata.get("starting_arm"):
        match = arm == metadata["starting_arm"]
        score += 0.6 if match else -0.6
        reasons.append(f"起始操作臂: {'一致' if match else '不同'}")
    if mode == "action" and arm and metadata.get("handoff_to"):
        match = arm == metadata["handoff_to"]
        score += 0.8 if match else -0.8
        reasons.append(f"接手臂: {'一致' if match else '不同'}")
    return score, reasons


def interaction_card(phase, phases):
    """Only recorded cue/action/feedback and its actual chronological successor."""
    metadata = phase.get("interaction")
    if not metadata:
        return None
    card = dict(
        source=phase["id"],
        interval=[phase["start"], phase["end"]],
        stage=metadata["stage"],
        active_arms=metadata["active_arms"],
        before=phase["cue"],
        operation=phase["action"],
        observed_after=phase["feedback"],
    )
    if metadata.get("caution"):
        card["evidence_limit"] = metadata["caution"]
    if metadata.get("handoff_to"):
        card["handoff_to"] = metadata["handoff_to"]
    if phase.get("next"):
        successor = next(
            p
            for p in phases
            if p["graph_id"] == phase["graph_id"] and p["node_id"] == phase["next"]["id"]
        )
        card["recorded_next_choice"] = dict(
            source=successor["id"], before=successor["cue"], operation=successor["action"]
        )
    return card


def plan_interactions(record, phases):
    metadata = record.get("interaction")
    if not metadata:
        return None
    local = [p for p in phases if p["graph_id"] == record["graph_id"]]
    selected = [next(p for p in local if p["node_id"] == node) for node in metadata["plan_phases"]]
    return dict(
        conditions={
            key: metadata[key]
            for key in ("object_side", "target_direction", "starting_arm")
            if key in metadata
        },
        preparation_limit=metadata["preparation_limit"],
        decisions=[interaction_card(p, local) for p in selected],
        semantics="TRAIN annotations + recorded chronological successors; not causal proof, "
        "a prescribed plan, verified contact or independent task-success labels. "
        "Select by the current interaction, not episode order. Source intervals are end-exclusive.",
    )
