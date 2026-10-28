"""Two-level, inspectable graph retrieval and bounded execution context.

This is a lexical + explicit condition + proprioception baseline, not a learned
semantic retriever. Phase names stay free text. Historical feedback is returned
as evidence, never used as the current observation matching field.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from typing import Literal
from urllib.parse import urlencode

from .graph_bank import tokens, search_documents, write_json, EXPERIENCE_SEMANTICS
from .graph_chunks import index_chunks, load_chunk

RetrievalArm = Literal["", "left", "right", "both", "左臂", "右臂", "双臂"]
ARM_VALUES = {"": "", "left": "left", "right": "right", "both": "",
              "左臂": "left", "右臂": "right", "双臂": ""}


def normalize_arm(arm):
    """Both arms means no single-arm filter; accept the exposed aliases."""
    if not isinstance(arm, str) or arm not in ARM_VALUES:
        raise ValueError("arm must be left, right, both, 左臂, 右臂, 双臂 or empty")
    return ARM_VALUES[arm]


ALIASES = {"黑色": ("深色", "黑块", "black"), "白色": ("白块", "white"),
           "紫色": ("紫块", "purple"), "木纹": ("木块", "wooden", "wood"),
           "抓取": ("抓住", "抓起", "夹取", "grasp", "grip"),
           "堆叠": ("叠放", "堆起来", "叠起来", "stacking", "stack"),
           "释放": ("松开", "放开", "release"), "底块": ("底座", "base"),
           "右爪": ("右夹爪", "右手", "right gripper"), "左爪": ("左夹爪", "左手", "left gripper")}
FAILURE = re.compile(r"失败|滑落|掉落|未抓住|没有抓住|抓空|failed|failure|slipp", re.I)


def normalize(text):
    text = str(text).lower()
    for canonical, variants in ALIASES.items():
        for variant in variants:
            text = text.replace(variant, canonical)
    return text


def object_key(text):
    value = normalize(text)
    for color in ("白色", "黑色", "紫色", "木纹"):
        if color in value:
            return color
    return value.strip()


def held_object(cue):
    # Only explicit carrying/holding language. '围住' and '闭合' are not proof.
    # Parse possession before normalizing action synonyms: 抓住 -> 抓取 would
    # erase the distinction between a current grasp and an intended action.
    cue = str(cue)
    for canonical in ("左爪", "右爪"):
        for variant in ALIASES[canonical]:
            cue = cue.replace(variant, canonical)
    match = re.search(r"(?:左爪|右爪|左臂|右臂)(?:已经|已)?(?:夹着|抓着|携带|携|持有|抓住了?|夹住了?)([^，。；,;]{1,18})", cue)
    return object_key(match.group(1)) if match else None


def support_object(text):
    match = re.search(r"(白色|黑色|紫色|木纹)(?:积木|木)?(?:底块|底座|块(?:的)?(?:上方|顶部|顶端))", normalize(text))
    return match.group(1) if match else None


def guidance(graph, phase):
    return {n["kind"]: n["text"] for n in graph["nodes"] if n.get("parent") == phase["id"]}


def phase_records(graph):
    phases = sorted((n for n in graph["nodes"] if n["kind"] == "phase"), key=lambda n: n["order"])
    result = []
    for i, phase in enumerate(phases):
        local = guidance(graph, phase)
        action = normalize(local.get("action", ""))
        arms = [side for side, pattern in (("left", r"左爪|左臂"), ("right", r"右爪|右臂")) if re.search(pattern, action)]
        result.append({"id": f"{graph['id']}:{phase['id']}", "graph_id": graph["id"],
                       "task": graph["task"], "task_name": graph["task_name"],
                       "node_id": phase["id"], "title": phase["title"], "summary": phase["summary"],
                       "start": phase["start_frame"], "end": phase["end_frame"],
                       "cue": local.get("cue", ""), "action": local.get("action", ""),
                       "feedback": local.get("feedback", ""),
                       "held_object": held_object(local.get("cue", "")),
                       "support_object": support_object(local.get("cue", "")), "arms": arms,
                       "previous": {"id": phases[i-1]["id"], "title": phases[i-1]["title"]} if i else None,
                       "next": {"id": phases[i+1]["id"], "title": phases[i+1]["title"]} if i+1 < len(phases) else None})
    return result


def build_index(directory, data_root, *, horizon=10, stride=5):
    directory, data_root = Path(directory), Path(data_root).resolve()
    graphs, phases, chunks = [], [], []
    for path in sorted((directory / "graphs").glob("*.json")):
        graph = json.loads(path.read_text())
        if "semantics" not in graph:
            graph["semantics"] = dict(EXPERIENCE_SEMANTICS)
            write_json(path, graph)
        local = phase_records(graph)
        source = (data_root / graph["source"]["path"]).resolve()
        if not source.is_relative_to(data_root):
            raise ValueError("Source must be inside the data directory")
        graphs.append({"id": graph["id"], "graph_id": graph["id"], "task": graph["task"],
                       "task_name": graph["task_name"], "title": graph["title"],
                       "instruction": graph["instruction"], "initial_cue": local[0]["cue"],
                       "reported_outcome": local[-1]["feedback"],
                       "episode_trace": [{"id": p["node_id"], "recorded_step": p["title"]} for p in local]})
        phases.extend(local)
        chunks.extend(index_chunks(source, graph, horizon=horizon, stride=stride))
    value = {"schema": "roborag.execution-index.v1", "built_at": datetime.now(timezone.utc).isoformat(),
             "method": "field BM25 + explicit conditions + optional start-state distance + diversity",
             "horizon": horizon, "stride": stride, "graphs": graphs, "phases": phases, "chunks": chunks,
             "counts": {"graphs": len(graphs), "phases": len(phases), "chunks": len(chunks)},
             "limitations": ["自由文本与词汇归一化检索；尚未使用学习型语义或图像嵌入。",
                             "阶段边界与文字来自自动标注；短窗口不是人工验证的技能单元。",
                             "关节/末端状态不等于物体状态；历史反馈不等于当前执行结果。"]}
    write_json(directory / "execution-index.json", value)
    return value


def field_scores(records, query, field):
    if not query.strip():
        return {}
    documents = [{"id": r["id"], "task": r["task"],
                  "terms": dict(Counter(tokens(normalize(field(r)))))} for r in records]
    hits = search_documents(documents, normalize(query), limit=len(documents))
    maximum = max((r["score"] for r in hits), default=1) or 1
    return {r["id"]: r["score"]/maximum for r in hits}


def state_distance(current, candidate, arm=""):
    """Same-frame/robot comparison only; no object geometry inferred from joints."""
    distances = []
    for side in ([arm] if arm else ["left", "right"]):
        pose_key, grip_key = f"{side}_ee_pose", f"{side}_ee_joint_state"
        if pose_key not in current:
            continue
        a, b = current[pose_key], candidate[pose_key]
        xyz = math.sqrt(sum((a[i]-b[i])**2 for i in range(3))) / .15
        norm = math.sqrt(sum(x*x for x in a[3:])) * math.sqrt(sum(x*x for x in b[3:]))
        cosine = min(1., abs(sum(a[i]*b[i] for i in range(3, 7))) / max(norm, 1e-12))
        rotation = 2*math.acos(cosine)/math.pi
        grip = abs(current[grip_key][0]-candidate[grip_key][0]) if grip_key in current else 0
        distances.append(xyz + rotation + .5*grip)
    return sum(distances)/len(distances) if distances else None


def validate_state(state):
    if not isinstance(state, dict):
        raise ValueError("state must be a JSON object")
    known = False
    for side in ("left", "right"):
        for suffix, size in (("ee_pose", 7), ("ee_joint_state", 1)):
            key = f"{side}_{suffix}"
            if key in state:
                values = state[key]
                if not isinstance(values, list) or len(values) != size or any(
                        not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
                    raise ValueError(f"{key} must have {size} finite numbers")
                if suffix == "ee_pose":
                    if sum(x*x for x in values[3:]) < 1e-12:
                        raise ValueError(f"{key} has a zero quaternion")
                    known = True
    if state and not known:
        raise ValueError("state needs at least one left/right_ee_pose (xyz+wxyz)")


def similarity(a, b):
    # Strip cosmetic object labels for duplicate-context suppression, not storage.
    def terms(value):
        value = normalize(value)
        for word in ("白色", "黑色", "紫色", "木纹", "左", "右"):
            value = value.replace(word, "")
        return set(tokens(value))
    ta, tb = terms(a), terms(b)
    return len(ta & tb) / max(1, len(ta | tb))


class GraphRetriever:
    def __init__(self, index, graph_loader, source_loader):
        self.index, self.graph_loader, self.source_loader = index, graph_loader, source_loader
        self.chunks = {c["id"]: c for c in index["chunks"]}
        self.by_phase = {}
        for chunk in index["chunks"]:
            self.by_phase.setdefault((chunk["graph_id"], chunk["node_id"]), []).append(chunk)

    def chunk(self, chunk_id):
        chunk = self.chunks[chunk_id]
        graph = self.graph_loader(chunk["graph_id"])
        return load_chunk(self.source_loader(graph), graph, chunk)

    def example_query(self, graph_id, node_id):
        record = next((r for r in self.index["phases"] if r["graph_id"] == graph_id and r["node_id"] == node_id), None)
        if record is None:
            raise KeyError("Phase not found")
        chunks = self.by_phase.get((graph_id, node_id), [])
        if not chunks:
            raise KeyError("Phase has no measured transition")
        chunk = chunks[0]
        return {"mode": "action", "task": record["task"], "goal": record["title"],
                "observation": record["cue"], "held_object": record["held_object"] or "",
                "target_object": record["support_object"] or "",
                "arm": record["arms"][0] if len(record["arms"]) == 1 else "",
                "state": chunk["before_state"], "action_space": chunk["action_space"],
                "exclude_graph_id": graph_id}

    def retrieve(self, request):
        started = time.perf_counter()
        if not isinstance(request, dict):
            raise ValueError("Request must be an object")
        mode = request.get("mode", "action")
        if mode not in ("plan", "action"):
            raise ValueError("mode must be plan or action")
        goal, observation, stage = [str(request.get(k, ""))[:1000] for k in ("goal", "observation", "stage")]
        task, arm = request.get("task", ""), normalize_arm(request.get("arm", ""))
        k = min(5, max(1, int(request.get("k", 2))))
        budget = min(50000, max(1200, int(request.get("max_context_chars", 6000))))
        current = request.get("state", {})
        validate_state(current)
        interface = request.get("action_space", "")
        warnings = []
        if current and not interface:
            warnings.append("未声明相同 action_space，未使用本体状态距离排序。")
        excluded = request.get("exclude_graph_id", "")
        records = [r for r in self.index["graphs" if mode == "plan" else "phases"]
                   if (not task or r["task"] == task) and r["graph_id"] != excluded]
        if mode == "action" and not (observation or stage or request.get("held_object")):
            warnings.append("生成动作需要当前观测或阶段；仅任务名称请使用 plan 检索。")
            records = []
        if mode == "action" and FAILURE.search(observation + stage):
            records = [r for r in records if FAILURE.search(r["cue"])]
            if not records:
                warnings.append("库中没有匹配的失败前置情境；未把普通成功示范当作恢复经验。")
        wanted_held = object_key(request["held_object"]) if request.get("held_object") else held_object(observation)
        wanted_support = object_key(request["target_object"]) if request.get("target_object") else support_object(goal) or support_object(observation)
        if mode == "action":
            records = [r for r in records if not (arm and r["arms"] and arm not in r["arms"])
                       and not (wanted_held and r["held_object"] and wanted_held != r["held_object"])
                       and not (wanted_support and r["support_object"] and wanted_support != r["support_object"])]
            for key, wanted in (("held_object", wanted_held), ("support_object", wanted_support)):
                exact = [r for r in records if r[key] == wanted] if wanted else []
                if exact:
                    records = exact
                elif wanted and records:
                    warnings.append(f"经验的 {key} 未能确认匹配；返回一般性参考，需要当前画面确认。")
        goal_scores = field_scores(records, goal + " " + stage,
            (lambda r: r["instruction"] + " " + r["title"] + " " + " ".join(p["recorded_step"] for p in r["episode_trace"]))
            if mode == "plan" else (lambda r: r["title"] + " " + r["summary"] + " " + r["action"]))
        observation_scores = field_scores(records, observation,
            (lambda r: r["initial_cue"]) if mode == "plan" else (lambda r: r["cue"]))
        ranked = []
        for record in records:
            gs, os = goal_scores.get(record["id"], 0), observation_scores.get(record["id"], 0)
            score = (.4*gs if goal.strip() or stage.strip() else 0) + (.6*os if observation else 0)
            if mode == "plan" and not (goal or observation or stage) and task:
                score = 1
            if score <= 0 and not wanted_held:
                continue
            reasons, chunk = [], None
            if gs: reasons.append("目标 / 阶段文本匹配")
            if os: reasons.append("执行前情境匹配")
            if mode == "action":
                if wanted_held and record["held_object"] == wanted_held:
                    score += .75
                    reasons.append("当前持物与经验持物一致")
                elif wanted_held:
                    reasons.append("经验未明确标注持物，需要结合画面判断")
                if wanted_support and record["support_object"] == wanted_support:
                    score += .4
                    reasons.append("目标支撑物与经验一致")
                candidates = self.by_phase.get((record["graph_id"], record["node_id"]), [])
                if interface:
                    candidates = [c for c in candidates if c["action_space"] == interface]
                if not candidates:
                    continue
                comparisons = [(state_distance(current, c["before_state"], arm), c) for c in candidates] if current and interface else []
                comparisons = [(d, c) for d, c in comparisons if d is not None]
                if comparisons:
                    distance, chunk = min(comparisons, key=lambda pair: (pair[0], pair[1]["start"]))
                    score += .2 / (1+distance)
                    reasons.append(f"同接口执行前状态距离 {distance:.3f}（越小越近）")
                else:
                    chunk = candidates[0]
                    reasons.append("未提供可比本体状态，动作窗口取阶段起点")
            ranked.append({"record": record, "chunk": chunk, "score": score, "reasons": reasons,
                           "text": " ".join(p["recorded_step"] for p in record["episode_trace"]) if mode == "plan" else record["action"]})
        selected = []
        while ranked and len(selected) < k:
            for entry in ranked:
                penalty = .15*max((similarity(entry["text"], other["text"]) for other in selected), default=0)
                entry["diverse_score"] = entry["score"] - penalty
            best = min(ranked, key=lambda e: (-e["diverse_score"], e["record"]["id"]))
            selected.append(best)
            # One reference per episode: overlapping windows cannot flood top-k.
            ranked = [r for r in ranked if r["record"]["graph_id"] != best["record"]["graph_id"]]
        results, memories = [], []
        for entry in selected:
            r, chunk = entry["record"], entry["chunk"]
            result = {"graph_id": r["graph_id"], "task_name": r["task_name"], "title": r["title"],
                      "score": round(entry["score"], 4), "reasons": entry["reasons"],
                      "node_id": r.get("node_id"), "start_frame": r.get("start", 0)}
            if mode == "plan":
                memory = {"source": r["id"], "historical_initial_scene": r["initial_cue"],
                          "episode_trace": r["episode_trace"], "historical_reported_outcome": r["reported_outcome"],
                          "evidence_status": "automatically extracted demonstration description; not independently verified"}
                memory["initial_image"] = "/api/frame?" + urlencode({"id": r["graph_id"], "frame": 0, "camera": "cam_head"})
                result.update(initial_cue=r["initial_cue"], episode_trace=r["episode_trace"], reported_outcome=r["reported_outcome"])
            else:
                chunk_ref = {"id": chunk["id"], "url": "/api/chunk?" + urlencode({"id": chunk["id"]}),
                             "interval": [chunk["start"], chunk["end"]], "frequency": chunk["frequency"],
                             "action_space": chunk["action_space"]}
                memory = {"source": r["id"], "historical_before": r["cue"], "subgoal": r["title"],
                          "demonstrated_operation": r["action"], "historical_phase_feedback": r["feedback"],
                          "previous": r["previous"], "next": r["next"], "action_reference": chunk_ref}
                memory["evidence_status"] = "phase text is automatically extracted; numerical transition and images are source recordings"
                result.update(cue=r["cue"], action=r["action"], feedback=r["feedback"], chunk=chunk_ref)
                if request.get("include_actions"):
                    memory["transition"] = self.chunk(chunk["id"])
            results.append(result)
            memories.append(memory)
        packet = {"purpose": "evidence_for_gpt_planning" if mode == "plan" else "evidence_for_gpt_action_generation",
                  "current": {"task": task, "goal": goal, "observation": observation, "stage": stage},
                  "use": "Generate a NEW plan or action chunk from the current goal, images and robot state. Use these historical records as examples/evidence and cite source IDs when adopting them. Recorded order is not a required dependency. Ground objects and coordinates in current observations; respect current controller limits. Historical feedback is not a live result and may occur after the short action window. Do not replay reference action rows directly.",
                  "memories": []}
        if wanted_held:
            packet["current"]["held_object"] = wanted_held
        if arm:
            packet["current"]["arm"] = arm
        if wanted_support:
            packet["current"]["target_object"] = wanted_support
        if current:
            packet["current"]["state"] = current
        if interface:
            packet["current"]["action_space"] = interface
        dump = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        # Never truncate a numeric action array mid-row to meet the budget.
        for result, memory in zip(results, memories):
            packet["memories"].append(memory)
            if len(dump(packet)) > budget and "transition" in memory:
                memory.pop("transition")
                result["actions_omitted_for_budget"] = True
            if len(dump(packet)) > budget:
                packet["memories"].pop()
                result["included_in_context"] = False
            else:
                result["included_in_context"] = True
        # Bound unusually long caller text too; expose omission rather than silent truncation.
        if len(dump(packet)) > budget:
            packet = {"purpose": packet["purpose"], "memories": [], "notice": "Query exceeds context budget; current observation remains in caller context."}
        if results and not packet["memories"]:
            warnings.append("上下文预算不足以放入一个完整经验；请提高预算。")
        if interface and not results and mode == "action":
            warnings.append("没有同时满足查询条件与 action_space 的动作参考。")
        context = dump(packet)
        return {"mode": mode, "results": results, "model_context": context, "warnings": warnings,
                "diagnostics": {"method": self.index["method"], "candidate_count": len(records),
                                "retrieved": len(results), "packed": len(packet["memories"]),
                                "context_chars": len(context), "context_budget_chars": budget,
                                "context_bytes": len(context.encode()), "online_model_calls": 0,
                                "elapsed_ms": round((time.perf_counter()-started)*1000, 2),
                                "score_meaning": "uncalibrated ranking score, not success probability"}}
