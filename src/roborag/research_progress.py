"""Local research ledger: durable records, optimistic revisions, source adapters."""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import uuid

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "roborag.research-progress.v1"
COLLECTIONS = {"tasks", "experiments", "notes"}
TRACKS = {"rag", "plan", "mcp"}
TASK_STATES = {"todo", "active", "review", "done", "blocked"}
RUN_STATES = {"planned", "running", "completed", "interrupted", "error"}


def now():
    return datetime.now(timezone.utc).isoformat()


class Conflict(ValueError):
    pass


def string(data, key, required=False, limit=12000):
    value = data.get(key, "")
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ValueError(f"{key} 必须是{'非空' if required else ''}文本（最多 {limit} 字符）")
    return value.strip()


def choice(data, key, values):
    value = string(data, key, True, 100)
    if value not in values:
        raise ValueError(f"{key} 不在支持的取值范围内")
    return value


def number(data, key, maximum=None, integer=False):
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} 必须是非负数或 null")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} 不能大于 {maximum}")
    if integer and (int(value) != value or value > 2**53 - 1):
        raise ValueError(f"{key} 必须是安全范围内的整数")
    return int(value) if integer else value


def date_field(data, key, required=False):
    value = string(data, key, required, 10)
    if value:
        date.fromisoformat(value)
    return value


def validate(collection, data):
    if not isinstance(data, dict):
        raise ValueError("记录必须为 JSON 对象")
    if collection == "tasks":
        result = {k: string(data, k, k == "title", 300 if k in {"title", "owner"} else 12000)
                  for k in ["title", "description", "acceptance", "evidence", "next_step", "blocker", "owner"]}
        result.update(track=choice(data, "track", TRACKS), status=choice(data, "status", TASK_STATES),
                      priority=choice(data, "priority", {"P0", "P1", "P2"}), due=date_field(data, "due"))
        for key in ("goals", "dependencies"):
            values = data.get(key, [])
            if not isinstance(values, list) or len(values) > 100 or any(not isinstance(v, str) or len(v) > 100 for v in values):
                raise ValueError(f"{key} 必须是文本列表")
            result[key] = list(dict.fromkeys(values))
        if not set(result["goals"]) <= {"ai", "cost", "sota"}:
            raise ValueError("未知研究目标")
    elif collection == "notes":
        result = {k: string(data, k, k in {"title", "body"}) for k in ["title", "body", "task_id"]}
        result.update(kind=choice(data, "kind", {"update", "decision", "finding", "blocker"}),
                      date=date_field(data, "date", True))
    elif collection == "experiments":
        result = {k: string(data, k, k in {"title", "task", "method", "protocol", "model", "effort", "condition"})
                  for k in ["title", "task", "method", "protocol", "model", "effort", "condition", "layout", "seed", "task_id",
                            "evidence", "notes", "mcp_server", "mcp_version", "trace", "bank_version", "code_version"]}
        result.update(date=date_field(data, "date", True), status=choice(data, "status", RUN_STATES),
                      split=choice(data, "split", {"development", "validation", "test", "unknown"}),
                      transport=choice(data, "transport", {"mcp", "legacy", "unknown"}),
                      usage=choice(data, "usage", {"complete", "partial", "unknown"}))
        success = data.get("success")
        if success is not None and not isinstance(success, bool):
            raise ValueError("success 必须为 true / false / null")
        result["success"] = success
        for key in ["wall_seconds", "retrieval_seconds", "execution_seconds", "completion", "input_tokens", "output_tokens",
                    "cached_tokens", "reasoning_tokens", "total_tokens"]:
            result[key] = number(data, key, maximum=100 if key == "completion" else None, integer=key.endswith("tokens"))
        if result["status"] != "completed" and (success is not None or result["completion"] is not None):
            raise ValueError("未完成回合的最终成功与完成度必须留空")
        if success is True and result["completion"] != 100:
            raise ValueError("完整任务成功时完成度应为 100")
        for subset, total in [("cached_tokens", "input_tokens"), ("reasoning_tokens", "output_tokens")]:
            if result[subset] is not None and result[total] is not None and result[subset] > result[total]:
                raise ValueError(f"{subset} 不能大于 {total}")
        if result["input_tokens"] is not None and result["output_tokens"] is not None:
            total = result["input_tokens"] + result["output_tokens"]
            if result["total_tokens"] is not None and result["total_tokens"] != total:
                raise ValueError("总 token 必须等于输入 + 输出；缓存和推理不能重复相加")
            result["total_tokens"] = total
        if result["usage"] == "complete" and result["total_tokens"] is None:
            raise ValueError("完整用量需要输入 + 输出或已知总 token")
        if result["usage"] == "unknown" and any(result[k] is not None for k in ["input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "total_tokens"]):
            raise ValueError("已有 token 数时请将用量完整性设为完整或已知下限")
        if result["transport"] == "mcp" and result["status"] == "completed":
            if not all(result[k] for k in ["mcp_server", "mcp_version", "trace", "evidence"]):
                raise ValueError("MCP 完整回合需要 server、协议版本、trace 与结果证据")
        result["readonly"] = False
    else:
        raise ValueError("未知记录类型")
    return result


def check_dependencies(tasks):
    by_id = {task["id"]: task for task in tasks}
    visited, visiting = set(), set()
    def visit(key):
        if key in visiting:
            raise ValueError("任务依赖不能形成环")
        if key in visited:
            return
        if key not in by_id:
            raise ValueError(f"依赖任务不存在：{key}")
        visiting.add(key)
        for dependency in by_id[key]["dependencies"]:
            visit(dependency)
        visiting.remove(key)
        visited.add(key)
    for key in by_id:
        visit(key)


class Store:
    def __init__(self, path, root=ROOT):
        self.path = Path(path)
        self.root = Path(root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS records (collection TEXT, id TEXT, body TEXT NOT NULL, PRIMARY KEY(collection,id))")
            db.execute("CREATE TABLE IF NOT EXISTS activity (seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, collection TEXT, record_id TEXT, action TEXT, title TEXT, snapshot TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM metadata WHERE key='initialized'").fetchone():
                seed = json.loads((self.root / "docs/research-progress.seed.json").read_text())
                for collection in ("tasks", "notes"):
                    for record in seed[collection]:
                        clean = validate(collection, record)
                        clean.update({k: record[k] for k in ["id", "revision", "created_at", "updated_at"]})
                        db.execute("INSERT INTO records VALUES (?,?,?)", (collection, record["id"], json.dumps(clean, ensure_ascii=False)))
                db.execute("INSERT INTO metadata VALUES ('initialized',?)", (now(),))

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def records(db, collection):
        return [json.loads(row[0]) for row in db.execute("SELECT body FROM records WHERE collection=? ORDER BY id", (collection,))]

    def state(self):
        with self.connect() as db:
            # One read transaction for a consistent multi-collection snapshot.
            db.execute("BEGIN")
            result = {key: self.records(db, key) for key in COLLECTIONS}
            result["activity"] = [dict(zip(["seq", "at", "collection", "record_id", "action", "title"], row)) for row in
                                  db.execute("SELECT seq,at,collection,record_id,action,title FROM activity ORDER BY seq DESC LIMIT 60")]
        external, sources = historical_experiments(self.root)
        result.update(schema=SCHEMA, sources=sources, generated_at=now(), storage=str(self.path), historical=external)
        return result

    def save(self, collection, data, record_id=None):
        clean = validate(collection, data)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if record_id is not None:
                row = db.execute("SELECT body FROM records WHERE collection=? AND id=?", (collection, record_id)).fetchone()
                if row is None:
                    raise KeyError("记录不存在，历史来源记录只读")
                old = json.loads(row[0])
                if data.get("revision") != old["revision"]:
                    raise Conflict("记录已被其他页面修改，请刷新后重试；当前输入仍保留")
            else:
                record_id = {"tasks": "TASK", "experiments": "RUN", "notes": "NOTE"}[collection] + "-" + uuid.uuid4().hex[:10]
                old = None
            clean.update(id=record_id, revision=old["revision"] + 1 if old else 1,
                         created_at=old["created_at"] if old else now(), updated_at=now())
            if collection == "tasks":
                check_dependencies([t for t in self.records(db, "tasks") if t["id"] != record_id] + [clean])
            if collection in {"notes", "experiments"} and clean["task_id"]:
                if not db.execute("SELECT 1 FROM records WHERE collection='tasks' AND id=?", (clean["task_id"],)).fetchone():
                    raise ValueError("关联任务不存在")
            db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)", (collection, record_id, json.dumps(clean, ensure_ascii=False)))
            db.execute("INSERT INTO activity(at,collection,record_id,action,title,snapshot) VALUES (?,?,?,?,?,?)",
                       (now(), collection, record_id, "updated" if old else "created", clean["title"], json.dumps({"before": old, "after": clean}, ensure_ascii=False)))
        return clean

    def delete(self, collection, record_id, revision):
        if collection not in COLLECTIONS:
            raise ValueError("未知记录类型")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM records WHERE collection=? AND id=?", (collection, record_id)).fetchone()
            if row is None:
                raise KeyError("记录不存在，历史来源记录只读")
            old = json.loads(row[0])
            if old["revision"] != revision:
                raise Conflict("记录已变化，请刷新后重试")
            if collection == "tasks":
                if any(record_id in t["dependencies"] for t in self.records(db, "tasks")):
                    raise ValueError("其他任务仍依赖此任务，请先移除依赖")
                if any(r["task_id"] == record_id for c in ("notes", "experiments") for r in self.records(db, c)):
                    raise ValueError("实验或日志仍关联此任务，请先解除关联")
            db.execute("DELETE FROM records WHERE collection=? AND id=?", (collection, record_id))
            db.execute("INSERT INTO activity(at,collection,record_id,action,title,snapshot) VALUES (?,?,?,?,?,?)",
                       (now(), collection, record_id, "deleted", old["title"], json.dumps({"before": old}, ensure_ascii=False)))


def historical_experiments(root):
    """Only allowlisted summaries; never scan credentials or execute experiment code."""
    runs, sources = [], []
    def read(name):
        path = root / "results" / name
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("expected object")
            sources.append({"path": f"results/{name}", "sha256": hashlib.sha256(raw).hexdigest(), "status": "loaded"})
            return data
        except (OSError, ValueError) as error:
            sources.append({"path": f"results/{name}", "status": "unavailable", "error": str(error)})
            return {}

    def base(key, source, **fields):
        record = dict(id="HIST-" + key, readonly=True, source=source, evidence=source, task_id="", date="", title=key,
                      task="", method="", protocol="", model="", effort="", condition="standard", layout="", seed="",
                      split="unknown", transport="legacy", status="planned", success=None, completion=None,
                      input_tokens=None, output_tokens=None, cached_tokens=None, reasoning_tokens=None, total_tokens=None,
                      usage="unknown", wall_seconds=None, retrieval_seconds=None, execution_seconds=None,
                      bank_version="", code_version="", mcp_server="", mcp_version="", trace="", notes="")
        record.update(fields)
        runs.append(record)

    source = "results/pilot-current.json"
    for trial in read("pilot-current.json").get("trials", []):
        s = trial["summary"]
        complete = s.get("valid_native_episode") is True and s.get("status") == "native_completed"
        base(s["run"], source, title=s["run"], date="2026-10-16", task=s["task"], method=s["mode"],
             protocol="pilot-" + trial["selector"], model=trial["model"], effort=trial["effort"],
             layout=str(s["layout"]), split=s["evaluation_role"], status="completed" if complete else "interrupted",
             success=bool(s["success_rate"]) if complete and s.get("success_rate") is not None else None,
             completion=s.get("score_0_100") if complete else None, total_tokens=s.get("known_total_tokens"),
             usage="complete" if s.get("calls_without_usage") == 0 else "partial",
             wall_seconds=(s.get("exit") or {}).get("wall_seconds"), bank_version=trial.get("corpus_sha256") or "",
             evidence=source + "\n" + (s.get("native_result") or ""),
             notes="历史双库 pilot；非 MCP 验证。单次布局结果不证明检索优势；并发耗时不是匹配延迟对照。")

    source = "results/hexashell-consumption-20261023.json"
    for case in read("hexashell-consumption-20261023.json").get("public_cases", []):
        s = case.get("execution")
        if not s:
            continue
        usage = s.get("accounting", {}).get("reported_usage", {})
        inp, out = usage.get("input_tokens"), usage.get("output_tokens")
        complete = s.get("status") == "native_completed"
        base(f"hexashell-{case['variant']}-{case['layout']}", source, title=f"HexaShell · {case['variant']} · L{case['layout']}",
             date="2026-10-23", task="fold_clothes", method="HexaShell reconstruction", protocol="hexashell-public-v2",
             model=s.get("model") or "", effort="xhigh", condition=case["variant"], layout=str(case["layout"]), seed=str(s.get("seed", "")),
             split="development" if case["split"] == "development" else "validation", status="completed" if complete else "interrupted",
             success=s.get("success") if complete else None, completion=s.get("progress") if complete else None,
             wall_seconds=s.get("wall_seconds"), input_tokens=inp, output_tokens=out,
             cached_tokens=usage.get("cached_input_tokens"), reasoning_tokens=usage.get("reasoning_output_tokens"),
             total_tokens=inp + out if inp is not None and out is not None else None,
             usage="complete" if s.get("accounting", {}).get("terminal_usage_available") else "partial",
             evidence=source + "\n" + (s.get("native_result") or ""),
             notes="外部方法复现，未启用 RoboRAG 检索；CLI 工具调用不等于 MCP。new_layouts 映射为 validation。批次并发 2；不与 pilot 直接比较。")

    source = "results/roboicl-public-comparison-20261020.json"
    for episode in read("roboicl-public-comparison-20261020.json").get("episodes", []):
        s = episode.get("evidence") or {}
        complete = s.get("valid_native_episode") is True and episode.get("status") == "native_completed"
        base(f"roboicl-{episode['task']}-{episode['layout']}", source, title=f"RoboRAG · {episode['task']} · L{episode['layout']}",
             date="2026-10-20", task=episode["task"], method="dual", protocol="roboicl-public-3task-20261020",
             model="gpt-6-astra", effort="xhigh", layout=str(episode["layout"]), split="validation",
             status="completed" if complete else ("planned" if episode.get("status") == "not_run" else "interrupted"), success=episode.get("roborag_success") if complete else None,
             completion=episode.get("roborag_progress") if complete else None, total_tokens=s.get("known_total_tokens"),
             usage="complete" if s.get("known_total_tokens") is not None and s.get("calls_without_usage") == 0 else "unknown",
             wall_seconds=(s.get("exit") or {}).get("wall_seconds"), evidence=source + "\n" + (s.get("native_result") or ""),
             notes="未完成的 9 回合公开对照；只导入各回合已有结果。外部公开基线不是本地受控消融，未运行项不记零。")
    return runs, sources
