#!/usr/bin/env python3
"""Restore a research JSON export into a NEW SQLite file; never overwrite a ledger."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from roborag.research_progress import COLLECTIONS, SCHEMA, Store, check_dependencies, validate


def restore(payload, destination):
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("不是支持的 RoboRAG 研究进度备份")
    prepared = {}
    for collection in COLLECTIONS:
        records = payload.get(collection)
        if not isinstance(records, list):
            raise ValueError(f"备份缺少 {collection}")
        prepared[collection] = []
        seen = set()
        for record in records:
            clean = validate(collection, record)
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id or len(record_id) > 100 or record_id in seen:
                raise ValueError("记录 ID 为空、重复或无效")
            if record.get("readonly"):
                raise ValueError("历史来源不能导入手动记录集合")
            seen.add(record_id)
            revision = record.get("revision")
            if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
                raise ValueError("版本号无效")
            for key in ("created_at", "updated_at"):
                if not isinstance(record.get(key), str):
                    raise ValueError("备份缺少时间字段")
            clean.update({key: record[key] for key in ["id", "revision", "created_at", "updated_at"]})
            prepared[collection].append(clean)
    check_dependencies(prepared["tasks"])
    task_ids = {t["id"] for t in prepared["tasks"]}
    for collection in ("notes", "experiments"):
        if any(r["task_id"] and r["task_id"] not in task_ids for r in prepared[collection]):
            raise ValueError("关联任务不存在")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids accidental overwrites, including concurrent ones.
    with destination.open("xb"):
        pass
    store = Store(destination)
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM records")
        for collection, records in prepared.items():
            for record in records:
                db.execute("INSERT INTO records VALUES (?,?,?)", (collection, record["id"], json.dumps(record, ensure_ascii=False)))
    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    parser.add_argument("--db", type=Path, required=True, help="新的数据库路径；已有文件将拒绝覆盖")
    args = parser.parse_args()
    try:
        store = restore(json.loads(args.backup.read_text(encoding="utf-8")), args.db)
    except (OSError, ValueError) as error:
        parser.exit(1, f"恢复未执行：{error}\n")
    print(f"Restored: {store.path}\nRun: python3 scripts/serve_research_progress.py --db '{store.path}'")
