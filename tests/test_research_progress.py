import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from roborag.research_progress import Conflict, ROOT, Store, historical_experiments, validate


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

server_module = script("serve_research_progress")
restore_module = script("restore_research_progress")


def task(title="测试研究任务"):
    return dict(title=title, track="rag", status="todo", priority="P0", goals=["ai"], dependencies=[])


def run(**overrides):
    record = dict(title="独立实验", task="stack_blocks", method="graph", protocol="frozen-v1", condition="standard",
                  model="frozen-model", effort="medium", date="2026-10-24", status="completed", split="test",
                  transport="mcp", usage="complete", success=True, completion=100, input_tokens=1000,
                  output_tokens=100, cached_tokens=800, reasoning_tokens=70, wall_seconds=12,
                  mcp_server="roborag", mcp_version="test-version", trace="runs/test/trace.json", evidence="runs/test/result.json")
    record.update(overrides)
    return record


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "ledger.sqlite3"
        self.store = Store(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_seed_only_once_and_reopen_persistence(self):
        self.assertEqual(len(self.store.state()["tasks"]), 16)
        t = self.store.save("tasks", task())
        t["status"] = "active"
        t = self.store.save("tasks", t, t["id"])
        reopened = Store(self.path)
        saved = next(r for r in reopened.state()["tasks"] if r["id"] == t["id"])
        self.assertEqual(saved["status"], "active")
        self.assertEqual(saved["revision"], 2)
        self.assertEqual(len(reopened.state()["tasks"]), 17)

    def test_stale_write_and_delete_do_not_overwrite(self):
        t = self.store.save("tasks", task())
        updated = self.store.save("tasks", {**t, "title": "new"}, t["id"])
        with self.assertRaises(Conflict):
            self.store.save("tasks", t, t["id"])
        with self.assertRaises(Conflict):
            self.store.delete("tasks", t["id"], 1)
        self.assertEqual(updated["revision"], 2)

    def test_dependency_cycle_unknown_and_delete_guard(self):
        a = self.store.save("tasks", task("a"))
        b = self.store.save("tasks", {**task("b"), "dependencies": [a["id"]]})
        with self.assertRaisesRegex(ValueError, "环"):
            self.store.save("tasks", {**a, "dependencies": [b["id"]]}, a["id"])
        with self.assertRaisesRegex(ValueError, "不存在"):
            self.store.save("tasks", {**task(), "dependencies": ["missing"]})
        with self.assertRaisesRegex(ValueError, "依赖"):
            self.store.delete("tasks", a["id"], 1)
        self.store.delete("tasks", b["id"], 1)
        self.store.delete("tasks", a["id"], 1)
        with self.store.connect() as db:
            snapshot = json.loads(db.execute("SELECT snapshot FROM activity ORDER BY seq DESC LIMIT 1").fetchone()[0])
        self.assertEqual(snapshot["before"]["id"], a["id"])

    def test_linked_notes_guard_and_invalid_reference(self):
        t = self.store.save("tasks", task())
        note = dict(title="note", body="text", kind="finding", date="2026-10-24", task_id=t["id"])
        self.store.save("notes", note)
        with self.assertRaisesRegex(ValueError, "关联"):
            self.store.delete("tasks", t["id"], 1)
        with self.assertRaisesRegex(ValueError, "不存在"):
            self.store.save("notes", {**note, "task_id": "missing"})

    def test_usage_subsets_not_counted_twice(self):
        clean = validate("experiments", run())
        self.assertEqual(clean["total_tokens"], 1100)
        for patch in [{"cached_tokens": 1001}, {"reasoning_tokens": 101}, {"total_tokens": 1900}, {"input_tokens": -1}, {"output_tokens": 0.5}, {"wall_seconds": float('nan')}]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate("experiments", run(**patch))

    def test_unknown_zero_and_partial_outcome(self):
        clean = validate("experiments", run(status="interrupted", success=None, completion=None, usage="partial", output_tokens=None, reasoning_tokens=None))
        self.assertIsNone(clean["total_tokens"])
        self.assertIsNone(clean["success"])
        zero = validate("experiments", run(input_tokens=0, output_tokens=0, cached_tokens=0, reasoning_tokens=0, success=False, completion=0))
        self.assertEqual(zero["total_tokens"], 0)
        self.assertIs(zero["success"], False)
        with self.assertRaisesRegex(ValueError, "未完成"):
            validate("experiments", run(status="interrupted"))
        with self.assertRaisesRegex(ValueError, "完整任务成功"):
            validate("experiments", run(completion=20))
        with self.assertRaises(ValueError):
            validate("experiments", run(usage="unknown"))

    def test_mcp_completed_requires_protocol_and_trace(self):
        for field in ["mcp_server", "mcp_version", "trace", "evidence"]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "MCP"):
                validate("experiments", run(**{field: ""}))
        record = self.store.save("experiments", run())
        self.assertEqual(record["total_tokens"], 1100)

    def test_historical_counts_and_boundaries(self):
        history, sources = historical_experiments(ROOT)
        self.assertEqual(len(history), 36)
        self.assertEqual(len(sources), 3)
        hexashell = [r for r in history if r["protocol"] == "hexashell-public-v2"]
        self.assertEqual(len(hexashell), 20)
        self.assertEqual(sum(r["success"] for r in hexashell), 8)
        self.assertAlmostEqual(sum(r["completion"] for r in hexashell) / 20, 51)
        self.assertEqual(sum(r["total_tokens"] for r in hexashell), 11540197 + 60516)
        self.assertTrue(all(r["transport"] == "legacy" and r["readonly"] for r in history))
        self.assertEqual(sum(r["status"] == "planned" for r in history), 3)
        self.assertEqual(sum(r["status"] == "interrupted" for r in history), 2)
        self.assertTrue(all(r["success"] is None and r["completion"] is None for r in history if r["status"] != "completed"))
        with self.assertRaises(KeyError):
            self.store.save("experiments", run(), hexashell[0]["id"])

    def test_json_restore_roundtrip_without_overwrite(self):
        self.store.save("tasks", task("exported"))
        self.store.save("experiments", run())
        state = self.store.state()
        destination = Path(self.temp.name) / "restore.sqlite3"
        restored = restore_module.restore(state, destination).state()
        for collection in ["tasks", "notes", "experiments"]:
            self.assertEqual(state[collection], restored[collection])
        with self.assertRaises(FileExistsError):
            restore_module.restore(state, destination)
        invalid = copy.deepcopy(state)
        invalid["tasks"][0]["dependencies"] = ["missing"]
        target = Path(self.temp.name) / "invalid.sqlite3"
        with self.assertRaises(ValueError):
            restore_module.restore(invalid, target)
        self.assertFalse(target.exists())

    def test_csv_has_bom_escapes_formulas_and_preserves_unknown(self):
        value = {"historical": [], "experiments": [dict(id="x", title="=danger()", success=False, completion=0)]}
        output = server_module.csv_export(value).decode("utf-8-sig")
        self.assertIn("'=danger()", output)
        self.assertIn("False,0", output)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.server = server_module.make_server(Store(Path(cls.temp.name) / "test.sqlite3"), port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def request(self, path, method="GET", payload=None, headers=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(self.url + path, data, method=method, headers=headers or {})
        try:
            with urlopen(req, timeout=3) as response:
                return response.status, response.read(), response.headers
        except HTTPError as error:
            return error.code, error.read(), error.headers

    def test_http_crud_and_conflict(self):
        headers = {"Content-Type": "application/json", "X-RoboRAG-Client": "research-progress", "Origin": self.url}
        status, body, _ = self.request("/api/tasks", "POST", task("HTTP test"), headers)
        self.assertEqual(status, 201)
        t = json.loads(body)
        status, _, _ = self.request("/api/tasks/" + t["id"], "PUT", {**t, "status": "review"}, headers)
        self.assertEqual(status, 200)
        self.assertEqual(self.request("/api/tasks/" + t["id"], "PUT", t, headers)[0], 409)
        self.assertEqual(self.request("/api/tasks/" + t["id"], "DELETE", {"revision": 2}, headers)[0], 200)

    def test_writes_reject_cross_origin_and_no_client_header(self):
        self.assertEqual(self.request("/api/tasks", "POST", task(), {"Content-Type": "application/json"})[0], 403)
        headers = {"Content-Type": "application/json", "X-RoboRAG-Client": "research-progress", "Origin": "http://example.com"}
        self.assertEqual(self.request("/api/tasks", "POST", task(), headers)[0], 403)
        self.assertEqual(self.request("/api/state", headers={"Host": "example.com"})[0], 403)

    def test_frontend_exports_and_evidence_boundaries(self):
        status, html, headers = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("研究工作台".encode(), html)
        self.assertIn("Content-Security-Policy", headers)
        for path in ["/app.js", "/style.css", "/metrics.js", "/api/state", "/api/export", "/api/export?format=csv", "/api/evidence?path=docs/architecture.md"]:
            self.assertEqual(self.request(path)[0], 200, path)
        for path in ["/local.runtime.json", "/../local.runtime.json", "/api/evidence?path=docs/../local.runtime.json", "/api/evidence?path=/etc/passwd"]:
            self.assertEqual(self.request(path)[0], 404, path)


if __name__ == "__main__":
    unittest.main()
