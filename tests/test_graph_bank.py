"""Exercise graph construction, retrieval, and graph-bank HTTP integration."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

from roborag.graph_bank import graph_documents, make_graph, publish_bank, search_documents, write_json
from roborag.graph_server import BankStore, make_handler


def example(task="stack_blocks", graph_id="stack-0", label="抓取紫色积木"):
    meta = dict(id=graph_id, task=task, episode=0, instruction="Stack blocks", path="demo.hdf5",
                sha256="example", frames=50, frequency=25, cameras=["cam_head"])
    annotation = dict(title="叠放示范", summary="抓取后放下", phases=[
        dict(title=label, summary="建立抓持", objects=["紫色积木"], start_frame=0, end_frame=20,
             cue="积木在桌面", action="闭合夹爪并抬起", feedback="物体随夹爪移动"),
        dict(title="放下积木", summary="完成放置", objects=["紫色积木"], start_frame=20, end_frame=50,
             cue="积木被夹持", action="下降后释放", feedback="积木留在目标位置")])
    return make_graph(meta, annotation, "test-model")


class GraphBankTest(unittest.TestCase):
    def test_chain_and_local_trees_keep_recorded_windows(self):
        graph = example()
        self.assertEqual(len(graph["nodes"]), 8)
        self.assertEqual([e for e in graph["edges"] if e["kind"] == "next"],
                         [dict(source="p01", target="p02", kind="next", label="下一阶段")])
        docs = graph_documents(graph)
        self.assertIn("物体随夹爪移动", docs[0]["text"])
        self.assertEqual(docs[0]["start_frame"], 0)
        self.assertEqual(docs[1]["start_frame"], 20)

    def test_chinese_retrieval_and_task_filter(self):
        docs = graph_documents(example()) + graph_documents(example("deposit_coin", "coin-0", "抓取硬币"))
        results = search_documents(docs, "抓取硬币", task="deposit_coin")
        self.assertEqual(results[0]["graph_id"], "coin-0")
        self.assertEqual(results[0]["node_id"], "p01")
        self.assertTrue(all(r["task"] == "deposit_coin" for r in results))
        self.assertEqual(search_documents(docs, "unseenword"), [])
        self.assertEqual(search_documents(docs, ""), [])

    def test_http_bank_graph_search_and_export_share_the_same_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            graph = example()
            write_json(root / "graphs/stack-0.json", graph)
            publish_bank(root)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(BankStore(root, root)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                def get(path):
                    with urlopen(base + path) as response:
                        return json.load(response)
                self.assertEqual(get("/api/bank")["counts"]["phases"], 2)
                self.assertEqual(get("/api/graph?id=stack-0"), graph)
                self.assertEqual(get("/api/export?id=stack-0"), graph)
                self.assertEqual(get("/api/search?q=Stack")["results"][0]["graph_id"], "stack-0")
                with self.assertRaises(HTTPError) as error:
                    get("/api/graph?id=missing")
                self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
