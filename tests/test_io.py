"""Shared file protocol must publish whole documents under concurrent writers."""

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from roborag.io import write_json


class JsonPublicationTests(unittest.TestCase):
    def test_concurrent_writers_do_not_share_temporary_file(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "state.json"
            barrier = threading.Barrier(2, timeout=5)
            replace = Path.replace
            sources = []

            def simultaneous_replace(source, target):
                sources.append(source)
                barrier.wait()
                return replace(source, target)

            values = [{"writer": i, "payload": "完整" * 2000} for i in range(2)]
            with patch.object(Path, "replace", simultaneous_replace), ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(write_json, destination, value) for value in values]
                for future in futures:
                    future.result(timeout=10)
            self.assertEqual(len(set(sources)), 2)
            self.assertIn(json.loads(destination.read_text()), values)
            self.assertEqual(list(Path(folder).iterdir()), [destination])

    def test_invalid_payload_or_failed_replace_preserves_previous_document(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "state.json"
            write_json(destination, {"previous": True})
            with self.assertRaises(ValueError):
                write_json(destination, {"invalid": float("nan")})
            with patch.object(Path, "replace", side_effect=OSError("disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    write_json(destination, {"new": True})
            self.assertEqual(json.loads(destination.read_text()), {"previous": True})
            self.assertEqual(list(Path(folder).iterdir()), [destination])
