"""RoboICL feedback and lossless observation transport, without model calls."""
import base64
from copy import deepcopy
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from roborag.cap.mcp_client import ToolError


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"), "optional RoboICL image dependencies")
class RoboICLMCPTests(unittest.TestCase):
    def result(self, step=0):
        import numpy as np
        from PIL import Image
        rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        buffer = BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG")
        state = {f"{s}_arm_joint_state": [0.] * 6 for s in ("left", "right")}
        state.update({f"{s}_ee_joint_state": [1.] for s in ("left", "right")})
        return {"structuredContent": {"state": state, "instruction": "fold", "cameras": ["head"],
                                       "head_calibration_context": {"source": "TEST_RUNTIME"}},
                "content": [{"type": "image", "mimeType": "image/png",
                             "data": base64.b64encode(buffer.getvalue()).decode()}]}, rgb

    def test_lossless_image_and_metadata_round_trip(self):
        import numpy as np
        from roborag.cap.roboicl import observation
        result, rgb = self.result()
        obs, calibration = observation(result)
        np.testing.assert_array_equal(obs["vision"]["head"]["color"], rgb)
        self.assertEqual(obs["state"], result["structuredContent"]["state"])
        self.assertEqual(calibration, {"source": "TEST_RUNTIME"})
        result["content"][0]["mimeType"] = "image/jpeg"
        with self.assertRaisesRegex(ValueError, "lossless"):
            observation(result)

    def test_rejected_prefix_feedback_and_native_terminal_limit(self):
        from roborag.cap.roboicl import run_episode
        result, _ = self.result()
        class MCP:
            step, rejected, finished = 0, False, False
            def call_tool(self, name, arguments=None):
                if name == "episode_status":
                    return {"structuredContent": {"step": self.step, "max_episode_steps": 2, "ended": self.step == 2}}
                if name == "read_observation": return deepcopy(result)
                if name == "control_step":
                    if not self.rejected:
                        self.rejected = True
                        raise ToolError({"isError": True, "content": [{"type": "text", "text": "IK rejected"}],
                                         "structuredContent": {"controller_event": {"type": "action_rejected_ik_continuity"}}})
                    self.step += 1
                    return {"structuredContent": {"executed_steps": 1}}
                if name == "finish_evaluation":
                    if self.step != 2: raise RuntimeError("early scoring")
                    self.finished = True
                    return {"structuredContent": {"native_completed": True}}
                raise AssertionError(name)
        class Policy:
            def reset(self): self.observations, self.rejections, self.finalized = [], [], False
            def update_obs(self, obs): self.observations.append(obs)
            def get_action(self): return [result["structuredContent"]["state"]] * 3
            def notify_action_rejection(self, value): self.rejections.append(value)
            def finalize(self): self.finalized = True
        class Tracking:
            def observe(self, receipt): return False
        feedback = types.ModuleType("roboicl.policy.action_feedback")
        feedback.TrackingGuard, feedback.tracking_receipt = Tracking, lambda a, b: {}
        policy_module = types.ModuleType("roboicl.policy.astra_policy")
        policy_module.plain = deepcopy
        config = types.SimpleNamespace(include_head_calibration=True, max_turns=None)
        mcp, policy = MCP(), Policy()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {
            "roboicl.policy.action_feedback": feedback, "roboicl.policy.astra_policy": policy_module}):
            summary = run_episode(mcp, policy, config, Path(tmp))
            logs = [json.loads(x) for x in (Path(tmp) / "mcp-calls.jsonl").read_text().splitlines()]
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["turns"], 2)
        self.assertTrue(mcp.finished and policy.finalized)
        self.assertEqual([o["astra_meta"]["step"] for o in policy.observations], [0, 1, 2])
        self.assertEqual(policy.rejections[0]["event"]["discarded_commands"], 3)
        self.assertEqual(policy.rejections[0]["step"], 0)
        self.assertEqual(sum(e["tool"] == "control_step" and e["status"] == "returned" for e in logs), 2)


if __name__ == "__main__":
    unittest.main()
