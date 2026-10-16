"""Real observation retrieval on top of the unchanged upstream action interface."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from roboicl.policy.dialogue_policy import Model
from roboicl.policy.astra_policy import CAMERAS, plain
from roboicl.policy.observation_images import camera_input_parts
from roboicl.policy.head_camera_prompt import calibration_content
from roboicl.policy.action_feedback import tracking_receipt
from roboicl.policy.trajectory_memory import omitted_interval
from .vision import VisualEncoder, distance


def text_message(text):
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def select_diverse(scored, count, separation, anchors=()):
    selected = list(anchors)
    for _, start in sorted(scored):
        if len(selected) >= count:
            break
        if all(abs(start - previous) >= separation for previous in selected):
            selected.append(start)
    # On short histories relax temporal diversity, while never repeating a record.
    for _, start in sorted(scored):
        if len(selected) >= count:
            break
        if start not in selected:
            selected.append(start)
    return sorted(selected[:count])


class RetrievalModel(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rag_mode = os.environ["ROBORAG_MODE"]
        self.long_k = int(os.environ.get("ROBORAG_LONG_K", "6"))
        self.short_k = int(os.environ.get("ROBORAG_SHORT_K", "4"))
        raw = Path(os.environ["ROBORAG_CORPUS"]).read_bytes()
        self.corpus_hash = hashlib.sha256(raw).hexdigest()
        self.corpus = json.loads(raw)
        if self.corpus["horizon"] != self.dialogue_config.predict_horizon:
            raise ValueError("Corpus horizon differs from executed action interface")
        if self.corpus["source_split"] != "train" or any(r["split"] != "train" for r in self.corpus["records"]):
            raise ValueError("Only training records are accepted in long-term memory")
        if self.corpus["encoder"] != VisualEncoder.name:
            raise ValueError("Corpus/query encoder mismatch")
        self.encoder = VisualEncoder()
        self.long_records = {r["start"]: r for r in self.corpus["records"]}
        if self.long_k < 2 or self.short_k < 2:
            raise ValueError("This pilot reserves at least two slots per memory")

    def reset(self):
        super().reset()
        self._feature_cache = {}
        self._train_message_cache = {}

    def _feature(self, step):
        if step not in self._feature_cache:
            self._feature_cache[step] = self.encoder.encode(self.archive[step]["images"], CAMERAS)
        return self._feature_cache[step]

    def _train_chunk(self, record):
        start = record["start"]
        if start in self._train_message_cache:
            return deepcopy(self._train_message_cache[start])
        before, after = record["before"], record["after"]
        call_id = "rag_train_" + record["record_id"].replace(":", "_")
        commands = plain(self._convert_tensor(record["actions"], before["state"]))
        c = self.dialogue_config
        def observation(value):
            content = [{"type": "input_text", "text": json.dumps({"source": "LONG_TERM_TRAIN",
                       "record_id": record["record_id"], "observation_step": value["frame"],
                       "robot_state": value["state"]})}]
            content.extend(camera_input_parts(value["frame"], value["prompt_images"], CAMERAS,
                           profile=c.image_profile, detail=c.image_detail, image_layout=c.image_layout))
            return {"role": "user", "content": content}
        receipt = {"source": "LONG_TERM_TRAIN", "executed_step_interval": [start, record["end"]],
                   "executed_steps": len(commands), "predicted_steps": len(commands), "unexecuted_steps": 0,
                   "last_submitted_command": commands[-1], "latest_tracking": tracking_receipt(commands[-1], after["state"]),
                   "result_state": after["state"], "controller_events": None, "success": None,
                   "notice": "Reconstructed recorded demonstration receipt; not measured live controller feedback."}
        messages = [observation(before), {"type": "function_call", "name": "act", "call_id": call_id,
                    "arguments": json.dumps({"actions": record["actions"], "execution_note": "Recorded training transition."})},
                    {"type": "function_call_output", "call_id": call_id, "output": json.dumps(receipt)}, observation(after)]
        self._train_message_cache[start] = messages
        return deepcopy(messages)

    def _request_inputs(self):
        state, feature = self.archive[self.step]["state"], self._feature(self.step)
        scores = [(distance(feature, state, record["visual"], record["before"]["state"]), start)
                  for start, record in self.long_records.items()]
        starts = sorted(self.long_records)
        if self.rag_mode == "fixed":
            selected_long = sorted(set(starts[i] for i in np.linspace(0, len(starts)-1, self.long_k).round().astype(int)))
        else:
            selected_long = select_diverse(scores, self.long_k, self.corpus["horizon"] * 2,
                                          anchors=(starts[0], starts[-1]))
        prefix = deepcopy(self.messages[:self.fixed_prefix_length])
        prefix.append(text_message("LONG-TERM TRAINING MEMORY: separate recorded episode. Adapt actions to current geometry. Gaps omit transitions."))
        calibration = calibration_content(self.corpus["head_reference"], self.long_records[starts[0]]["before"]["head_image"],
                                          "TRAIN", "Verified source HDF5", image_profile="native_jpeg", image_layout="triptych")
        prefix.append({"role": "user", "content": [calibration]})
        end = 0
        for start in selected_long:
            if start > end:
                prefix.append(omitted_interval("TRAIN", end, start, episode_id=self.corpus["source_sha256"][:12]))
            prefix.extend(self._train_chunk(self.long_records[start]))
            end = self.long_records[start]["end"]
        prefix.append(text_message("END LONG-TERM MEMORY. SHORT-TERM MEMORY below belongs only to the active episode. Latest observation is authoritative."))

        eligible = [r for r in self.live_memory._records if r["end"] <= self.step]
        latest = eligible[-1] if eligible else None
        history = eligible[:-1]
        short_scores = [(distance(feature, state, self._feature(r["end"]), self.archive[r["end"]]["state"]), r["start"])
                        for r in history]
        if self.rag_mode in ("fixed", "long_only"):
            selected_short = [r["start"] for r in history[-(self.short_k-1):]]
        else:
            selected_short = select_diverse(short_scores, self.short_k-1, self.corpus["horizon"]*2)
        chosen = set(selected_short + ([latest["start"]] if latest else []))
        observation = lambda step: {"role": "user", "content": self._observations([step], CAMERAS)}
        live = [observation(0)]
        last = 0
        for record in eligible:
            if record["start"] not in chosen:
                continue
            if record["start"] > last:
                live.append(omitted_interval("LIVE", last, record["start"]))
                live.append(observation(record["start"]))
            live.extend(deepcopy(record["items"]))
            live.append(observation(record["end"]))
            last = record["end"]
        if last < self.step:
            live.append(omitted_interval("LIVE", last, self.step))
            live.append(observation(self.step))
        live.extend(deepcopy(self.messages[self.live_history_tail_start:]))
        self._log("retrieval.jsonl", {"step": self.step, "mode": self.rag_mode,
                  "corpus_sha256": self.corpus_hash, "long_selected": selected_long,
                  "long_distances": {str(start): score for score, start in scores if start in selected_long},
                  "short_selected": sorted(chosen), "short_pool_size": len(eligible),
                  "latest_interval": [latest["start"], latest["end"]] if latest else None,
                  "inputs": "RGB and robot proprioception only; no scene object poses or reward"})
        return prefix + live
