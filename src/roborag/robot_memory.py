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
from .vision import VisualEncoder, distance, scene_distance


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


def render_selected_history(records, chosen, current_step, observation):
    """Keep observed intent/receipts for omitted intervals without asserting outcomes."""
    output = [observation(0)]
    last_shown, previous_end, pending = 0, 0, []
    def flush():
        if pending:
            output.append(omitted_interval("LIVE", pending[0]["action_interval"][0],
                          pending[-1]["action_interval"][1], reason="compressed_history",
                          summary={"chunks": deepcopy(pending)}))
            pending.clear()
    for record in records:
        if record["end"] > current_step:
            continue
        start, end = record["start"], record["end"]
        if start > previous_end:
            pending.append({"action_interval": [previous_end, start], "reason": "unrecorded_interval"})
        if start in chosen:
            flush()
            if start != last_shown:
                output.append(observation(start))
            output.extend(deepcopy(record["items"]))
            output.append(observation(end))
            last_shown = end
        else:
            pending.append({"action_interval": [start, end], "execution": record.get("summary")})
        previous_end = end
    if previous_end < current_step:
        pending.append({"action_interval": [previous_end, current_step], "reason": "unrecorded_interval"})
    flush()
    if last_shown != current_step:
        output.append(observation(current_step))
    return output


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
        self.multi_episode = self.corpus["schema"] == "roborag.visual-action-corpus.v2"
        if self.multi_episode:
            if self.corpus["scene_encoder"] != self.encoder.scene_name:
                raise ValueError("Initial scene encoder mismatch")
            if any(e["split"] != "train" for e in self.corpus["episodes"]):
                raise ValueError("Non-training episode in long-term bank")
            self.episodes = {e["source_sha256"]: e for e in self.corpus["episodes"]}
            if any(r["source_sha256"] not in self.episodes for r in self.corpus["records"]):
                raise ValueError("Orphan trajectory chunk")
        else:
            self.episodes = {self.corpus["source_sha256"]: self.corpus}
        if self.long_k < 2 or self.short_k < 2:
            raise ValueError("This pilot reserves at least two slots per memory")

    def reset(self):
        super().reset()
        self._feature_cache = {}
        self._train_message_cache = {}
        self._selected_source = None

    def _select_source(self):
        # Freeze the source for the live episode so retrieved transitions keep
        # one coherent arm assignment and one camera calibration.
        if self._selected_source is not None:
            return
        if self.multi_episode:
            scene = self.encoder.scene(self.archive[0]["images"]["cam_head"])
            self._source_scores = {sha: scene_distance(scene, episode["scene"])
                                   for sha, episode in self.episodes.items()}
            self._selected_source = min(self._source_scores, key=lambda sha: (self._source_scores[sha], sha))
        else:
            self._selected_source = self.corpus["source_sha256"]
            self._source_scores = {self._selected_source: 0.}
        self.source_episode = self.episodes[self._selected_source]
        records = [r for r in self.corpus["records"] if r["source_sha256"] == self._selected_source]
        self.long_records = {r["start"]: r for r in records}
        if len(self.long_records) != len(records) or not records:
            raise ValueError("Empty or duplicate intervals in selected source")

    def _feature(self, step):
        if step not in self._feature_cache:
            self._feature_cache[step] = self.encoder.encode(self.archive[step]["images"], CAMERAS)
        return self._feature_cache[step]

    def _train_chunk(self, record):
        start = record["start"]
        cache_key = record["record_id"]
        if cache_key in self._train_message_cache:
            return deepcopy(self._train_message_cache[cache_key])
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
        self._train_message_cache[cache_key] = messages
        return deepcopy(messages)

    def _request_inputs(self):
        self._select_source()
        state, feature = self.archive[self.step]["state"], self._feature(self.step)
        scores = [(distance(feature, state, record["visual"], record["before"]["state"],
                            state_weight=.2 if self.multi_episode else .5), start)
                  for start, record in self.long_records.items()]
        starts = sorted(self.long_records)
        if self.rag_mode == "fixed":
            selected_long = sorted(set(starts[i] for i in np.linspace(0, len(starts)-1, self.long_k).round().astype(int)))
        else:
            selected_long = select_diverse(scores, self.long_k, self.corpus["horizon"] * 2,
                                          anchors=(starts[0], starts[-1]))
        prefix = deepcopy(self.messages[:self.fixed_prefix_length])
        live_calibration = []
        if self.dialogue_config.include_head_calibration:
            # The upstream initializer puts runtime calibration last. Keep it
            # at the TRAIN/LIVE boundary after inserting our training context.
            block = prefix[-1]
            value = json.loads(block["content"][0]["text"])
            if value["head_camera_calibration"]["source"] != "TEST_RUNTIME":
                raise ValueError("Expected runtime calibration at the fixed-prefix boundary")
            live_calibration.append(prefix.pop())
        prefix.append(text_message("LONG-TERM TRAINING MEMORY: separate recorded episode. Adapt actions to current geometry. Gaps omit transitions. Training instruction: " + self.source_episode["instruction"]))
        calibration = calibration_content(self.source_episode["head_reference"], self.long_records[starts[0]]["before"]["head_image"],
                                          "TRAIN", "Verified source HDF5", image_profile="native_jpeg", image_layout="triptych")
        prefix.append({"role": "user", "content": [calibration]})
        end = 0
        for start in selected_long:
            if start > end:
                prefix.append(omitted_interval("TRAIN", end, start, episode_id=self._selected_source[:12]))
            prefix.extend(self._train_chunk(self.long_records[start]))
            end = self.long_records[start]["end"]
        prefix.append(text_message("END LONG-TERM MEMORY. SHORT-TERM MEMORY below belongs only to the active episode. Latest observation is authoritative."))
        prefix.extend(live_calibration)

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
        live = render_selected_history(eligible, chosen, self.step, observation)
        live.extend(deepcopy(self.messages[self.live_history_tail_start:]))
        self._log("retrieval.jsonl", {"step": self.step, "mode": self.rag_mode,
                  "corpus_sha256": self.corpus_hash, "long_selected": selected_long,
                  "source_episode_sha256": self._selected_source, "source_scene_distances": self._source_scores,
                  "retrieval_version": "scene-then-transition-v2" if self.multi_episode else "transition-v1",
                  "long_distances": {str(start): score for score, start in scores if start in selected_long},
                  "short_selected": sorted(chosen), "short_pool_size": len(eligible),
                  "latest_interval": [latest["start"], latest["end"]] if latest else None,
                  "inputs": "RGB and robot proprioception only; no scene object poses or reward"})
        return prefix + live
