"""Deterministic protocol fixture, not a physics simulator or robot benchmark."""
from copy import deepcopy
import math
from uuid import uuid4

from .device import DeviceError, description, opening01, position3


class SyntheticDevice:
    def __init__(self, *, reject_first_move=False):
        self.session_id = uuid4().hex
        self.seq = 0
        self.position = [0.0, 0.0, 0.3]
        self.opening = 1.0
        self.block = [0.2, 0.0, 0.05]
        self.target = [0.4, 0.0, 0.05]
        self.held = False
        self.stopped = False
        self.reject_first_move = reject_first_move

    def describe(self):
        return description("synthetic-arm", ["left"], "fixture_world", backend="synthetic",
                           limits={"workspace_xyz": [[-1, 1], [-1, 1], [0, 1]],
                                   "stop": "latched session stop; no physical hardware"})

    def observe(self):
        return deepcopy({"session_id": self.session_id, "state_seq": self.seq,
                         "source": "synthetic_fixture", "frame": "fixture_world",
                         "arms": {"left": {"position": self.position,
                                           "gripper_commanded": self.opening}},
                         "objects": {"block": self.block, "target": self.target},
                         "held": self.held, "stopped": self.stopped, "images": []})

    def _check(self, arm):
        if self.stopped:
            raise DeviceError("session is stopped; create a new device session")
        if arm != "left":
            raise ValueError("unknown arm")

    def _receipt(self, skill, before):
        return {"session_id": self.session_id, "skill": skill, "status": "completed",
                "start_seq": before, "end_seq": self.seq, "executed_steps": self.seq - before,
                "task_success": None}

    def move(self, arm, position):
        self._check(arm)
        position = position3(position)
        if not (-1 <= position[0] <= 1 and -1 <= position[1] <= 1 and 0 <= position[2] <= 1):
            raise ValueError("outside fixture workspace")
        if self.reject_first_move:
            self.reject_first_move = False
            raise DeviceError("injected rejection before motion; executed_steps=0; reobserve and retry")
        before = self.seq
        self.position = position
        if self.held:
            self.block = position.copy()
        self.seq += 1
        return self._receipt("move", before)

    def gripper(self, arm, opening):
        self._check(arm)
        opening = opening01(opening)
        before = self.seq
        if opening < 0.5 and self.opening >= 0.5:
            self.held = math.dist(self.position, self.block) < 0.025
        if opening >= 0.5:
            self.held = False
        self.opening = opening
        self.seq += 1
        return self._receipt("gripper", before)

    def stop(self):
        self.stopped = True
        return {"session_id": self.session_id, "status": "stopped", "state_seq": self.seq}
