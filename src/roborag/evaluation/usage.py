"""Auditable consumption accounting; missing usage is unknown, never zero."""

from __future__ import annotations

import json
from pathlib import Path

USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def read_events(path):
    events, malformed = [], 0
    path = Path(path)
    if not path.exists():
        return events, malformed
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed process can leave its final JSONL record unfinished.
            malformed += 1
    return events, malformed


def event_accounting(events, malformed=0):
    turns = [event for event in events if event.get("type") == "turn.completed"]
    usages = [turn["usage"] for turn in turns if isinstance(turn.get("usage"), dict)]
    known = {
        key: sum(value[key] for value in usages if isinstance(value.get(key), int))
        for key in USAGE_FIELDS
    }
    totals = {
        key: known[key]
        if usages and all(isinstance(value.get(key), int) for value in usages)
        else None
        for key in USAGE_FIELDS
    }
    complete = (
        bool(turns)
        and len(usages) == len(turns)
        and not malformed
        and not any(event.get("type") == "turn.failed" for event in events)
    )
    return dict(
        reported_usage=totals,
        known_token_lower_bounds=known,
        usage_reported_turns=len(usages),
        completed_native_turns=len(turns),
        terminal_usage_available=complete,
        malformed_event_lines=malformed,
        model_request_count=None,
        model_request_count_note="Native CLI turn totals aggregate multiple model/tool round trips; a turn is not one API request.",
        token_note="Cached input is part of input; reasoning output is retained as a breakdown, not added again to output.",
    )


def resource_accounting(samples):
    energy_wh, energy_seconds = 0.0, 0.0
    for previous, current in zip(samples, samples[1:]):
        dt = current["epoch"] - previous["epoch"]
        a, b = previous.get("gpu_power_w"), current.get("gpu_power_w")
        if dt > 0 and dt <= 30 and isinstance(a, (int, float)) and isinstance(b, (int, float)):
            energy_wh += (a + b) * 0.5 * dt / 3600
            energy_seconds += dt
    owned_peak = max(
        (
            sum(
                p["rss_bytes"]
                for processes in s.get("owned_processes", {}).values()
                for p in processes
            )
            for s in samples
        ),
        default=0,
    )
    return dict(
        sample_count=len(samples),
        sampled_wall_seconds=(samples[-1]["epoch"] - samples[0]["epoch"]) if samples else None,
        peak_gpu_memory_mib=max((int(s["gpu"].split(",")[1]) for s in samples), default=None),
        peak_simultaneous_owned_rss_bytes=owned_peak
        if any("owned_processes" in s for s in samples)
        else None,
        estimated_gpu_board_energy_wh=energy_wh if energy_seconds else None,
        power_sample_coverage_seconds=energy_seconds,
        energy_note="Trapezoidal integration of sampled total GPU-board watts, including unrelated workloads; not whole-computer energy or experiment-only billing.",
    )
