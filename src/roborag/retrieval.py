"""Deterministic lexical baseline; no visual or semantic embedding claims."""

import re
from typing import Iterable

from .memory import EpisodeMemory, LongTermMemory
from .schema import Context, Experience, Query, nonnegative_int


def _terms(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _compatible(record: Experience, query: Query) -> bool:
    return record.robot == query.robot and record.action_space == query.action_space


def _rank(records: Iterable[Experience], query: Query, k: int) -> tuple[Experience, ...]:
    query_terms = _terms(f"{query.task} {query.stage} {query.observation}")
    scored = []
    for record in records:
        # Match against the candidate's pre-action situation, not a future query outcome.
        record_terms = _terms(f"{record.task} {record.stage} {record.observation_before}")
        score = len(query_terms & record_terms) / len(query_terms | record_terms)
        if score > 0:
            scored.append((score, record))
    scored.sort(key=lambda pair: (-pair[0], pair[1].record_id))
    return tuple(record for _, record in scored[:k])


def retrieve_context(
    long_term: LongTermMemory,
    short_term: EpisodeMemory,
    query: Query,
    *,
    long_term_k: int = 2,
    short_term_k: int = 3,
) -> Context:
    """short_term_k includes the latest visible interaction, reserved before ranking."""
    nonnegative_int(long_term_k, "long_term_k")
    nonnegative_int(short_term_k, "short_term_k")
    if short_term_k < 1:
        raise ValueError("short_term_k must reserve at least one slot for immediate feedback")
    if query.episode_id != short_term.episode_id:
        raise ValueError("query episode does not match short-term memory")

    visible = short_term.records_before(query.step)
    # A changed online interface requires an adapter/reset, not silent omission of feedback.
    if any(not _compatible(record, query) for record in visible):
        raise ValueError("live interaction interface differs from query interface")
    latest = visible[-1] if visible else None
    historical = visible[:-1] if latest else ()
    selected_short = _rank(historical, query, short_term_k - (1 if latest else 0))
    selected_short = tuple(sorted(selected_short, key=lambda record: record.start_step))
    selected_long = _rank(
        (record for record in long_term.records
         if _compatible(record, query) and record.episode_id != query.episode_id),
        query, long_term_k,
    )
    return Context(query, long_term.version, long_term.sha256, selected_long, selected_short, latest)

