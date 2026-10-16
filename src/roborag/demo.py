"""Synthetic storage/retrieval demonstration; never connects to a robot or model."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .memory import EpisodeMemory, LongTermMemory, build_long_term
from .retrieval import retrieve_context
from .schema import Experience, Query


def synthetic_experience() -> Experience:
    return Experience(
        record_id="train-recovery", episode_id="train-001", start_step=0, end_step=1,
        robot="synthetic-arm", action_space="synthetic-delta-xyz-gripper:base:m:v1",
        task="insert peg", stage="alignment",
        observation_before="peg offset from socket touching rim",
        action_chunk=((0.0, 0.0, 0.01, 0.0),),
        receipt="Synthetic executed retreat; observed contact released.",
        observation_after="peg above rim ready to realign",
        outcome="success", source="demonstration", split="train",
    )


def main() -> None:
    seed = synthetic_experience()
    with TemporaryDirectory(prefix="roborag-demo-") as directory:
        path = Path(directory) / "long-term-v1.json"
        build_long_term(
            [seed, replace(seed, record_id="train-grasp", episode_id="train-002",
                           task="grasp block", stage="grasp",
                           observation_before="block on table gripper open")],
            path, version="synthetic-v1",
        )
        long_term = LongTermMemory.load(path)
        live = EpisodeMemory("eval-001")
        live.append(replace(
            seed, record_id="live-earlier-contact", episode_id=live.episode_id,
            source="live", split="online", outcome="failure",
            receipt="Synthetic collision; insertion attempt failed.",
            observation_after="peg still on rim",
        ))
        live.append(replace(
            seed, record_id="live-latest", episode_id=live.episode_id,
            start_step=1, end_step=2, source="live", split="online",
            task="camera check", stage="inspection", observation_before="camera occluded",
            receipt="Synthetic camera check; no new evidence of insertion success.",
            observation_after="camera remains occluded", outcome="unknown",
        ))
        query = Query(live.episode_id, 2, seed.robot, seed.action_space,
                      "insert peg", "alignment", "peg offset from socket touching rim")
        context = retrieve_context(long_term, live, query, long_term_k=1, short_term_k=2)
        print(json.dumps(context.to_dict(), ensure_ascii=False, indent=2))
        live.close()
        print("\nSynthetic demo finished; episode memory cleared. No robot/model was called.")


if __name__ == "__main__":
    main()

