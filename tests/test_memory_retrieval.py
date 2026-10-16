from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from roborag import EpisodeMemory, LongTermMemory, Query, build_long_term, retrieve_context
from roborag.demo import synthetic_experience


class DualMemoryTests(unittest.TestCase):
    def setUp(self):
        self.seed = synthetic_experience()
        self.long = LongTermMemory("test-v1", (self.seed,))
        self.live = EpisodeMemory("eval-001")
        self.query = Query("eval-001", 10, self.seed.robot, self.seed.action_space,
                           self.seed.task, self.seed.stage, self.seed.observation_before)

    def live_record(self, **changes):
        base = replace(self.seed, record_id="live-1", episode_id=self.live.episode_id,
                       source="live", split="online")
        return replace(base, **changes)

    def test_snapshot_round_trip_and_exclusive_creation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            original = build_long_term([self.seed], path, version="v1")
            loaded = LongTermMemory.load(path)
            self.assertEqual(original, loaded)
            self.assertEqual(original.sha256, loaded.sha256)
            with self.assertRaises(FileExistsError):
                build_long_term([self.seed], path, version="v2")
            self.assertEqual(loaded.sha256, LongTermMemory.load(path).sha256)

    def test_long_term_excludes_online_and_heldout_data(self):
        for record in (self.live_record(), replace(self.seed, split="test"),
                       replace(self.seed, split="validation")):
            with self.subTest(source=record.source, split=record.split):
                with self.assertRaises(ValueError):
                    LongTermMemory("v1", (record,))

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            LongTermMemory("v1", (self.seed, self.seed))
        self.live.append(self.live_record())
        with self.assertRaises(ValueError):
            self.live.append(self.live_record(start_step=1, end_step=2))

    def test_live_writes_do_not_change_long_term(self):
        before = self.long.sha256
        self.live.append(self.live_record())
        self.assertEqual(before, self.long.sha256)
        self.assertEqual(len(self.long.records), 1)
        with self.assertRaises(FrozenInstanceError):
            self.long.version = "changed"

    def test_episode_isolation_and_lifecycle(self):
        with self.assertRaises(ValueError):
            self.live.append(self.live_record(episode_id="another-episode"))
        with self.assertRaises(ValueError):
            self.live.append(self.seed)
        with self.assertRaises(ValueError):
            retrieve_context(self.long, self.live, replace(self.query, episode_id="other"))
        self.live.append(self.live_record())
        self.live.close()
        self.assertEqual(len(self.live), 0)
        with self.assertRaises(RuntimeError):
            self.live.append(self.live_record())
        with self.assertRaises(RuntimeError):
            retrieve_context(self.long, self.live, self.query)
        fresh = EpisodeMemory("eval-002")
        self.assertEqual(fresh.records_before(100), ())

    def test_future_and_overlapping_interactions(self):
        past = self.live_record()
        future = self.live_record(record_id="future", start_step=11, end_step=12)
        self.live.append(past)
        self.live.append(future)
        context = retrieve_context(self.long, self.live, self.query)
        self.assertEqual(context.latest_interaction, past)
        self.assertNotIn(future, context.short_term_retrieved)
        with self.assertRaises(ValueError):
            self.live.append(self.live_record(record_id="overlap", start_step=11, end_step=13))

    def test_retrieves_old_relevant_failure_and_preserves_irrelevant_latest(self):
        early = self.live_record(outcome="failure", receipt="Observed collision")
        middle = self.live_record(record_id="middle", start_step=1, end_step=2,
                                  task="inspect camera", stage="inspect",
                                  observation_before="camera occluded")
        latest = replace(middle, record_id="latest", start_step=2, end_step=3)
        for item in (early, middle, latest):
            self.live.append(item)
        context = retrieve_context(self.long, self.live, self.query, long_term_k=1, short_term_k=2)
        self.assertEqual(context.short_term_retrieved, (early,))
        self.assertEqual(context.latest_interaction, latest)
        self.assertEqual(context.short_term_retrieved[0].outcome, "failure")
        self.assertNotIn(latest, context.short_term_retrieved)

    def test_compatibility_and_same_episode_exclusion(self):
        bad_robot = replace(self.seed, record_id="bad-robot", robot="other")
        bad_space = replace(self.seed, record_id="bad-space", action_space="different-frame")
        same_episode = replace(self.seed, record_id="same-episode", episode_id=self.query.episode_id)
        memory = LongTermMemory("v1", (bad_robot, bad_space, same_episode, self.seed))
        context = retrieve_context(memory, self.live, self.query, long_term_k=10)
        self.assertEqual(context.long_term_examples, (self.seed,))
        self.live.append(self.live_record(robot="other"))
        with self.assertRaises(ValueError):
            retrieve_context(memory, self.live, self.query)

    def test_empty_results_and_zero_long_term_budget(self):
        unrelated = replace(self.query, task="xyz", stage="abc", observation="def")
        context = retrieve_context(self.long, self.live, unrelated)
        self.assertEqual(context.long_term_examples, ())
        self.assertEqual(context.short_term_retrieved, ())
        self.assertIsNone(context.latest_interaction)
        context = retrieve_context(self.long, self.live, self.query, long_term_k=0)
        self.assertEqual(context.long_term_examples, ())

    def test_short_context_is_chronological_and_budgeted(self):
        for i, record_id in enumerate(("z-first", "a-second", "latest")):
            self.live.append(self.live_record(record_id=record_id, start_step=i, end_step=i + 1))
        context = retrieve_context(self.long, self.live, self.query, short_term_k=3)
        self.assertEqual([r.record_id for r in context.short_term_retrieved], ["z-first", "a-second"])
        minimal = retrieve_context(self.long, self.live, self.query, short_term_k=1)
        self.assertEqual(minimal.short_term_retrieved, ())
        self.assertEqual(minimal.latest_interaction.record_id, "latest")

    def test_invalid_budgets_and_actions(self):
        for budgets in ({"long_term_k": -1}, {"short_term_k": 0}, {"short_term_k": 1.5}):
            with self.assertRaises(ValueError):
                retrieve_context(self.long, self.live, self.query, **budgets)
        for actions in ((), ((float("nan"),),), ((1,), (1, 2))):
            with self.assertRaises(ValueError):
                replace(self.seed, action_chunk=actions)


if __name__ == "__main__":
    unittest.main()
