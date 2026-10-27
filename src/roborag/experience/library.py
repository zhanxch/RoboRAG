"""An independent SOP/index layer; never annotates or rewrites a source bank.

Rules are exact, hash-bound paragraphs/bullets from the established task prompt.
Reviewed stage associations select historical intervals, not causal dependencies.
Only query results are cached: scene validity and adoption remain live decisions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from roborag.bank.retrieval import GraphRetriever
from roborag.tasks.registry import TASK_ROOT

SUPPORTED_TASKS = ("fold_clothes", "push_T", "pour_balls_into_vase")
SEMANTICS = {
    "sop": "operation_rules with explicit applicability; not current scene facts",
    "examples": "recorded_experience; source feedback may be unknown or occluded",
    "association": "reviewed interaction correspondence, not a mandatory sequence or causal proof",
    "planning": "GPT binds current objects and creates a separate persistent future plan",
    "execution": "live observations and effects belong in a separate execution record",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fragments(text):
    """Keep exact original paragraph/bullet text, including its punctuation."""
    result = {}
    for index, paragraph in enumerate(text.strip().split("\n\n")):
        prefix = f"p{index:02d}"
        if "\n- " in paragraph:
            heading, *bullets = paragraph.split("\n- ")
            result[prefix + ".heading"] = heading
            for number, bullet in enumerate(bullets, 1):
                result[f"{prefix}.b{number:02d}"] = "- " + bullet
        else:
            result[prefix] = paragraph
    return result


def load_sop(task):
    """Read packaged rules without needing a bank or simulator."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"No staged-v1 SOP for task: {task}")
    path = TASK_ROOT / task / "sop.json"
    sop = json.loads(path.read_text())
    prompt = TASK_ROOT / task / "prompt.txt"
    if sop.get("task") != task or sop.get("role") != "operation_rules":
        raise ValueError("SOP task/role does not match the requested task")
    if _sha(prompt) != sop["source_prompt"]["sha256"]:
        raise ValueError("Task prompt changed: publish a new SOP version before staged execution")
    fragments = _fragments(prompt.read_text())
    stages = sop["stages"]
    ids = [stage["id"] for stage in stages]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate SOP stage")
    used = set(sop["global_rule_refs"])
    for stage in stages:
        used.update(stage["rule_refs"])
        if not set(stage.get("lookahead", [])) <= set(ids):
            raise ValueError("Unknown SOP lookahead stage")
    if used != set(fragments):
        raise ValueError("SOP rules must cover every original prompt fragment without unknown refs")
    return sop


def _rule_records(task, refs):
    fragments = _fragments((TASK_ROOT / task / "prompt.txt").read_text())
    return [
        {
            "id": ref,
            "role": "operation_rule",
            "source": f"src/roborag/tasks/{task}/prompt.txt#{ref}",
            "text": fragments[ref],
        }
        for ref in refs
    ]


def _overview(sop):
    visible = ("id", "goal", "applicability", "completion", "branches", "dependencies", "lookahead")
    return {
        "role": "sop_overview",
        "task": sop["task"],
        "sop_id": sop["id"],
        "sop_version": sop["version"],
        "stage_ids": [stage["id"] for stage in sop["stages"]],
        "stages": [{key: stage[key] for key in visible if key in stage} for stage in sop["stages"]],
        "semantics": SEMANTICS,
    }


def bootstrap(task):
    sop = load_sop(task)
    return {
        "role": "sop_bootstrap",
        "overview": _overview(sop),
        "global_rules": _rule_records(task, sop["global_rule_refs"]),
        "source_prompt": sop["source_prompt"],
        "sop_sha256": _sha(TASK_ROOT / task / "sop.json"),
    }


def persistent_guidance(task):
    """Task-wide established text only; phase details arrive on stage entry."""
    return "\n\n".join(rule["text"] for rule in bootstrap(task)["global_rules"])


class StageLibrary:
    def __init__(self, bank, task):
        self.bank, self.task = bank, task
        self.sop = load_sop(task)
        self.stages = {stage["id"]: stage for stage in self.sop["stages"]}
        self._cache = {}
        self._hits = self._misses = 0
        index_path = bank.directory / "execution-index.json"
        index_hash = _sha(index_path)
        if index_hash != self.sop["experience_source"]["execution_index_sha256"]:
            raise ValueError(
                "SOP experience mappings require their declared execution-index version"
            )
        graph_hashes = {}
        self._versions = {index_path: self._stat(index_path)}
        for graph_id, expected in self.sop["experience_source"]["graphs"].items():
            path = bank.directory / "graphs" / f"{graph_id}.json"
            actual = _sha(path)
            if actual != expected:
                raise ValueError(f"SOP experience graph version mismatch: {graph_id}")
            graph_hashes[graph_id] = actual
            self._versions[path] = self._stat(path)
        self._retriever = bank.retriever()
        self.version = {
            "architecture": "staged-v1",
            "sop_id": self.sop["id"],
            "sop_version": self.sop["version"],
            "sop_sha256": _sha(TASK_ROOT / task / "sop.json"),
            "prompt_sha256": self.sop["source_prompt"]["sha256"],
            "experience_index_sha256": index_hash,
            "graph_sha256": graph_hashes,
        }
        self.version["id"] = hashlib.sha256(
            json.dumps(self.version, sort_keys=True).encode()
        ).hexdigest()
        self._sidecar = self._build_sidecar()

    @staticmethod
    def _stat(path):
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    def validate_version(self):
        """Reject stale prepared context even when a caller reuses its own cache."""
        if any(self._stat(path) != version for path, version in self._versions.items()):
            raise ValueError(
                "Source bank changed during this episode; start a new versioned session"
            )

    def _build_sidecar(self):
        by_id = {phase["id"]: phase for phase in self._retriever.index["phases"]}
        associations = []
        for stage in self.stages.values():
            for source in stage["experience_refs"]:
                phase = by_id.get(source)
                if phase is None or phase["task"] != self.task:
                    raise ValueError(f"Unknown or cross-task SOP experience source: {source}")
                associations.append(
                    {
                        "sop_stage": stage["id"],
                        "source": source,
                        "historical_interval": [phase["start"], phase["end"]],
                        "chunk_ids": [
                            chunk["id"]
                            for chunk in self._retriever.by_phase.get(
                                (phase["graph_id"], phase["node_id"]), []
                            )
                        ],
                        "association": "reviewed_interaction_correspondence",
                    }
                )
        return {
            "schema": "roborag.sop-experience-index.v1",
            "role": "stage_to_historical_evidence_index",
            "version": self.version,
            "semantics": SEMANTICS,
            "recording_sources": {
                graph_id: self.bank.graph(graph_id)["source"]
                for graph_id in self.version["graph_sha256"]
            },
            "associations": associations,
        }

    def export_index(self):
        """The caller may persist this next to episode outputs, never in the bank."""
        self.validate_version()
        return copy.deepcopy(self._sidecar)

    def overview(self):
        self.validate_version()
        return _overview(self.sop) | {"version": copy.deepcopy(self.version)}

    def bootstrap(self):
        # Global text is already in the stable initial prompt, preserving its
        # provider cache prefix; do not resend long geometry paragraphs here.
        return {
            "role": "sop_bootstrap",
            "overview": self.overview(),
            "source_prompt": copy.deepcopy(self.sop["source_prompt"]),
            "global_rule_refs": self.sop["global_rule_refs"][:],
            "global_guidance_location": "persistent initial task prompt",
            "version": copy.deepcopy(self.version),
        }

    def persistent_guidance(self):
        return persistent_guidance(self.task)

    def prepare(
        self,
        stage_id,
        *,
        goal="",
        observation="",
        arm="",
        held_object="",
        target_object="",
        known_refs=(),
        refresh=False,
    ):
        """Prepare stage guidance and query only its associated historical phases.

        A repeated semantic request is reused; explicit state-change/gap/exception
        refresh bypasses the cache. Known IDs mark context reuse, never prove that
        the historical conditions still apply to the current scene.
        """
        self.validate_version()
        if stage_id not in self.stages:
            raise ValueError(f"Unknown SOP stage: {stage_id}")
        if arm not in ("", "left", "right", "both"):
            raise ValueError("arm must be left, right, both or empty")
        # Bimanual interaction uses its stage/current-scene terms, not a
        # single-arm filter. Cache identity follows that effective query.
        effective_arm = "" if arm == "both" else arm
        stage = self.stages[stage_id]

        def normalize(value):
            return re.sub(r"\s+", " ", str(value)).strip()

        query = {
            "mode": "action",
            "task": self.task,
            "goal": normalize(goal) or stage["query"],
            "observation": normalize(observation),
            "stage": stage["query"],
            "arm": effective_arm,
            "held_object": held_object,
            "target_object": target_object,
            "k": 2,
            "max_context_chars": 14000,
        }
        key = json.dumps([self.version["id"], stage_id, query], sort_keys=True, ensure_ascii=False)
        hit = key in self._cache and not refresh
        if hit:
            self._hits += 1
            result = copy.deepcopy(self._cache[key])
        else:
            allowed = set(stage["experience_refs"])
            original = self._retriever.index
            index = original | {
                "phases": [phase for phase in original["phases"] if phase["id"] in allowed],
                "chunks": [
                    chunk
                    for chunk in original["chunks"]
                    if f"{chunk['graph_id']}:{chunk['node_id']}" in allowed
                ],
            }
            retriever = GraphRetriever(index, self.bank.graph, self.bank.source)
            retrieved = retriever.retrieve(query)
            packet = json.loads(retrieved["model_context"])
            examples = packet.get("memories", [])
            neighbors = {}
            for example in examples:
                example["role"] = "recorded_experience"
                example["sop_stage"] = stage_id
                example["association_semantics"] = SEMANTICS["association"]
                graph_id = example["source"].split(":", 1)[0]
                for relation in ("previous", "next"):
                    neighbor = example.get(relation)
                    if neighbor:
                        source = graph_id + ":" + neighbor["id"]
                        neighbors[source] = {
                            "source": source,
                            "title": neighbor["title"],
                            "relation": "historical_" + relation,
                            "relative_to": example["source"],
                            "semantics": "Recorded temporal neighbor available for context; "
                            "not a stage match, current fact or required dependency.",
                        }
            result = {
                "role": "prepared_sop_stage",
                "sop_stage": stage_id,
                "stage": {
                    key: stage[key]
                    for key in (
                        "goal",
                        "applicability",
                        "completion",
                        "branches",
                        "dependencies",
                        "lookahead",
                    )
                    if key in stage
                },
                "rules": _rule_records(self.task, stage["rule_refs"]),
                "historical_examples": examples,
                "temporal_neighbors": list(neighbors.values()),
                "allowed_refs": sorted(
                    {example["source"] for example in examples} | set(neighbors)
                ),
                "version_id": self.version["id"],
                "warnings": retrieved["warnings"],
                "diagnostics": retrieved["diagnostics"],
                "use": "Check applicability against live evidence. Record the adopted operation, "
                "current adaptation and expected effect in the execution record. An empty "
                "match allows a reasoned attempt or an explicit blocked/unknown result. "
                "Preparing or looking ahead does not revise the persistent plan.",
            }
            self._misses += 1
            self._cache[key] = copy.deepcopy(result)
        result["diagnostics"].update(
            {
                "cache_hit": hit,
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "refresh_requested": bool(refresh),
                "search_performed": not hit,
                "requested_arm": arm,
                "effective_arm": effective_arm,
                "known_refs_reused": sorted(set(known_refs) & set(result["allowed_refs"])),
                "validity": "Cache identity is not evidence that current scene conditions still hold",
            }
        )
        return result
