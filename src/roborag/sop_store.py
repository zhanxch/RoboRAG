"""Standalone manipulation SOP examples and explicitly archived experience windows.

No trajectory loader, graph-bank fallback, online annotation or execution policy.
The original lexical retriever operates on the retained historical text only.
"""

import copy
import hashlib
import json
import re
from pathlib import Path

from .graph_retrieval import GraphRetriever, held_object, normalize, support_object

SCHEMA = "roborag.manipulation-sop-package.v1"


def default_bank(root, task):
    # The default fold reference retains all v3 information. The compact SOP
    # archive remains an explicit opt-in; it must not limit recovery queries.
    from .rag_tasks import GENERALIZATION_TASKS
    if task in GENERALIZATION_TASKS:
        return Path(root) / "artifacts/generalization-v3-full" / task
    return Path(root) / ("artifacts/fold-v3-full" if task == "fold_clothes" else "data/graph-bank/v1")


def open_evidence_bank(directory, data_root):
    if (Path(directory) / "manifest.json").exists():
        return SOPStore(directory)
    from .graph_server import BankStore
    return BankStore(directory, data_root)


def evidence_source_paths(directory):
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        store = SOPStore(directory)
        return [directory / "manifest.json", *(store.path(p) for p in store.manifest["files"])]
    return [directory / "execution-index.json", *sorted((directory / "graphs").glob("*.json"))]


def sop_prompt(prompt):
    return prompt.replace("Retrieve historical evidence in plan mode", "Retrieve manipulation SOP demonstrations in plan mode").replace(
        "use include_actions=true only if exact rows are needed.",
        "this SOP package stores measured summaries and selected images, without raw action arrays.")


def build_text_index(catalog):
    graphs, phases, chunks = [], [], {}
    for sop in catalog["sops"]:
        overview, meta = sop["overview"], sop["retrieval_metadata"]
        trace = overview["episode_trace"]
        graphs.append(dict(id=sop["id"], graph_id=sop["id"], task=catalog["task"],
            task_name=meta["task_name"], title=meta["title"], instruction=meta["instruction"],
            initial_cue=overview["historical_initial_scene"],
            reported_outcome=overview["historical_reported_outcome"], episode_trace=trace))
        for i, step in enumerate(trace):
            node = step["id"]
            if node not in sop["phase_texts"]:
                continue
            text = sop["phase_texts"][node]
            action = normalize(text["operation"])
            phases.append(dict(id=sop["id"]+":"+node, graph_id=sop["id"], task=catalog["task"],
                task_name=meta["task_name"], node_id=node, title=step["recorded_step"], summary="",
                start=text["phase_interval"][0] if text["phase_interval"] else 0,
                end=text["phase_interval"][1] if text["phase_interval"] else 0,
                cue=text["before"], action=text["operation"], feedback=text["phase_feedback"],
                held_object=held_object(text["before"]), support_object=support_object(text["before"]),
                arms=[side for side, pattern in (("left", r"左爪|左臂"), ("right", r"右爪|右臂")) if re.search(pattern, action)],
                previous=dict(id=trace[i-1]["id"], title=trace[i-1]["recorded_step"]) if i else None,
                next=dict(id=trace[i+1]["id"], title=trace[i+1]["recorded_step"]) if i+1 < len(trace) else None))
    for window in [*catalog["windows"].values(), *catalog["unexpanded_window_refs"]]:
        chunks[window["id"]] = {key: window[key] for key in (
            "id", "graph_id", "node_id", "start", "end", "frequency", "action_space")}
    return dict(graphs=graphs, phases=phases,
        chunks=sorted(chunks.values(), key=lambda c: (c["graph_id"], c["node_id"], c["start"])),
        method="v3 lexical retrieval over retained SOP text; no full-trajectory access")


class SOPRetriever(GraphRetriever):
    def __init__(self, store):
        self.store = store
        super().__init__(build_text_index(store.catalog), None, None)

    def chunk(self, chunk_id):
        raise ValueError("Raw action arrays are not stored in this SOP package; expand an available experience section.")

    def retrieve(self, request):
        if request.get("state"):
            raise ValueError("This text-only SOP retrieval index does not contain unexpanded window states")
        if request.get("include_actions"):
            raise ValueError("Raw action arrays are not stored in this SOP package")
        result = super().retrieve(request)
        packet = json.loads(result["model_context"])
        packet = self.store.context(packet)
        budget = result["diagnostics"]["context_budget_chars"]
        while packet["memories"] and len(json.dumps(packet, ensure_ascii=False, separators=(",", ":"))) > budget:
            packet["memories"].pop()
        result["model_context"] = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
        included = {m["source"] for m in packet["memories"]}
        for entry in result["results"]:
            source = entry["graph_id"] + (":"+entry["node_id"] if entry["node_id"] else "")
            entry["included_in_context"] = source in included
        result["diagnostics"].update(packed=len(included), context_chars=len(result["model_context"]),
                                     context_bytes=len(result["model_context"].encode()))
        return result


class SOPStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported SOP package schema")
        for name, expected in self.manifest["files"].items():
            if hashlib.sha256(self.path(name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"SOP package checksum mismatch: {name}")
        self.catalog = json.loads(self.path("catalog.json").read_text())
        self.sops = {s["id"]: s for s in self.catalog["sops"]}
        self._retriever = SOPRetriever(self)

    def path(self, name):
        relative = Path(name)
        path = (self.directory / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.directory):
            raise ValueError("SOP package path must stay inside the package")
        if name not in self.manifest["files"]:
            raise ValueError(f"File not declared in SOP package: {name}")
        return path

    def retriever(self):
        return self._retriever

    def sections(self, source):
        graph, separator, node = source.partition(":")
        if graph not in self.sops:
            raise ValueError("SOP not found in this package")
        return sorted(self.sops[graph]["experiences"].get(node, {})) if separator else []

    def context(self, packet):
        """Only change reference organization/availability, never historical prose."""
        packet = copy.deepcopy(packet)
        for memory in packet["memories"]:
            source = memory["source"]
            memory["kind"] = "manipulation_sop_demonstration"
            if "episode_trace" in memory:
                memory["initial_image"] = None  # no initial frame was actually expanded
                for phase in memory["episode_trace"]:
                    phase["available_sections"] = self.sections(source+":"+phase["id"])
            else:
                memory["available_sections"] = self.sections(source)
                reference = memory["action_reference"]
                reference["available"] = reference["id"] in self.catalog["windows"]
                reference["url"] = ("sop://experience/"+reference["id"]) if reference["available"] else None
        return packet

    def reference(self, source, section, *, include_actions=False, images=True):
        if include_actions:
            raise ValueError("Raw action arrays are not stored in this SOP package; use include_actions=false")
        available = self.sections(source)
        if section not in available:
            raise ValueError(f"Experience section {source}/{section} is not stored in this SOP package. "
                             f"Available sections: {', '.join(available) or 'none'}. No archive fallback.")
        graph, node = source.split(":", 1)
        sop = self.sops[graph]
        selection = sop["experiences"][node][section]
        transition = json.loads(self.path(self.catalog["windows"][selection["window_id"]]["file"]).read_text())
        text = sop["phase_texts"][node]
        packet = dict(role="HISTORICAL_REFERENCE", source=source, phase_interval=text["phase_interval"],
            window_section=section, window_selection=selection["window_selection"], before=text["before"],
            operation=text["operation"], phase_feedback=text["phase_feedback"], transition=transition)
        illustrations = []
        if images:
            for label in ("before", "after"):
                state = transition[label]
                image = self.path(state["images"]["cam_head"]).read_bytes()
                illustrations.append((f"HISTORICAL {source} {label} frame {state['frame']}", image))
        return packet, illustrations
