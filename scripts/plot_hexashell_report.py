#!/usr/bin/env python3
"""Render scientific figures from a completed, verified HexaShell public suite."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "results/hexashell-public-v2-20261023"


def main():
    report = json.loads((ROOT / "results/hexashell-consumption-20261023.json").read_text())
    assert report["public_suite_complete"], "Wait for all 20 native-scored episodes"
    assert report["primary_evidence_audit"]["completed_action_logs_complete"]
    batch = next(b for b in report["batches"] if b["category"] == "primary")
    assert batch["complete"] and batch["scored_episodes"] == 20
    status = json.loads((ROOT / batch["directory"] / "status.json").read_text())
    records = sorted(status["finished"], key=lambda r: r["started_epoch"])
    figure_data = dict(protocol=report["protocol"], local=report["split_summaries"],
        author=report["reference_split_summaries"], timeline=[{
            key: r[key] for key in ("variant", "layout", "started_epoch", "ended_epoch")
        } for r in records], batch_wall_seconds=batch["batch_wall_seconds"],
        note="One trial per layout. Development reference predates recipes/notes; local uses final prompt. Reconstructed executor; hidden-seed official evaluation is a separate protocol.")
    data_path = PREFIX.with_suffix(".figure-data.json")
    data_path.write_text(json.dumps(figure_data, ensure_ascii=False, indent=2)+"\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    keys = [f"{split}/{variant}" for split in ("development", "new_layouts") for variant in ("standard", "random")]
    labels = ["Development\nstandard", "Development\nclutter", "New layouts\nstandard", "New layouts\nclutter"]
    x, width = np.arange(4), .34
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    fig.suptitle("HexaShell cloth folding: public seed-0 layouts (20 episodes)", fontsize=14)
    for ax, metric, ylabel in zip(axes, ("success", "progress"), ("Success rate (%)", "Mean native progress (0-100)")):
        for offset, name, color, source in ((-width/2,"Author public results","#9B9B9B","reference_split_summaries"),
                                             (width/2,"Local reproduction","#28699D","split_summaries")):
            groups = [report[source][key] for key in keys]
            values = [g["successful_episodes"]*20 if metric=="success" else g["mean_progress"] for g in groups]
            bars = ax.bar(x+offset, values, width, color=color, label=name)
            text = [f"{g['successful_episodes']}/5" if metric=="success" else f"{v:g}" for g,v in zip(groups,values)]
            ax.bar_label(bars, labels=text, padding=3, fontsize=9)
        ax.set(xticks=x, xticklabels=labels, ylim=(0,115), ylabel=ylabel)
        ax.set_yticks(range(0,101,20))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.supxlabel("One trial/layout; n=5 per group. Development prompts differ (see report).\nSources: native RoboDojo results and HexaFuture/HexaShell public JSON; local run started 2026-10-23 UTC.", fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(f"{PREFIX}-comparison.{extension}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    start = min(r["started_epoch"] for r in records)
    for index, record in enumerate(records):
        left = (record["started_epoch"]-start)/60
        duration = (record["ended_epoch"]-record["started_epoch"])/60
        ax.barh(index, duration, left=left, color="#28699D" if record["variant"]=="standard" else "#8C8C8C", height=.65)
    ax.set(yticks=range(len(records)), yticklabels=[f"{r['variant']} L{r['layout']}" for r in records],
           xlabel="Minutes since first episode launch", ylabel="Independent episode",
           title="Observed execution timeline: maximum concurrency 2")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.supxlabel(f"Source: scheduler start/end records (5-second polling); batch elapsed {batch['batch_wall_seconds']/60:.1f} min.\nBars include initialization and actor execution. No matched serial run: this is not a measured speedup.", fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(f"{PREFIX}-timeline.{extension}", dpi=180)
    plt.close(fig)
    print(json.dumps(dict(figure_data=str(data_path), sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                         figures=[f"{PREFIX}-{name}.png" for name in ("comparison", "timeline")]), ensure_ascii=False))


if __name__ == "__main__":
    main()
