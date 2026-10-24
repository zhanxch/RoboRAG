#!/usr/bin/env python3
"""Export a matched robot benchmark figure and paired case CSV."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    parser.add_argument("--label",default="RoboRAG")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.report.read_text())
    baseline,candidate=report["baseline"],report["candidate"]
    if candidate["success_rate"] is None:
        raise ValueError("Wait for all planned cases to receive native scores before plotting a complete comparison")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    names=["HexaShell (local)",args.label]
    colors=["#52677d","#007f79"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.titleweight":"bold"})
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.0))
    def bars(ax,values,title,unit,labels=None):
        ax.set_title(title,loc="left",pad=16)
        ax.set_ylabel(unit)
        ax.set_xticks([0,1],names)
        if any(v is None for v in values):
            ax.text(.5,.5,"Unavailable",ha="center",va="center",transform=ax.transAxes)
            return
        ax.bar([0,1],values,color=colors,width=.55)
        ax.set_ylim(0,max(max(values)*1.25,1))
        for i,v in enumerate(values):
            ax.text(i,v+ax.get_ylim()[1]*.025,labels[i] if labels else f"{v:,.1f}",ha="center",va="bottom")
    bars(axes[0,0],[100*x["success_rate"] for x in (baseline,candidate)],
         "Native success on matched scenes","%",
         [f"{x['successes']}/{x['planned']} ({100*x['success_rate']:.0f}%)" for x in (baseline,candidate)])
    joint=report["jointly_successful"]
    bars(axes[0,1],[joint[x]["mean_control_steps"] for x in ("baseline","candidate")],
         f"Control steps: both successful (n={len(joint['cases'])})","Native dispatches")
    bars(axes[1,0],[x["mean_wall_seconds"] for x in (baseline,candidate)],
         "Episode elapsed time: all cases","Seconds")
    bars(axes[1,1],[x["uncached_input_tokens"]/x["recorded"] if x["uncached_input_tokens"] is not None else None
                    for x in (baseline,candidate)],"Uncached input per attempted case","Tokens")
    output=[x["usage"]["output_tokens"] for x in (baseline,candidate)]
    output_text=" / ".join(f"{n:,.0f}" if n is not None else "unknown" for n in output)
    fig.suptitle("Fold clothes | GPT-6 Astra / xhigh | fixed public layouts",x=.08,ha="left",fontsize=15)
    fig.text(.08,.025,f"Total output tokens (baseline / candidate): {output_text}. Cached input is included in input.\n"
             "One trial per layout; development layouts included. Different control interfaces. Dollar charges unavailable.",
             fontsize=8,color="#52677d")
    fig.tight_layout(rect=(.02,.09,1,.94),h_pad=3,w_pad=3)
    fig.savefig(args.output.with_suffix(".png"),dpi=180)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    with args.output.with_suffix(".csv").open("w",newline="") as stream:
        fields=["case","method","success","progress","steps","wall_seconds","input_tokens","cached_input_tokens","output_tokens"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for pair in report["pairs"]:
            for method in ("baseline","candidate"):
                row=pair[method]
                writer.writerow(dict(case=pair["case"],method=method,
                    **{k:row[k] for k in ("success","progress","steps","wall_seconds")},
                    **{k:row["usage"].get(k) for k in ("input_tokens","cached_input_tokens","output_tokens")}))


if __name__=="__main__":
    main()
