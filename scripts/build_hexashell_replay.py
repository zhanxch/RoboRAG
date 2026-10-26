#!/usr/bin/env python3
"""Build a read-only replay of the recorded HexaShell batch (never rerun physics)."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "runs/hexashell-public-calibrated-v2-20261023"
DEFAULT_OUTPUT = ROOT / "runs/hexashell-replay-viewer-20261024"
CAMERAS = {"head": "cam_head", "wrist_l": "cam_left_wrist", "wrist_r": "cam_right_wrist"}
TOOLS = ["cloth_landmarks", "surface_transfer", "edge_transfer", "table_xy", "table_plane",
         "surface_outline", "crease_transfer", "view_reacquire"]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(job):
    frames, output = job
    fingerprints = [dict(path=str(p), sha256=digest(p)) for p in frames]
    cache = output.with_suffix(".json")
    if output.exists() and cache.exists():
        old = read(cache)
        if old.get("sources") == fingerprints and old.get("sha256") == digest(output):
            return old
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
           "-f", "image2pipe", "-framerate", "1", "-vcodec", "png", "-i", "pipe:0",
           "-an", "-vf", "fps=10,format=yuv420p", "-c:v", "libx264", "-threads", "2",
           "-preset", "fast", "-crf", "22", "-movflags", "+faststart",
           "-metadata", "title=HexaShell recorded command observations",
           "-metadata", "comment=One recorded snapshot per second; not real-time motion. No intermediate motion generated.",
           str(output)]
    subprocess.run(cmd, input=b"".join(p.read_bytes() for p in frames), check=True, capture_output=True)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=nb_frames,duration,width,height", "-of", "json", str(output)], text=True))["streams"][0]
    assert abs(float(probe["duration"]) - len(frames)) < .11, (output, probe)
    result = dict(sources=fingerprints, sha256=digest(output), snapshot_count=len(frames),
                  snapshot_seconds=1, encoded_fps=10, probe=probe,
                  note="Recorded command-boundary snapshots, held for one second. Not continuous or real-time motion.")
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def session_record(directory, commands):
    """Keep recorded public messages/tool IO; do not infer unlogged model turns."""
    events, starts, active, overlaps = [], {}, {}, []
    cursor, shell_count, robo_attempts = 0, 0, 0
    raw = directory / "agent-events.jsonl"
    actions = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines()]
    assert [a["step"] for a in actions] == list(range(1, commands[-1]["after"] + 1))
    events.append(dict(id="task-input", kind="task", title="任务输入", text=(directory / "agent-prompt.txt").read_text(), frame=0))
    for line_number, line in enumerate(raw.read_text().splitlines(), 1):
        record = json.loads(line)
        item = record.get("item", {})
        item_type, event_type = item.get("type"), record["type"]
        if event_type == "item.started" and item_type == "command_execution":
            starts[item["id"]] = line_number
            active[item["id"]] = item.get("command", "")
            if len(active) > 1:
                overlaps.append(dict(line=line_number, ids=list(active), commands=list(active.values())))
        if event_type == "turn.completed":
            events.append(dict(id="turn-completed", kind="completion", title="CLI 回合结束", line=line_number,
                               frame=cursor, usage=record.get("usage", {})))
        if event_type != "item.completed":
            continue
        common = dict(id=item.get("id", f"line-{line_number}"), line=line_number, frame=cursor)
        if item_type == "agent_message":
            events.append(dict(**common, kind="message", title="Codex 消息", text=item.get("text", "")))
        elif item_type == "error":
            events.append(dict(**common, kind="notice", title="CLI 运行提示", text=item.get("message", "")))
        elif item_type == "command_execution":
            active.pop(item["id"], None)
            shell_count += 1
            shell = item.get("command", "")
            words = shlex.split(shell)
            body = words[-1] if "-lc" in words else shell
            argv = shlex.split(body)
            linked, rejected = [], False
            if argv and argv[0] == "robo":
                robo_attempts += 1
                if cursor < len(commands) and argv == commands[cursor]["argv"]:
                    cursor += 1
                    linked.append(cursor)
                    common["frame"] = cursor
                else:
                    # The client refuses new requests after termination; these
                    # still belong in the Codex session, but not the robot log.
                    receipt = json.loads(item.get("aggregated_output", ""))
                    assert receipt.get("episode_over") is True, (directory, item["id"], body)
                    rejected = True
            category = "robot" if argv and argv[0] == "robo" else "python" if argv and argv[0] in ("python", "python3") else "shell"
            title = "机器人命令请求" if category == "robot" else "执行 Python 脚本" if category == "python" else "读取文件 / shell"
            events.append(dict(**common, kind="shell", title=title, category=category, shellNumber=shell_count,
                               startLine=starts.get(item["id"]), command=shell, script=body,
                               stdout=item.get("aggregated_output", ""), exitCode=item.get("exit_code"),
                               status=item.get("status"), commandIndices=linked, clientRejected=rejected))
        # Intentionally exclude private/unsupported event payloads. The current
        # batch contains only the three public item types handled above.
    assert cursor == len(commands), (directory, cursor, len(commands))
    for command in commands:
        command["actionCount"] = sum(command["before"] < a["step"] <= command["after"] for a in actions)
        assert command["actionCount"] == command["after"] - command["before"]
    return dict(schema="hexashell.visible-session.v1", episode=directory.name, events=events, actions=actions,
                summary=dict(shellCalls=shell_count, roboAttempts=robo_attempts, dispatchedCommands=cursor,
                             clientRejected=robo_attempts-cursor, actionRecords=len(actions),
                             messages=sum(e["kind"] == "message" for e in events),
                             pythonCalls=sum(e.get("category") == "python" for e in events),
                             overlappingShellGroups=len(overlaps)), overlaps=overlaps,
                sourceSha256=digest(raw), source="agent-events.jsonl",
                coverageNote="按 CLI 记录展示消息、shell 请求和返回；未记录逐次模型请求分组、独立看图工具事件和工具内部动作帧。")


def build(batch=DEFAULT_BATCH, output=DEFAULT_OUTPUT):
    batch, output = batch.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "media").mkdir(exist_ok=True)
    (output / "sessions").mkdir(exist_ok=True)
    assets, episodes, jobs = {}, [], []
    counts, coverage = Counter(), Counter()

    def asset(path):
        path = path.resolve()
        relative = path.relative_to(batch).as_posix()
        assert path.is_file(), path
        url = "/evidence/" + quote(relative, safe="/")
        assets[url] = str(path)
        return url

    for directory in sorted(batch.iterdir(), key=lambda p: (not p.name.startswith("standard"), p.name)):
        if not (directory / "commands.jsonl").exists():
            continue
        rows = [json.loads(line) for line in (directory / "commands.jsonl").read_text().splitlines()]
        result, run = read(directory / "result.json"), read(directory / "run.json")
        assert result["complete"] and result["scored"], directory
        initial = directory / "observations/000_initial"
        folders = [initial] + [Path(row["observation"]) for row in rows]
        assert all(p.resolve().is_relative_to(directory.resolve()) for p in folders)
        frames = []
        for index, folder in enumerate(folders):
            state = read(folder / "state.json")
            frames.append(dict(index=index, step=state["step"],
                               joints={arm:state[arm]["joints"] for arm in ("left", "right")},
                               images={camera: asset(folder / f"{camera}.png") for camera in CAMERAS}))
        commands = []
        for index, row in enumerate(rows, 1):
            argv = row["request"]["argv"]
            assert argv[0] == "robo"
            name = argv[1]
            counts[name] += 1
            assert row["step_after"] == frames[index]["step"]
            commands.append(dict(index=index, name=name, argv=argv, result=row["result"],
                                 exitCode=row["exit_code"], before=row["step_before"],
                                 after=row["step_after"], wallSeconds=row["wall_seconds"]))
        coverage.update(set(c["name"] for c in commands))
        session = session_record(directory, commands)
        (output / "sessions" / f"{directory.name}.json").write_text(json.dumps(session, ensure_ascii=False, separators=(",", ":")) + "\n")
        videos, originals = {}, {}
        for camera, native_name in CAMERAS.items():
            target = output / "media" / f"{directory.name}-{camera}-observations.mp4"
            jobs.append(([p / f"{camera}.png" for p in folders], target))
            videos[camera] = "/media/" + target.name
            native = list(directory.glob(f"native/**/episode_0000000_{native_name}_*.mp4"))
            assert len(native) == 1, native
            originals[camera] = asset(native[0])
        episode = dict(id=directory.name, variant=run["arguments"]["variant"], layout=run["arguments"]["layout"],
                       split="development" if run["arguments"]["layout"] < 5 else "new_layouts",
                       success=result["native"]["success_rate"] == 1, score=result["native"]["score"],
                       steps=rows[-1]["step_after"], wallSeconds=result["wall_seconds"],
                       usage=result.get("native_agent", {}).get("usage", {}),
                       startedAt=run["started_at"], commands=commands, frames=frames,
                       sessionUrl=f"/sessions/{directory.name}.json", sessionSummary=session["summary"],
                       videos=videos, originals=originals,
                       sources={"commands": asset(directory / "commands.jsonl"), "result": asset(directory / "result.json"),
                                "run": asset(directory / "run.json"), "actions":asset(directory / "actions.jsonl"),
                                "prompt":asset(directory / "agent-prompt.txt"), "manual":asset(directory / "INTERFACE.md")},
                       sourceHashes={name:digest(directory / name) for name in ("commands.jsonl", "result.json", "run.json")})
        episodes.append(episode)
    assert len(episodes) == 20, f"Expected the completed 20-episode batch, got {len(episodes)}"
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, _ in enumerate(pool.map(encode, jobs), 1):
            if i % 6 == 0:
                print(f"Prepared {i}/{len(jobs)} camera replays", flush=True)
    summary = dict(episodes=len(episodes), successes=sum(e["success"] for e in episodes),
                   meanScore=sum(e["score"] for e in episodes)/len(episodes), commands=sum(counts.values()),
                   toolCalls=sum(counts[t] for t in TOOLS), snapshots=sum(len(e["frames"]) for e in episodes),
                   commandCounts=dict(counts), commandCoverage=dict(coverage))
    summary["session"] = {key:sum(e["sessionSummary"][key] for e in episodes) for key in episodes[0]["sessionSummary"]}
    manifest = dict(schema="hexashell.replay.v1", generatedAt=datetime.now(timezone.utc).isoformat(),
                    batch=batch.name, sourceDate="2026-10-23 UTC", model="GPT-6-Astra", effort="xhigh",
                    controlHz=25, snapshotSeconds=1, stepLimit=500, parallel=2,
                    videoNote="每秒一张真实观测；仅含初始状态和命令结束后的截图。没有连续动作帧，也未合成中间动作。",
                    summary=summary, episodes=episodes)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n")
    (output / "asset-map.json").write_text(json.dumps(assets, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.batch, args.output)
