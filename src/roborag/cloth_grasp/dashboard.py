"""Loopback-only, read-only experiment viewer with bounded file serving."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse

WEB = Path(__file__).resolve().parents[3] / "web/cloth-grasp"
DOWNLOADS = ['cloth-grasp-model-and-results.zip','policy.ts','policy.pt','bc-policy.ts','benchmark.json',
             'benchmark-results.json','benchmark.csv','model-card.json','audit.json',
             'benchmark-summary.png','benchmark-summary.pdf','coverage.json']
FOCUS_DOWNLOADS=['policy.ts','policy.pt','bc-policy.ts','best.pt','best-export/policy.ts','training-protocol.json','progress.json']
ROW_CACHE={}

def cached_rows(path):
    if not path.exists():return []
    stamp=path.stat().st_mtime_ns
    if path not in ROW_CACHE or ROW_CACHE[path][0]!=stamp:
        rows=[]
        for line in path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
        ROW_CACHE[path]=(stamp,rows)
    return ROW_CACHE[path][1]

def focus_snapshot(root):
    def read(path):
        try:return json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):return None
    plan=read(root/'plan.json')
    if not plan:return None
    train=cached_rows(root/'train/episodes.jsonl')
    episodes=[dict(r,focus_path='train/') for r in train[-80:]]
    folders=sorted((root/'probes').glob('*'),key=lambda p:p.stat().st_mtime)
    for folder in folders[-2:]:
        episodes.extend(dict(r,focus_path=str(folder.relative_to(root))+'/') for r in cached_rows(folder/'episodes.jsonl'))
    livepaths=[p for p in [root/'train/live.json',*(root/'probes').glob('*/live.json')] if p.is_file()]
    live=None
    if livepaths:
        path=max(livepaths,key=lambda p:p.stat().st_mtime);live=read(path)
        if live:live['frames']={k:str(path.parent.relative_to(root))+'/'+v for k,v in live['frames'].items()}
    probes=[read(p) for p in sorted((root/'probes').glob('probe-*/result.json'))]
    return dict(plan=plan,status=read(root/'status.json'),progress=read(root/'progress.json'),live=live,
        metrics=cached_rows(root/'metrics.jsonl'),episodes=episodes,probes=[p for p in probes if p],
        train_episodes=len(train),train_steps=sum(r.get('steps',0) for r in train),
        downloads=[p for p in FOCUS_DOWNLOADS if (root/p).is_file()])


def mimic_snapshot(root):
    def read(p):
        try:return json.loads(p.read_text())
        except (FileNotFoundError,json.JSONDecodeError):return None
    plan=read(root/'effective-plan.json') or read(root/'plan.json')
    if not plan:return None
    episodes=[]
    for folder in [root/'references',root/'train',*sorted((root/'probes').glob('*'))[-2:]]:
        episodes.extend(dict(r,mimic_path=str(folder.relative_to(root))+'/') for r in cached_rows(folder/'episodes.jsonl'))
    paths=[p for p in [root/'references/live.json',root/'train/live.json',*(root/'probes').glob('*/live.json')] if p.exists()]
    live=None
    if paths:
        path=max(paths,key=lambda p:p.stat().st_mtime);live=read(path)
        if live:live['frames']={k:str(path.parent.relative_to(root))+'/'+v for k,v in live['frames'].items()}
    return dict(plan=plan,status=read(root/'status.json'),progress=read(root/'progress.json'),references=read(root/'references.json'),
        model=read(root/'model-card.json'),episodes=episodes,metrics=cached_rows(root/'metrics.jsonl'),
        probes=cached_rows(root/'probes.jsonl'),best=read(root/'best.json'),live=live,
        downloads=[n for n in ['plan.json','effective-plan.json','adapter-amendment.json','model-card.json','policy.ts','last.pt','best.pt','best-policy.ts','references.json','verification-protocol.json','conclusion.json','audit.json','supermimic-cloth-model.zip','confirmed-policy.ts'] if (root/n).exists()])


def snapshot(run):
    def read(name, default=None):
        try: return json.loads((run / name).read_text())
        except (FileNotFoundError, json.JSONDecodeError): return default
    def lines(name):
        rows = []
        try:
            for line in (run / name).read_text().splitlines():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass  # writer may be midway through final line
        except FileNotFoundError: pass
        return rows
    scale_root=run.parent/'grasp-scale-72h'
    def scale_read(path,default=None):
        try:return json.loads((scale_root/path).read_text())
        except (FileNotFoundError,json.JSONDecodeError):return default
    def scale_rows(path):
        try:return [json.loads(x) for x in (scale_root/path).read_text().splitlines()]
        except (FileNotFoundError,json.JSONDecodeError):return []
    models=[]
    for folder in sorted((scale_root/'models').glob('demos-*')):
        rows=scale_rows(folder.relative_to(scale_root)/'episodes.jsonl')
        models.append(dict(name=folder.name,trained=(folder/'checkpoint-freeze.json').exists(),
            completed=len(rows),policies={p:dict(total=sum(r.get('policy')==p for r in rows),
            successes=sum(bool(r.get('success')) for r in rows if r.get('policy')==p)) for p in ['bc','rl']}))
    scale=dict(plan=scale_read('plan.json'),status=scale_read('status.json'),data=scale_read('data-status.json'),
               results=scale_read('development-results.json'),physics_completed=len(scale_rows('physics/episodes.jsonl')),models=models)
    return dict(demo=read("demo.json"), status=read("status.json", {"status":"prepared"}),
                config=read("experiment.json"), metrics=lines("metrics.jsonl"),
                episodes=lines("episodes.jsonl"), evaluation=read("evaluation.json"),
                live=read("live.json"), probe=read("probe.json"), run=run.name,
                comparisons=read("comparisons.json"),baseline=read("baseline-evaluation.json"),
                benchmark=read("benchmark.json"),results=read("benchmark-results.json"),
                coverage=read("coverage.json"),scale=scale,focus=focus_snapshot(run.parent/'cloth-focus-72h'),mimic=mimic_snapshot(run.parent/'supermimic-cloth-v1'),
                downloads=[name for name in DOWNLOADS if (run/name).is_file()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--port", type=int, default=8768)
    args = p.parse_args()
    run = args.run.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            route = unquote(urlparse(self.path).path)
            if route == "/api/state":
                data = json.dumps(snapshot(run), ensure_ascii=False, allow_nan=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if route.startswith('/mimic/'):
                base=run.parent/'supermimic-cloth-v1';name=route[7:]
                if name not in ['plan.json','effective-plan.json','adapter-amendment.json','model-card.json','policy.ts','last.pt','best.pt','best-policy.ts','references.json','verification-protocol.json','conclusion.json','audit.json','supermimic-cloth-model.zip','confirmed-policy.ts'] and not ('/rollouts/' in name and name.endswith(('.jpg','/geometry.json'))):
                    return self.send_error(404)
            elif route.startswith('/focus/'):
                base=run.parent/'cloth-focus-72h';name=route[7:]
                if name not in FOCUS_DOWNLOADS and not ('/rollouts/' in name and name.endswith(('.jpg','/geometry.json'))):
                    return self.send_error(404)
            elif route.startswith("/data/"):
                base = run
                name = route[6:]
                if not name.startswith(("demo_frames/", "rollouts/", "probe_")) and name not in DOWNLOADS:
                    return self.send_error(404)
            else:
                base = WEB
                name = route.lstrip("/") or "index.html"
            path = (base / name).resolve()
            if not path.is_relative_to(base.resolve()) or not path.is_file():
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            if (route.startswith('/data/') and name in DOWNLOADS) or (route.startswith('/focus/') and name in FOCUS_DOWNLOADS):
                self.send_header('Content-Disposition',f'attachment; filename="{path.name}"')
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(path.read_bytes())

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Local: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__": main()
