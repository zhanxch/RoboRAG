#!/usr/bin/env python3
"""Bounded, resumable multi-demo, three-seed development run on one local GPU."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from roborag.cloth_grasp.protocol import digest,TARGETS


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def state(root,phase,**kwargs):
    write(root/'status.json',dict(status=phase,updated=time.time(),**kwargs))


def freeze(root):
    path=root/'development-protocol.json'
    if path.exists():
        v=json.loads(path.read_text());checksum=v.pop('sha256')
        if digest(v)!=checksum:raise ValueError('Frozen protocol was modified')
        v['sha256']=checksum;return v
    manifest=json.loads((root/'data-manifest.json').read_text())
    sources=[r for r in manifest['files'] if r.get('cloth_windows_valid') and r['status']=='verified']
    if len(sources)!=10:raise RuntimeError(f'Require 10 verified, structurally valid fold demonstrations, found {len(sources)}; inspect source windows before proceeding')
    for source in sources:
        extracted=Path(source['extract'])
        meta=json.loads((extracted/'demo.json').read_text())
        if meta['frequency_hz']!=25 or meta['sha256']!=source['sha256']:
            raise ValueError('Demonstration frequency or source hash does not match training contract')
        source['extracted_sha256']=hashlib.sha256((extracted/'demonstration.npz').read_bytes()).hexdigest()
    jobs=[]
    for source in range(10):
        for repeat in range(2):
            for target in TARGETS:
                n=len(jobs)
                jobs.append(dict(id=f'collect-{n+1:03d}',source_index=source,target=target,seed=600000+n,
                                 role='train',jitter=.012 if repeat==0 else .025))
    for i in range(8):jobs.append(dict(id=f'collect-{len(jobs)+1:03d}',source_index=0,target=TARGETS[i%4],seed=610000+i,role='world_holdout',jitter=.02))
    cases=[]
    for garment in [9,7]:
        for target in TARGETS:
            for label,offset in [('low',[-.015,.01,.06]),('high',[.01,.015,.08]),('lateral',[-.025,-.01,.07])]:
                n=len(cases)
                cases.append(dict(id=f'd{n+1:03d}',seed=930000+n,garment=garment,subtask=target,
                    distribution='train_asset' if garment==9 else 'development_asset',position=label,
                    tcp_offset_m=offset,garment_position=[-.01,-.11,.8441]))
    v=dict(schema='grasp.scale.development.v1',frozen_at=time.time(),sources=sources,
        collection=jobs,development_cases=cases,demonstration_counts=[1,10],training_seeds=[31001,31002,31003],
        learning=dict(bc_steps=4800,world_steps=1200,policy_updates=50,bc_weight=20.,rl_learning_rate=2e-5),
        selection='All models fixed after offline updates; development results only. No pilot test data, and no repeated use as a final test.',
        scope='One local cloth subskill, not native full-task scores. No success-based filtering of TRAIN teacher trajectories.',
        criteria=dict(lift_m=.08,gap_m=.055,final_retention_steps=10,horizon=80))
    v['sha256']=digest(v);write(path,v);return v


def stage_demo(out):
    out.mkdir(parents=True,exist_ok=True)
    pilot=ROOT/'runs/cloth-grasp-bench-v1'
    for name in ['demo.json','demonstration.npz']:
        if not (out/name).exists():shutil.copy2(pilot/name,out/name)


def owned_job(root,out,label,cmd,deadline,timeout,simulator=True):
    """Reap only the process group created here; never affect another workload."""
    if time.time()>=deadline:raise TimeoutError('Authorized 72-hour deadline reached')
    state(root,'running',stage=label,output=str(out))
    logpath=root/(label.replace('/','-')+'.log')
    with logpath.open('a') as log:
        offset=log.tell();child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        began=time.time();finished_at=None
        write(root/'owned-process.json',dict(pid=child.pid,pgid=child.pid,label=label,started_at=began,deadline=deadline))
        try:
            while child.poll() is None:
                if time.time()>=deadline:raise TimeoutError('Authorized 72-hour deadline reached')
                if time.time()-began>timeout:raise TimeoutError(f'Worker timeout: {label}')
                if simulator:
                    try:s=json.loads((out/'status.json').read_text())
                    except (FileNotFoundError,json.JSONDecodeError):s={}
                    with logpath.open('rb') as f:
                        f.seek(max(offset,logpath.stat().st_size-16384));tail=f.read().decode(errors='replace')
                    if 'ERROR_OUT_OF_DEVICE_MEMORY' in tail:raise RuntimeError('GPU renderer OOM; completed rollouts preserved')
                    if s.get('updated',0)>=began:
                        if s.get('status')=='failed':raise RuntimeError(s.get('error','Simulation failed'))
                        if s.get('status')=='complete':
                            finished_at=finished_at or time.time()
                            if time.time()-finished_at>15:break
                time.sleep(2)
            if not simulator and child.poll()!=0:raise RuntimeError(f'Offline training failed: {label}')
            if simulator and not finished_at:
                s=json.loads((out/'status.json').read_text())
                if s.get('status')!='complete' or s.get('updated',0)<began:raise RuntimeError(f'Simulator did not complete: {label}')
        finally:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
            write(root/'owned-process.json',dict(pid=None,last_label=label,finished_at=time.time()))


def main(root):
    plan=json.loads((root/'plan.json').read_text());deadline=plan['deadline']
    runtime=json.loads((ROOT/'local.runtime.json').read_text())
    while True:
        if time.time()>=deadline:raise TimeoutError('Authorized deadline reached while waiting')
        try:data=json.loads((root/'data-status.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):data={}
        if data.get('status')=='failed':raise RuntimeError('TRAIN preparation failed: '+data.get('error',''))
        if data.get('status')=='complete' and (ROOT/'runs/cloth-grasp-bench-v1/audit.json').exists():break
        state(root,'waiting',stage='Verified TRAIN data and completed pilot audit');time.sleep(10)
    protocol=freeze(root)
    with (root/'pipeline.lock').open('w') as ownlock,open('/tmp/roborag-cloth-grasp-sim.lock','w') as simlock:
        fcntl.flock(ownlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:fcntl.flock(simlock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.time()>=deadline:raise TimeoutError('Deadline while waiting for owned simulator lock')
                time.sleep(10)
        physics=root/'physics';stage_demo(physics)
        base=[sys.executable,str(ROOT/'scripts/cloth_grasp.py')]
        for start in range(0,len(protocol['collection']),8):
            # A completed batch is skipped without launching Isaac Sim again.
            rows=[json.loads(x) for x in (physics/'episodes.jsonl').read_text().splitlines()] if (physics/'episodes.jsonl').exists() else []
            done={r['id'] for r in rows}
            if all(j['id'] in done for j in protocol['collection'][start:start+8]):continue
            cmd=base+['train','--run',str(physics),'--scale-plan',str(root/'development-protocol.json'),'--case-start',str(start),'--case-count','8','--seed','600000']
            owned_job(root,physics,f'collect-{start//8+1:02d}',cmd,deadline,1800)
        for count in protocol['demonstration_counts']:
            for seed in protocol['training_seeds']:
                out=root/f'models/demos-{count}-seed-{seed}';stage_demo(out)
                cmd=[runtime['policy_python'],'-c',"import sys;from pathlib import Path;sys.path.insert(0,'src');from roborag.cloth_grasp.scale import fit_offline;fit_offline(Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))",str(root),str(count),str(seed)]
                owned_job(root,out,f'fit-{count}-{seed}',cmd,deadline,1800,simulator=False)
                for start in range(0,len(protocol['development_cases']),4):
                    rows=[json.loads(x) for x in (out/'episodes.jsonl').read_text().splitlines()] if (out/'episodes.jsonl').exists() else []
                    done={r['id'] for r in rows}
                    if all(f"dev-{p}-{c['id']}" in done for c in protocol['development_cases'][start:start+4] for p in ['bc','rl']):continue
                    cmd=base+['evaluate','--run',str(out),'--scale-plan',str(root/'development-protocol.json'),'--case-start',str(start),'--case-count','4','--seed',str(seed)]
                    owned_job(root,out,f'dev-{count}-{seed}-{start//4+1}',cmd,deadline,1800)
        results=[]
        for count in protocol['demonstration_counts']:
            for seed in protocol['training_seeds']:
                out=root/f'models/demos-{count}-seed-{seed}'
                rows=[json.loads(x) for x in (out/'episodes.jsonl').read_text().splitlines()]
                expected={(p,c['id']) for p in ['bc','rl'] for c in protocol['development_cases']}
                observed=[(r['policy'],r['case_id']) for r in rows]
                if set(observed)!=expected or len(observed)!=len(expected):raise ValueError('Incomplete/duplicate development pairs')
                rates={p:sum(r['success'] for r in rows if r['policy']==p)/24 for p in ['bc','rl']}
                results.append(dict(demonstrations=count,seed=seed,rates=rates))
        expanded=[r for r in results if r['demonstrations']==10]
        gate=plan['dev_gate']
        passed=(sum(r['rates']['rl'] for r in expanded)/3>=gate['minimum_rl_rate'] and
                sum(r['rates']['rl']-r['rates']['bc'] for r in expanded)/3>=gate['minimum_rl_minus_bc'] and
                all(r['rates']['rl']>=r['rates']['bc'] for r in expanded))
        write(root/'development-results.json',dict(protocol_sha256=protocol['sha256'],results=results,expansion_gate_passed=passed,
            completed_at=time.time(),scope='Development data only; full tasks not evaluated',
            next_action='Freeze new final benchmark and validate rigid adapters' if passed else 'Diagnose TRAIN/dev failure before broader rollout; preserve negative results'))
        state(root,'development_complete',expansion_gate_passed=passed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();root=args.run.resolve()
    try:main(root)
    except Exception as exc:
        state(root,'failed',error=str(exc));raise
