#!/usr/bin/env python3
"""Own and bound the paper-based SuperMimic cloth run; never restart older runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from roborag.supermimic_cloth.protocol import freeze,write,rows,sha,TARGETS,digest,verification as read_verification
from run_grasp_scale import owned_job

def state(root,status,**kw):write(root/'status.json',dict(status=status,updated=time.time(),**kw))
def stage(out,plan):
    out.mkdir(parents=True,exist_ok=True)
    for name in ['demo.json','demonstration.npz']:
        if not (out/name).exists():shutil.copy2(Path(plan['sources'][0]['folder'])/name,out/name)
def sim(root,out,phase,round_number=0,start=0,count=4):
    plan=freeze(root);stage(out,plan)
    cmd=[sys.executable,str(ROOT/'scripts/cloth_grasp.py'),'train','--run',str(out),
         '--mimic-plan',str(root/'plan.json'),'--mimic-phase',phase,'--focus-round',str(round_number),
         '--case-start',str(start),'--case-count',str(count),'--seed','720000']
    with open('/tmp/roborag-cloth-grasp-sim.lock','w') as lock:
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.time()>=plan['deadline']:raise TimeoutError('Deadline waiting for simulation lock')
                state(root,'waiting_for_simulation_lock');time.sleep(5)
        owned_job(root,out,f'{phase}-{round_number:05d}-{start}',cmd,plan['deadline'],1800)

def main(args):
    root=args.run.resolve();root.mkdir(parents=True,exist_ok=True);plan=freeze(root)
    with (root/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(root/'coordinator.json',dict(pid=os.getpid(),pgid=os.getpgrp(),started_at=time.time(),deadline=plan['deadline']))
        paths=list((ROOT/'src/roborag/supermimic_cloth').glob('*.py'))
        hashes={p.name:sha(p) for p in paths}
        source=root/'source'/digest(hashes);source.mkdir(parents=True,exist_ok=True)
        for p in paths:shutil.copy2(p,source/p.name)
        write(root/'source-hashes.json',hashes)
        physics=root/'references'
        for start in range(0,len(plan['candidates']),4):
            done={r['id'] for r in rows(physics/'episodes.jsonl')}
            if not all(c['id'] in done for c in plan['candidates'][start:start+4]):sim(root,physics,'reference',start=start)
        active_ids={c['id'] for c in plan['candidates']}
        candidates=[r for r in rows(physics/'episodes.jsonl') if r['id'] in active_ids]; selected=[]
        for target in TARGETS:
            passing=[r for r in candidates if r['target']==target and r['success']]
            if passing:
                r=passing[0];selected.append(dict(id=r['id'],path=r['transition_path'],sha256=r['transition_sha256'],case=r['case']))
        write(root/'references.json',dict(selected=selected,total=len(candidates),passed=sum(r['success'] for r in candidates),gate_passed=len(selected)==4,criterion=plan['criterion']))
        if len(selected)!=4:
            state(root,'reference_gate_failed',missing_targets=[t for t in TARGETS if not any(r['case']['target']==t for r in selected)],detail='Diagnose reference contact on TRAIN. No policy training starts from failing references.');return
        if args.references_only:state(root,'references_ready');return
        if args.reset_check_only:
            sim(root,root/'reset-checks','check')
            checks=rows(root/'reset-checks/episodes.jsonl')
            write(root/'reset-checks/result.json',dict(total=len(checks),passed=sum(r['success'] for r in checks),gate_passed=len(checks)==4 and all(r['success'] for r in checks)))
            state(root,'reset_check_complete');return
        from roborag.supermimic_cloth.protocol import append
        runtime=json.loads((ROOT/'local.runtime.json').read_text())
        def fit(round_number):
            cmd=[runtime['policy_python'],'-u','-m','roborag.supermimic_cloth.learn','--run',str(root),'--round',str(round_number)]
            # module imports need source path; launch through an explicit script wrapper.
            cmd=[runtime['policy_python'],str(ROOT/'scripts/train_supermimic_cloth.py'),'--run',str(root),'--round',str(round_number)]
            owned_job(root,root,f'learn-{round_number:05d}',cmd,plan['deadline'],7200,simulator=False)
        fit(0)
        verification=read_verification(root)
        round_number=1
        while time.time()<plan['deadline']:
            used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            if used>plan['storage_budget_gb']*1e9 or shutil.disk_usage(root).free<5e9:raise RuntimeError('Experiment storage limit reached; preserve existing results')
            if (root/f'round-{round_number:05d}.json').exists():round_number+=1;continue
            sim(root,root/'train','collect',round_number)
            fit(round_number)
            if round_number%plan['probe_every_rounds']==0:
                out=root/'probes'/f'round-{round_number:05d}';sim(root,out,'probe',round_number)
                result=rows(out/'episodes.jsonl')
                rate=sum(r['success'] for r in result)/len(result)
                by_target={t:sum(r['success'] for r in result if r['target']==t)/sum(r['target']==t for r in result) for t in TARGETS}
                summary=dict(round=round_number,rate=rate,n=len(result),successes=sum(r['success'] for r in result),by_target=by_target,scope='seen-condition development',policy_sha256=sha(root/'policy.ts'))
                write(out/'result.json',summary);append(root/'probes.jsonl',summary)
                best=json.loads((root/'best.json').read_text()) if (root/'best.json').exists() else {'rate':-1}
                if rate>best['rate']:
                    shutil.copy2(root/'last.pt',root/'best.pt');shutil.copy2(root/'policy.ts',root/'best-policy.ts');write(root/'best.json',summary)
                checkpoints=root/'checkpoints';checkpoints.mkdir(exist_ok=True)
                shutil.copy2(root/'last.pt',checkpoints/f'round-{round_number:05d}.pt')
                recent=rows(root/'probes.jsonl')[-2:]
                passed=lambda x:x['n']==16 and x['rate']>=.9 and min(x['by_target'].values())>=.75
                if len(recent)==2 and all(passed(x) for x in recent):
                    confirm=root/'probes'/f'confirm-{round_number:05d}';sim(root,confirm,'confirm',round_number)
                    results=rows(confirm/'episodes.jsonl')
                    rate=sum(r['success'] for r in results)/len(results)
                    per={t:sum(r['success'] for r in results if r['target']==t)/sum(r['target']==t for r in results) for t in TARGETS}
                    passed_confirmation=len(results)==32 and rate>=.9 and min(per.values())>=.75
                    write(root/'confirmation.json',dict(passed=passed_confirmation,rate=rate,n=len(results),by_target=per,hold_seconds=3,scope='seen-condition confirmation',policy_sha256=sha(root/'policy.ts')))
                    if passed_confirmation:
                        shutil.copy2(root/'policy.ts',root/'confirmed-policy.ts')
                        finalize(root,'effective');return
            write(root/f'round-{round_number:05d}.json',dict(complete=True,updated=time.time()))
            round_number+=1
        finalize(root,'deadline_reached')


def finalize(root,status):
    runtime=json.loads((ROOT/'local.runtime.json').read_text())
    import subprocess
    command=[runtime['policy_python'],str(ROOT/'scripts/audit_supermimic_cloth.py'),'--run',str(root),'--package']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=300)
    write(root/'finalization.json',dict(status=status,audit_returncode=result.returncode,output=result.stdout[-4000:],error=result.stderr[-4000:],updated=time.time()))
    state(root,status,audit_passed=result.returncode==0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'runs/supermimic-cloth-v1');p.add_argument('--references-only',action='store_true');p.add_argument('--reset-check-only',action='store_true');args=p.parse_args()
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Coordinator signal {signum}; reap own worker group')
    signal.signal(signal.SIGTERM,interrupted)
    try:main(args)
    except TimeoutError as e:
        if time.time()>=freeze(args.run.resolve())['deadline']:finalize(args.run.resolve(),'deadline_reached')
        else:state(args.run.resolve(),'failed',error=str(e));raise
    except BaseException as e:
        state(args.run.resolve(),'failed',error=str(e));raise
