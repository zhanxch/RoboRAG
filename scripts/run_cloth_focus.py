#!/usr/bin/env python3
"""Keep collecting and improving cloth RL until seen-condition efficacy or deadline."""
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
from roborag.cloth_grasp.focus import freeze,rows,cases,score,sha,write,amendment_for_round
from run_grasp_scale import stage_demo,owned_job


def state(root,status,**extra):write(root/'status.json',dict(status=status,updated=time.time(),**extra))

def progress(root,**extra):
    p=root/'progress.json';v=json.loads(p.read_text()) if p.exists() else {}
    v.update(extra,updated=time.time());write(p,v)


def evaluate(root,plan,phase,round_number,checkpoint):
    folder=root/'probes'/('baseline' if phase=='baseline' else f'{phase}-{round_number:04d}')
    stage_demo(folder)
    total=len(cases(plan,phase,round_number))
    for start in range(0,total,8):
        expected={c['id'] for c in cases(plan,phase,round_number)[start:start+8]}
        if expected<={r['id'] for r in rows(folder/'episodes.jsonl')}:continue
        cmd=[sys.executable,str(ROOT/'scripts/cloth_grasp.py'),'evaluate','--run',str(folder),
             '--focus-plan',str(root/'training-protocol.json'),'--focus-phase',phase,'--focus-round',str(round_number),
             '--checkpoint',str(checkpoint),'--case-start',str(start),'--case-count','8','--seed',str(plan['seed'])]
        owned_job(root,folder,f'{phase}-{round_number:04d}-{start//8}',cmd,plan['deadline'],1800)
    evaluation=rows(folder/'episodes.jsonl')
    audits=rows(folder/'scene-audit.jsonl')
    if {r['id'] for r in audits}!={r['id'] for r in evaluation} or len(audits)!=total:
        raise ValueError('Physical scene audit is incomplete')
    if any(r['asset']!=9 or r['checkpoint_sha256']!=sha(checkpoint) for r in audits):raise ValueError('Evaluation scene/model provenance mismatch')
    result=score(evaluation,total,plan['effectiveness_gate'])
    result.update(round=round_number,phase=phase,checkpoint_sha256=sha(checkpoint),checkpoint=str(checkpoint),
                  protocol_sha256=plan['sha256'],scope='Training-seen garment and poses, physical resets; not held-out generalization',updated=time.time())
    amendment=amendment_for_round(root,plan,round_number)
    result['learning_revision']=amendment['id'] if amendment else 'v1'
    write(folder/'result.json',result)
    return result


def run(root):
    plan=freeze(root);runtime=json.loads((ROOT/'local.runtime.json').read_text())
    if time.time()>=plan['deadline']:raise TimeoutError('Authorized deadline reached')
    with (root/'pipeline.lock').open('w') as ownlock,open('/tmp/roborag-cloth-grasp-sim.lock','w') as simlock:
        fcntl.flock(ownlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:fcntl.flock(simlock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.time()>=plan['deadline']:raise TimeoutError('Deadline waiting for simulation lock')
                state(root,'waiting',stage='等待本实验仿真锁');time.sleep(10)
        write(root/'pipeline-process.json',dict(pid=os.getpid(),started_at=time.time(),deadline=plan['deadline']))
        source=root/'initial-source';source.mkdir(exist_ok=True)
        for path in (ROOT/'src/roborag/cloth_grasp').glob('*.py'):
            if not (source/path.name).exists():shutil.copy2(path,source/path.name)
        if not (source/Path(__file__).name).exists():shutil.copy2(Path(__file__),source/Path(__file__).name)
        def update(round_number):
            cmd=[runtime['policy_python'],'-c',"import sys;from pathlib import Path;sys.path.insert(0,'src');from roborag.cloth_grasp.focus import train_update;train_update(Path(sys.argv[1]),int(sys.argv[2]))",str(root),str(round_number)]
            owned_job(root,root,f'learn-{round_number:04d}',cmd,plan['deadline'],1800,simulator=False)
            metadata=json.loads((root/f'checkpoints/round-{round_number:04d}.json').read_text())
            progress(root,round=round_number,rl_updates=metadata['total_rl_gradient_updates'],train_transitions=metadata['train_transitions'],
                     latest_checkpoint=str(root/f'checkpoints/round-{round_number:04d}.pt'),world_holdout_rmse=metadata['world_holdout_rmse'],
                     learning_revision=metadata.get('learning_revision','v1'),rl_updates_since_actor_reset=metadata.get('rl_updates_since_actor_reset'))
        if not (root/'checkpoints/round-0000.pt').exists():update(0)
        train=root/'train';stage_demo(train)
        for rnd in range(1,plan['max_rounds']+1):
            if time.time()>=plan['deadline']:raise TimeoutError('Authorized deadline reached')
            completed={r['id'] for r in rows(train/'episodes.jsonl')}
            if not {f'train-{rnd:04d}-{i:02d}' for i in range(8)}<=completed:
                cmd=[sys.executable,str(ROOT/'scripts/cloth_grasp.py'),'train','--run',str(train),
                    '--focus-plan',str(root/'training-protocol.json'),'--focus-phase','collect','--focus-round',str(rnd),
                    '--checkpoint',str(root/f'checkpoints/round-{rnd-1:04d}.pt'),'--case-start','0','--case-count','8','--seed',str(plan['seed'])]
                owned_job(root,train,f'collect-{rnd:04d}',cmd,plan['deadline'],1800)
            if not (root/f'checkpoints/round-{rnd:04d}.pt').exists():update(rnd)
            training=[r for r in rows(train/'episodes.jsonl') if r['id'].startswith(f'train-{rnd:04d}-')]
            progress(root,last_collection_round=rnd,last_collection_successes=sum(bool(r['success']) for r in training),last_collection_total=8)
            amendment=amendment_for_round(root,plan,rnd)
            if rnd%plan['probe_every_rounds'] and rnd not in (amendment or {}).get('additional_probe_rounds',[]):continue
            baseline_file=root/'probes/baseline/result.json'
            if not baseline_file.exists():
                baseline=evaluate(root,plan,'baseline',0,root/'bc-baseline.pt');progress(root,baseline=baseline)
            result=evaluate(root,plan,'probe',rnd,root/f'checkpoints/round-{rnd:04d}.pt')
            bestpath=root/'best.json';best=json.loads(bestpath.read_text()) if bestpath.exists() else None
            if best is None or result['rate']>best['rate']:
                write(bestpath,result);shutil.copy2(result['checkpoint'],root/'best.pt')
                # Export the selected candidate, even when resuming older probes.
                cmd=[runtime['policy_python'],'-c',"import sys;from pathlib import Path;sys.path.insert(0,'src');from roborag.cloth_grasp.focus import restore,export;from roborag.cloth_grasp.learning import Learner;p=Path(sys.argv[1]);l=Learner('cpu',binary_gripper=True);m=restore(l,p/'best.pt');o=p/'best-export';o.mkdir(exist_ok=True);export(l,o,m)",str(root)]
                owned_job(root,root,f'export-best-{rnd:04d}',cmd,plan['deadline'],1800,simulator=False)
                best=result
            previous=[json.loads(p.read_text()) for p in sorted((root/'probes').glob('probe-*/result.json'))]
            streak=0
            for r in reversed(previous):
                if not r['passed'] or r.get('learning_revision','v1')!=result.get('learning_revision','v1'):break
                streak+=1
            progress(root,last_probe=result,best_probe=best,consecutive_passing_probes=streak)
            if streak>=plan['effectiveness_gate']['consecutive_probes']:
                confirmation=evaluate(root,plan,'confirm',rnd,Path(best['checkpoint']))
                progress(root,confirmation=confirmation)
                if confirmation['passed']:
                    write(root/'result.json',dict(status='seen_condition_effective',best=best,confirmation=confirmation,
                        scope=plan['scope'],deadline=plan['deadline'],completed_at=time.time()))
                    state(root,'effective',stage='训练见过条件的局部抓取通过复核；不代表完整叠衣或泛化')
                    return
        state(root,'round_limit',stage='达到预设训练轮数上限，需要检查剩余预算与训练效果')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();root=args.run.resolve()
    if (root/'user-method-correction.json').exists():
        sys.exit('This custom training run was stopped by user method correction; use a new approved method/protocol.')
    def stop(signum,frame):raise KeyboardInterrupt('Owned focus runner stopped')
    signal.signal(signal.SIGTERM,stop)
    try:run(root)
    except BaseException as exc:
        deadline=json.loads((root/'plan.json').read_text())['deadline']
        state(root,'budget_exhausted' if time.time()>=deadline else 'failed',error=str(exc));raise
