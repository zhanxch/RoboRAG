"""New-data development experiments; never reuses pilot test trajectories."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from .learning import Learner


def read_plan(root):
    from .protocol import digest
    value=json.loads((root/'development-protocol.json').read_text())
    expected=value.pop('sha256')
    if digest(value)!=expected:raise ValueError('Development protocol changed')
    value['sha256']=expected
    return value


def fit_offline(root,count,seed,device='cuda'):
    """Fit only TRAIN transitions; whole-episode world holdout stays out."""
    torch.set_num_threads(4)
    plan=read_plan(root)
    out=root/f'models/demos-{count}-seed-{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint-freeze.json').exists():return
    learner=Learner(seed=seed,binary_gripper=True,device=device)
    rows=[json.loads(x) for x in (root/'physics/episodes.jsonl').read_text().splitlines()]
    source_by_id={j['id']:j for j in plan['collection']}
    used=[]
    for row in rows:
        if row['role'] not in ('train','world_holdout'):
            raise ValueError('Only TRAIN and world-holdout collection rows are permitted')
        job=source_by_id[row['id']]
        if job['source_index']>=count and row['role']!='world_holdout':continue
        if row.get('reset_failed'):continue
        d=np.load(root/'physics/rollouts'/row['id']/'transitions.npz')
        learner.replay.add(d['states'],d['actions'],holdout=row['role']=='world_holdout',expert=True)
        used.append(row['id'])
    metadata=dict(algorithm='SuperMimic-inspired BC + learned dynamics BPTT; expanded TRAIN data',
        algorithm_revision='scale-development-v1',seed=seed,demonstrations=count,binary_gripper=True,
        protocol_sha256=plan['sha256'],physics_episode_ids=used,settings=plan['learning'],
        observations='39D privileged geometry',selection='fixed final update; development gate only',started_at=time.time(),
        train_transitions=sum(len(e['a']) for e in learner.replay.episodes if not e['holdout']),
        world_holdout_transitions=sum(len(e['a']) for e in learner.replay.episodes if e['holdout']))
    source=out/'source';source.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,source/path.name)
    config=plan['learning']
    bc_loss=learner.warm_start(config['bc_steps'])
    learner.save(out/'bc-baseline.pt',dict(metadata,bc_loss=bc_loss))
    world_loss=learner.fit_world(steps=config['world_steps'],horizon=8)
    before=learner.validation_rmse()
    for group in learner.actor_opt.param_groups:group['lr']=config['rl_learning_rate']
    metrics=learner.improve(steps=config['policy_updates'],horizon=16,bc_weight=config['bc_weight'])
    metadata.update(training_completed_at=time.time(),world_mse=world_loss,world_holdout_rmse=before,
                    metrics=metrics,bc_loss=bc_loss,actor_parameters=sum(p.numel() for p in learner.actor.parameters()))
    learner.save(out/'final.pt',metadata)
    class Deployment(torch.nn.Module):
        def __init__(self,actor):super().__init__();self.actor=actor
        def forward(self,state):
            a=self.actor(state)
            return torch.cat([a[...,:3],torch.where(a[...,3:4]>=0,1.,-1.)],-1)
    actor=deepcopy(learner.actor).cpu().eval()
    torch.jit.trace(Deployment(actor).eval(),torch.zeros(1,39)).save(str(out/'policy.ts'))
    torch.save(dict(actor=actor.state_dict(),metadata=metadata),out/'policy.pt')
    for name,value in [('experiment.json',metadata),('checkpoint-freeze.json',
          {n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['bc-baseline.pt','final.pt','policy.ts']})]:
        (out/name).write_text(json.dumps(value,indent=2))


def simulator_run(env,out,args):
    from .formal import register_cases,file_hash
    from .training import LocalGrasp,rollout,PregraspError,append
    from .sim_worker import atomic_json
    root=args.scale_plan.parent
    plan=read_plan(root)
    demo=json.loads((out/'demo.json').read_text())
    learner=Learner(seed=args.seed,binary_gripper=True)
    local=LocalGrasp(env,out,demo)
    existing=[json.loads(x) for x in (out/'episodes.jsonl').read_text().splitlines()] if (out/'episodes.jsonl').exists() else []
    completed={r['id'] for r in existing}
    if args.mode=='train':
        jobs=plan['collection'][args.case_start:args.case_start+args.case_count]
        for job in jobs:
            if job['id'] in completed:continue
            source=plan['sources'][job['source_index']]
            sourcepath=Path(source['extract'])
            if hashlib.sha256((sourcepath/'demonstration.npz').read_bytes()).hexdigest()!=source['extracted_sha256']:
                raise ValueError('Frozen demonstration action arrays changed')
            metadata=json.loads((sourcepath/'demo.json').read_text())
            local.events={e['target']:e for e in metadata['events']}
            local.demo_arrays=dict(np.load(sourcepath/'demonstration.npz'))
            atomic_json(out/'status.json',dict(status='collecting',detail=f"{job['id']} · TRAIN source {job['source_index']+1} · {job['target']}",updated=time.time()))
            try:
                rollout(local,learner,job['target'],job['seed'],job['id'],'train-demo-retarget',
                        role=job['role'],jitter=job['jitter'])
            except PregraspError as exc:
                append(out/'episodes.jsonl',dict(id=job['id'],role=job['role'],target=job['target'],seed=job['seed'],
                    reset_failed=True,success=False,reason=str(exc),steps=0,frames=local.frames))
        return
    register_cases(env,out,{'cases':plan['development_cases']})
    frozen=json.loads((out/'checkpoint-freeze.json').read_text())
    for name in ['bc-baseline.pt','final.pt']:
        if file_hash(out/name)!=frozen[name]:raise ValueError('Frozen development checkpoint modified')
    for case in plan['development_cases'][args.case_start:args.case_start+args.case_count]:
        for policy in (['bc','rl'] if case['seed']%2 else ['rl','bc']):
            eid=f"dev-{policy}-{case['id']}"
            if eid in completed:continue
            learner.load(out/('bc-baseline.pt' if policy=='bc' else 'final.pt'))
            learner.actor.eval()
            atomic_json(out/'status.json',dict(status='evaluating',detail=eid,updated=time.time()))
            try:
                rollout(local,learner,case['subtask'],case['seed'],eid,policy+'-policy',role='development',case=case,policy=policy)
            except PregraspError as exc:
                append(out/'episodes.jsonl',dict(id=eid,role='development',case_id=case['id'],policy=policy,
                    target=case['subtask'],seed=case['seed'],garment=case['garment'],success=False,
                    reset_failed=True,reason=str(exc),steps=0,frames=local.frames))
            lm=env.scene_manager.layout_manager
            name=lm.get_instance_name(label='target',env_idx=0)
            category,asset=lm.decode_inst_name(name)
            if category!='Top_Long' or asset!=case['garment']:raise ValueError('Development asset mismatch')
            append(out/'scene-audit.jsonl',dict(id=eid,asset=asset,instance=name,seed=case['seed']))
