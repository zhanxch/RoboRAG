"""Persistent on-policy cloth RL, calibrated explicitly on seen conditions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from .learning import Learner
from .protocol import TARGETS,digest


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_plan(root):
    p=json.loads((root/'training-protocol.json').read_text());expected=p.pop('sha256')
    if digest(p)!=expected:raise ValueError('Focused training protocol changed')
    p['sha256']=expected;return p


def amendment_for_round(root,plan,round_number):
    """Version optimization changes without rewriting the frozen base protocol."""
    found=[]
    for path in sorted((root/'amendments').glob('*.json')):
        value=json.loads(path.read_text());checksum=value.pop('sha256')
        if digest(value)!=checksum or value['base_protocol_sha256']!=plan['sha256']:
            raise ValueError('Training amendment hash/parent mismatch')
        value['sha256']=checksum
        if value['effective_round']<=round_number:found.append(value)
    return max(found,key=lambda x:x['effective_round']) if found else None


def freeze(root):
    if (root/'training-protocol.json').exists():return read_plan(root)
    p=json.loads((root/'plan.json').read_text())
    manifest=json.loads(Path(p['source_manifest']).read_text())
    sources=[r for r in manifest['files'] if r['task']=='fold_clothes' and r['status']=='verified' and r.get('cloth_windows_valid')]
    if len(sources)!=10:raise ValueError('Ten verified fold_clothes sources required')
    for source in sources:
        folder=Path(source['extract']);meta=json.loads((folder/'demo.json').read_text())
        if meta['frequency_hz']!=25 or meta['sha256']!=source['sha256']:raise ValueError('Demo source contract mismatch')
        source['action_sha256']=sha(folder/'demonstration.npz')
    p.update(sources=sources,frozen_at=time.time(),orientation='Fixed original demonstration-0 target-side orientation for both teacher and policy; additional demonstrations contribute translated TCP action trajectories only.')
    p['sha256']=digest(p);write(root/'training-protocol.json',p);return p


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def load_replay(learner,root,plan,through_round):
    """Only original TRAIN/whole-world-holdout and current TRAIN enter replay."""
    pilot=Path(plan['pilot_train']);used=[]
    for r in rows(pilot/'episodes.jsonl'):
        if r['role'] not in ('train','world_holdout'):continue
        data=np.load(pilot/'rollouts'/r['id']/'transitions.npz')
        learner.replay.add(data['states'],data['actions'],holdout=r['role']=='world_holdout',
                          expert=r['controller'] in ('train-demo-retarget','demo-guided'))
        used.append('pilot/'+r['id'])
    for r in rows(root/'train/episodes.jsonl'):
        if r['role']!='train':raise ValueError('Evaluation data found in TRAIN replay directory')
        rnd=int(r['id'].split('-')[1])
        if rnd>through_round or r.get('reset_failed'):continue
        data=np.load(root/'train/rollouts'/r['id']/'transitions.npz')
        learner.replay.add(data['states'],data['actions'],expert=r['controller']=='train-demo-retarget')
        used.append('focus/'+r['id'])
    return used


def restore(learner,path):
    x=torch.load(path,map_location=learner.device,weights_only=False)
    for key in ['actor','world','critic','actor_opt','world_opt','critic_opt']:
        getattr(learner,key).load_state_dict(x[key])
    learner.updates=x['updates']
    if 'replay_rng_state' in x:learner.replay.rng.bit_generator.state=x['replay_rng_state']
    if 'torch_rng_state' in x:torch.set_rng_state(x['torch_rng_state'].cpu())
    if x.get('cuda_rng_state') is not None and str(learner.device).startswith('cuda'):
        torch.cuda.set_rng_state_all([state.cpu() for state in x['cuda_rng_state']])
    return x['metadata']


def export(learner,out,metadata):
    class Deployment(torch.nn.Module):
        def __init__(self,actor):super().__init__();self.actor=actor
        def forward(self,s):
            a=self.actor(s);return torch.cat([a[...,:3],torch.where(a[...,3:4]>=0,1.,-1.)],-1)
    actor=deepcopy(learner.actor).cpu().eval()
    torch.jit.trace(Deployment(actor).eval(),torch.zeros(1,39)).save(str(out/'policy.ts.tmp'))
    (out/'policy.ts.tmp').replace(out/'policy.ts')
    torch.save(dict(actor=actor.state_dict(),metadata=metadata),out/'policy.pt.tmp')
    (out/'policy.pt.tmp').replace(out/'policy.pt')


def train_update(root,round_number,device='cuda'):
    torch.set_num_threads(4)
    plan=read_plan(root);cfg=dict(plan['learning']);amendment=amendment_for_round(root,plan,round_number)
    if amendment:cfg.update(amendment['learning'])
    checkpoint=root/f'checkpoints/round-{round_number:04d}.pt'
    checkpoint.parent.mkdir(exist_ok=True)
    learner=Learner(device=device,seed=plan['seed'],binary_gripper=True)
    if checkpoint.exists():
        metadata=restore(learner,checkpoint)
        if metadata['protocol_sha256']!=plan['sha256']:raise ValueError('Checkpoint protocol mismatch')
    else:
        if round_number==0:
            pilot=Path(plan['pilot_train'])
            if sha(pilot/'bc-baseline.pt')!=json.loads((pilot/'checkpoint-freeze.json').read_text())['bc_sha256']:
                raise ValueError('Pilot BC changed')
            restore(learner,pilot/'bc-baseline.pt')
            shutil.copy2(pilot/'bc-baseline.pt',root/'bc-baseline.pt')
            shutil.copy2(pilot/'bc-policy.ts',root/'bc-policy.ts')
        else:restore(learner,root/f'checkpoints/round-{round_number-1:04d}.pt')
        reference=None
        if amendment and cfg.get('update_method')=='conservative':
            anchor=root/amendment['reference_checkpoint']
            if sha(anchor)!=amendment['reference_sha256']:raise ValueError('Frozen trust reference changed')
            reference=deepcopy(learner.actor)
            reference.load_state_dict(torch.load(anchor,map_location=device,weights_only=False)['actor'])
            reference.eval()
            if round_number==amendment['effective_round'] and amendment.get('reset_actor_to_reference'):
                learner.actor.load_state_dict(reference.state_dict())
                learner.actor_opt=torch.optim.Adam(learner.actor.parameters(),lr=cfg['rl_learning_rate'])
        used=load_replay(learner,root,plan,round_number)
        if round_number:
            completed={r['id'] for r in rows(root/'train/episodes.jsonl')}
            if not {f'train-{round_number:04d}-{i:02d}' for i in range(8)}<=completed:
                raise ValueError('Cannot update before all collection outcomes are recorded')
        for group in learner.actor_opt.param_groups:group['lr']=cfg['rl_learning_rate']
        began=time.time()
        world=learner.fit_world(steps=cfg['world_initial_steps'] if round_number==0 else cfg['world_steps_per_round'],horizon=cfg['world_window'])
        validation=learner.validation_rmse()
        if round_number==0:metrics={}
        elif reference is not None:
            metrics=learner.improve_conservative(reference,steps=cfg['policy_updates_per_round'],
                horizon=cfg['imagination_horizon'],bc_weight=cfg['bc_weight'],trust_weight=cfg['trust_weight'])
        else:metrics=learner.improve(steps=cfg['policy_updates_per_round'],horizon=cfg['imagination_horizon'],bc_weight=cfg['bc_weight'])
        metadata=dict(algorithm='SuperMimic-inspired iterative on-policy cloth RL with TRAIN BC anchor',
            protocol_sha256=plan['sha256'],round=round_number,seed=plan['seed'],replay_episode_ids=used,
            actor_parameters=sum(p.numel() for p in learner.actor.parameters()),binary_gripper=True,
            train_transitions=sum(len(e['a']) for e in learner.replay.episodes if not e['holdout']),
            holdout_transitions=sum(len(e['a']) for e in learner.replay.episodes if e['holdout']),
            total_rl_gradient_updates=round_number*cfg['policy_updates_per_round'],
            world_train_mse=world,world_holdout_rmse=validation,metrics=metrics,started_at=began,finished_at=time.time())
        metadata.update(learning=cfg,learning_revision=amendment['id'] if amendment else 'v1',
            amendment_sha256=amendment['sha256'] if amendment else None,
            source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
        if amendment and amendment.get('reset_actor_to_reference'):
            metadata['rl_updates_since_actor_reset']=(round_number-amendment['effective_round']+1)*cfg['policy_updates_per_round']
        learner.save(checkpoint.with_suffix('.pt.tmp'),metadata);checkpoint.with_suffix('.pt.tmp').replace(checkpoint)
    write(root/f'checkpoints/round-{round_number:04d}.json',metadata)
    all_metrics=[json.loads(p.read_text()) for p in sorted((root/'checkpoints').glob('round-*.json'))]
    temp=root/'metrics.jsonl.tmp';temp.write_text('\n'.join(json.dumps(x) for x in all_metrics)+'\n');temp.replace(root/'metrics.jsonl')
    export(learner,root,metadata)
    write(root/'latest.json',dict(round=round_number,path=str(checkpoint),sha256=sha(checkpoint),updated=time.time()))
    return metadata


def cases(plan,phase,round_number):
    training=phase=='collect';total=8 if training else plan['effectiveness_gate']['confirmation_trials' if phase=='confirm' else 'probe_trials']
    output=[]
    for i in range(total):
        target=TARGETS[i%4]
        position=((round_number+i//4)%2 if training else (i//4)%2)
        seed=(1000000+round_number*8+i if training else 2500000+round_number*32+i if phase=='confirm' else 2000000+i)
        output.append(dict(id=f'{phase}-{round_number:04d}-{i:02d}',seed=seed,garment=plan['garment'],subtask=target,
            distribution='training_seen_condition',position=f'seen-{position}',tcp_offset_m=plan['positions'][position],
            garment_position=plan['fixed_garment_position']))
    return output


def score(evaluations,expected,gate):
    if len(evaluations)!=expected or len({r['id'] for r in evaluations})!=expected:raise ValueError('Incomplete/duplicate physical evaluation')
    by_target={t:dict(successes=sum(bool(r['success']) for r in evaluations if r['target']==t),total=sum(r['target']==t for r in evaluations)) for t in TARGETS}
    rate=sum(bool(r['success']) for r in evaluations)/expected
    passed=rate>=gate['success_rate'] and all(v['total'] and v['successes']/v['total']>=gate['per_target_minimum_rate'] for v in by_target.values())
    return dict(successes=sum(bool(r['success']) for r in evaluations),total=expected,rate=rate,by_target=by_target,passed=passed)


def simulator_run(env,out,args):
    from .formal import register_cases
    from .training import LocalGrasp,rollout,PregraspError,append
    root=args.focus_plan.parent;plan=read_plan(root)
    demo=json.loads((out/'demo.json').read_text());anchor={e['target']:e for e in demo['events']}
    local=LocalGrasp(env,out,demo);learner=Learner(seed=plan['seed'],binary_gripper=True)
    learner.load(args.checkpoint);learner.actor.eval();model_sha=sha(args.checkpoint)
    allcases=cases(plan,args.focus_phase,args.focus_round)
    register_cases(env,out,{'cases':allcases})
    done={r['id'] for r in rows(out/'episodes.jsonl')}
    snapshot=out/'source';snapshot.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,snapshot/path.name)
    for i,case in enumerate(allcases):
        if not args.case_start<=i<args.case_start+args.case_count:continue
        training=args.focus_phase=='collect';teacher=training and i<4
        eid=f'train-{args.focus_round:04d}-{i:02d}' if training else case['id']
        if eid in done:continue
        source_index=(args.focus_round-1)%len(plan['sources'])
        local.events=deepcopy(anchor)
        if teacher:
            source=plan['sources'][source_index];folder=Path(source['extract'])
            if sha(folder/'demonstration.npz')!=source['action_sha256']:raise ValueError('Frozen TRAIN commands changed')
            meta=json.loads((folder/'demo.json').read_text())
            local.events={e['target']:dict(e,quaternion_wxyz=anchor[e['target']]['quaternion_wxyz']) for e in meta['events']}
            local.demo_arrays=dict(np.load(folder/'demonstration.npz'))
        controller='train-demo-retarget' if teacher else 'learned-policy' if training else 'bc-policy' if args.focus_phase=='baseline' else 'rl-policy'
        role='train' if training else 'seen_confirmation' if args.focus_phase=='confirm' else 'seen_development'
        write(out/'status.json',dict(status='collecting' if training else 'evaluating',detail=eid,updated=time.time()))
        try:
            rollout(local,learner,case['subtask'],case['seed'],eid,controller,role=role,
                noise=plan['learning']['policy_exploration_sigma'] if training and not teacher else 0.,case=case,policy=controller)
        except PregraspError as exc:
            append(out/'episodes.jsonl',dict(id=eid,role=role,target=case['subtask'],seed=case['seed'],case_id=case['id'],
                policy=controller,controller=controller,reset_failed=True,success=False,reason=str(exc),steps=0,frames=local.frames))
        lm=env.scene_manager.layout_manager;name=lm.get_instance_name(label='target',env_idx=0);category,asset=lm.decode_inst_name(name)
        if category!='Top_Long' or asset!=9:raise ValueError('Seen training garment mismatch')
        append(out/'scene-audit.jsonl',dict(id=eid,asset=asset,instance=name,checkpoint_sha256=model_sha,
            source_index=source_index if teacher else None,round=args.focus_round,position=case['tcp_offset_m']))
