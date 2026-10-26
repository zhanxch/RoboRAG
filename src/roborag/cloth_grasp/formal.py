"""One fixed training run followed by a pre-registered paired physical test."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .learning import Learner, advance_replay_rng
from .protocol import TARGETS, read, summarize
from .sim_worker import atomic_json
from .training import LocalGrasp, PregraspError, append, rollout


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register_cases(env, out, protocol):
    template = json.loads((out/'training-layout.json').read_text())
    folder = out/'test-layouts'
    folder.mkdir(exist_ok=True)
    for case in protocol['cases']:
        layout = deepcopy(template)
        garment = layout['Garment']['Top_Long'][0]
        garment['category_idx'] = case['garment']
        garment['default_pos'] = case['garment_position']
        garment['default_ori'] = [1., 0., 0., 0.]
        path = folder/(case['id']+'.json')
        atomic_json(path, layout)
        env.seed_manager.seed_info[case['seed']] = {'scene_layout':str(path)}


def export_actor(learner, out, config):
    actor = deepcopy(learner.actor).cpu().eval()
    class Deployment(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.actor = network
        def forward(self, state):
            a = self.actor(state)
            return torch.cat([a[..., :3], torch.where(a[..., 3:4]>=0,
                              torch.ones_like(a[..., 3:4]), -torch.ones_like(a[..., 3:4]))], -1)
    policy = Deployment(actor).eval()
    torch.jit.trace(policy, torch.zeros(1,39)).save(str(out/'policy.ts'))
    torch.save(dict(actor=actor.state_dict(),metadata=config),out/'policy.pt')
    test = torch.randn(17,39)
    torch.testing.assert_close(torch.jit.load(str(out/'policy.ts'))(test),policy(test))


def run(env, out, args):
    torch.set_num_threads(4)
    protocol = read(args.protocol)
    demo = json.loads((out/'demo.json').read_text())
    if demo['sha256'] != protocol['demo_sha256']:
        raise ValueError('TRAIN data differs from frozen protocol')
    settings = protocol['training']
    learner = Learner(seed=settings['seed'], binary_gripper=True)
    local = LocalGrasp(env,out,demo)
    if args.mode == 'train':
        previous=[]
        if (out/'episodes.jsonl').exists():
            previous=[json.loads(x) for x in (out/'episodes.jsonl').read_text().splitlines()]
            if any(r['role']=='test' for r in previous) or (out/'final.pt').exists():
                raise RuntimeError('Cannot resume training after final checkpoint or testing')
            if len(previous)<12 or not (out/'latest.pt').is_file():
                raise RuntimeError('Recovery requires a completed post-BC checkpoint')
        config = dict(algorithm='SuperMimic-inspired cloth MBRL + retargeted demonstration BC',
                      algorithm_revision='formal-v1', protocol_sha256=protocol['sha256'],
                      actor_parameters=sum(p.numel() for p in learner.actor.parameters()),
                      episodes=settings['episodes'], demonstrations=1, demo_sha256=demo['sha256'],
                      observation='39D privileged local cloth-particle geometry + robot state',
                      observation_label='39D 特权布料几何 + 机器人状态', action='xyz / 8mm + binary gripper',
                      binary_gripper=True, seed=settings['seed'],
                      demo_usage='Original TRAIN action TCP trajectories and gripper commands in 4 windows, translated to local cloth targets and bounded to 8mm/axis. First 26 steps per retarget rollout use these commands; 54 steps append a scripted lift/hold. Real physics supplies cloth observations. Four additional geometric-teacher rollouts augment coverage.',
                      retarget_unique_arm_frames=104, warmup_rollouts=12, rl_rollouts=8,
                      success=protocol['criterion'], evaluation_scope=protocol['scope'],
                      device=torch.cuda.get_device_name(0), started_at=time.time(),
                      sources=['https://arxiv.org/pdf/2610.28480'])
        if previous:
            config=json.loads((out/'experiment.json').read_text())
            checkpoint=torch.load(out/'latest.pt',map_location=learner.device,weights_only=False)
            if checkpoint['updates']!=len(previous)-11:
                raise RuntimeError('Checkpoint and completed rollout count differ; manual recovery required')
            learner.load(out/'latest.pt')
            for name in ['actor_opt','world_opt','critic_opt']:
                getattr(learner,name).load_state_dict(checkpoint[name])
            # Version 1 stored optimizers but not replay RNG. Replay.sample makes
            # exactly two independent integer draws per sample. Reconstruct its
            # progress from the fixed schedule and complete episode boundaries.
            def advance(count,horizon,calls):
                advance_replay_rng(learner.replay,count,horizon,calls)
            for i,row in enumerate(previous):
                if row['id']!=f'train-{i+1:03d}':raise RuntimeError('Non-contiguous recovery episodes')
                data=np.load(out/'rollouts'/row['id']/'transitions.npz')
                learner.replay.add(data['states'],data['actions'],holdout=row['role']=='world_holdout',expert=i<12)
                if i==11:advance(10,1,settings['bc_steps'])
                if i>=11:
                    advance(i-1,settings['world_window'],300 if i==11 else 100)
                    for _ in range(settings['policy_updates_per_round']):
                        advance(i-1,1,1);advance(10,1,1)
                    advance(2,8,1)
            if 'replay_rng_state' in checkpoint:
                if learner.replay.rng.bit_generator.state!=checkpoint['replay_rng_state']:
                    raise RuntimeError('Recovered replay RNG does not match saved state')
            config.setdefault('recoveries',[]).append(dict(after_episode=len(previous),time=time.time(),
                checkpoint_sha256=file_hash(out/'latest.pt'),optimizers_restored=True,
                replay_rng='reconstructed from exact fixed sampling schedule',reason='Renderer out of device memory under shared GPU use'))
        source = out/('recovery-source' if previous else 'source')
        source.mkdir(exist_ok=True)
        hash_key='recovery_source_sha256' if previous else 'source_sha256'
        config[hash_key] = {}
        for path in Path(__file__).parent.glob('*.py'):
            shutil.copy2(path,source/path.name)
            config[hash_key][path.name] = file_hash(path)
        config['settings'] = settings
        atomic_json(out/'experiment.json',config)
        for i in range(len(previous),settings['episodes']):
            controller = ('train-demo-retarget' if i<8 else 'demo-guided' if i<12 else 'learned-policy')
            holdout = i+1 in settings['world_holdout_episode_numbers']
            atomic_json(out/'status.json',dict(status='collecting',episode=i+1,sim_steps=i*80,
                        detail=f"{TARGETS[i%4]} · {controller} · {'动力学留出' if holdout else '训练'}",updated=time.time()))
            states,actions,row = rollout(local,learner,TARGETS[i%4],settings['seed']+i,
                            f'train-{i+1:03d}',controller,role='world_holdout' if holdout else 'train',
                            noise=.02 if i>=12 else 0.)
            learner.replay.add(states,actions,holdout=holdout,expert=i<12)
            if i==11:
                atomic_json(out/'status.json',dict(status='learning',detail='用示范重定向与几何教师轨迹进行 BC 初始化',sim_steps=960,updated=time.time()))
                bc_loss=learner.warm_start(settings['bc_steps'])
                learner.save(out/'bc-baseline.pt',dict(config,algorithm='BC baseline',bc_loss=bc_loss))
                atomic_json(out/'checkpoint-freeze.json',dict(bc_sha256=file_hash(out/'bc-baseline.pt'),time=time.time()))
            if i>=11:
                atomic_json(out/'status.json',dict(status='learning',episode=i+1,sim_steps=(i+1)*80,
                            detail='拟合 8 步世界模型；反传 16 步策略回报',updated=time.time()))
                loss=learner.fit_world(steps=300 if i==11 else 100,horizon=settings['world_window'])
                stats=learner.improve(steps=settings['policy_updates_per_round'],
                                      horizon=settings['imagination_horizon'],bc_weight=settings['bc_weight'])
                stats.update(update=learner.updates,sim_steps=(i+1)*80,wm_train_mse=loss,
                             wm_validation_rmse=learner.validation_rmse(),updated=time.time())
                append(out/'metrics.jsonl',stats)
                learner.save(out/'latest.pt',config)
        config['training_completed_at']=time.time()
        config['train_transitions']=18*80
        config['world_holdout_transitions']=2*80
        config['total_local_training_physics_steps']=20*80
        learner.save(out/'final.pt',config)
        export_actor(learner,out,config)
        atomic_json(out/'experiment.json',config)
        atomic_json(out/'checkpoint-freeze.json',dict(bc_sha256=file_hash(out/'bc-baseline.pt'),
                    rl_sha256=file_hash(out/'final.pt'),policy_sha256=file_hash(out/'policy.ts'),
                    protocol_sha256=protocol['sha256'],time=time.time()))
        return
    if args.mode!='evaluate':
        raise ValueError(args.mode)
    frozen=json.loads((out/'checkpoint-freeze.json').read_text())
    models={'bc':out/'bc-baseline.pt','rl':out/'final.pt'}
    for name,path in models.items():
        if file_hash(path)!=frozen[name+'_sha256']:
            raise ValueError('Frozen checkpoint modified before evaluation')
    register_cases(env,out,protocol)
    # Evaluation-only provenance instrumentation, independent of model updates.
    evaluation_source=out/'evaluation-source'
    evaluation_source.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,evaluation_source/path.name)
    cases=protocol['cases'][args.case_start:args.case_start+args.case_count]
    existing=[json.loads(x) for x in (out/'episodes.jsonl').read_text().splitlines()]
    completed={(r.get('policy'),r.get('case_id')) for r in existing if r['role']=='test'}
    for case in cases:
        # Alternate order to reduce systematic ordering effects; reset both models.
        order=['bc','rl'] if case['seed']%2 else ['rl','bc']
        for policy in order:
            if (policy,case['id']) in completed:
                continue
            learner.load(models[policy])
            learner.actor.eval()
            eid=f"test-{policy}-{case['id']}"
            atomic_json(out/'status.json',dict(status='evaluating',detail=f"{case['id']} · 衣物 {case['garment']} · {case['subtask']} · {case['position']} · {policy.upper()}",
                        case_id=case['id'],policy=policy,test_completed=len(completed),test_total=48,updated=time.time()))
            try:
                rollout(local,learner,case['subtask'],case['seed'],eid,policy+'-policy',
                        role='test',case=case,policy=policy)
            except PregraspError as exc:
                row=dict(id=eid,case_id=case['id'],policy=policy,role='test',target=case['subtask'],
                         seed=case['seed'],garment=case['garment'],distribution=case['distribution'],
                         position=case['position'],tcp_offset_m=case['tcp_offset_m'],success=False,
                         reason=str(exc),steps=0,frames=local.frames,max_lift_m=0.,max_hold_steps=0,
                         final_hold_steps=0,controller=policy+'-policy',reset_failed=True)
                append(out/'episodes.jsonl',row)
            lm=env.scene_manager.layout_manager
            instance=lm.get_instance_name(label='target',env_idx=0)
            category,asset_id=lm.decode_inst_name(instance)
            if category!='Top_Long' or asset_id!=case['garment']:
                raise RuntimeError(f'Loaded asset {instance} differs from frozen case {case["id"]}')
            append(out/'test-scene-audit.jsonl',dict(case_id=case['id'],policy=policy,
                   instance=instance,asset_id=asset_id,vertices=len(local.cloth.sample_mesh_vertices()[0]),
                   layout_sha256=file_hash(out/'test-layouts'/(case['id']+'.json'))))
            completed.add((policy,case['id']))
    rows=[json.loads(x) for x in (out/'episodes.jsonl').read_text().splitlines()]
    if sum(r['role']=='test' for r in rows)==48:
        result=summarize(protocol,rows)
        atomic_json(out/'benchmark-results.json',result)
        rl=result['models']['rl']
        atomic_json(out/'evaluation.json',dict(total=rl['total'],successes=rl['successes'],
                    by_target=rl['groups']['target'],controller='rl-policy',
                    description='预先冻结的 24 个案例：4 部位 × 3 夹爪位置 × 2 件衣物；最终抬升 ≥8 cm 并保持 ≥0.4 s。'))
        atomic_json(out/'comparisons.json',result)
