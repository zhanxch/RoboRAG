"""One demonstration initializes orientation; all dynamics data comes from PhysX."""
from copy import deepcopy
import json
from pathlib import Path
import time
import hashlib
import shutil

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
import torch

from .learning import Learner, reward
from .sim_worker import atomic_json

TARGETS=['left_sleeve','right_sleeve','left_hem','right_hem']


def append(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def matrix(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()


class LocalGrasp:
    def __init__(self,env,out,demo):
        self.env,self.out,self.demo=env,out,demo
        self.events={e['target']:e for e in demo['events']}
        self.frames=[]
    def reset(self,target,seed,episode_id,jitter=.015,case=None):
        self.target,self.side,self.episode_id=target,target.split('_')[0],episode_id
        self.rng=np.random.default_rng(seed)
        # Full reset includes actual deformable-particle state and zeroed robot controls.
        layout_seed=case['seed'] if case else getattr(self.env,'cloth_training_seed',self.env.env_seeds[0])
        self.env.reset(seed=[layout_seed])
        self.env.take_action_cnt=[0];self.env.end_flag=[False]
        lm=self.env.scene_manager.layout_manager
        name=lm.get_instance_name(label='target',env_idx=0)
        self.cloth=lm.get_scene_object(inst_name=name,env_idx=0)
        metadata=lm.get_instance_metadata(inst_name=name,env_idx=0)
        self.indices=np.array(metadata['passive']['functional'][target]['id'],dtype=int)
        points=self.cloth.sample_mesh_vertices()[0]
        self.initial_target=points[self.indices].mean(0)
        # Keep the same material points for the whole trial, including after folding.
        nearest=np.argsort(np.linalg.norm(points-self.initial_target,axis=1))[:8]
        self.patch_indices=nearest
        self.quat=np.array(self.events[target]['quaternion_wxyz']);self.quat/=np.linalg.norm(self.quat)
        self.offset=matrix(self.quat)@np.array([.145,0,0])
        self.robots={r.arm_name.split('_')[0]:r for r in self.env.robot_manager.robot_list if r.type=='target'}
        self.park={side:self.env.robot_manager.get_real_endpose(r)[0].copy() for side,r in self.robots.items()}
        self.grip=1.
        self.frames=[];self.step_index=0;self.close_step=None
        offset=case['tcp_offset_m'] if case else [self.rng.uniform(-jitter,jitter),self.rng.uniform(-jitter,jitter),.065]
        desired_tcp=self.initial_target+np.array(offset)
        desired_pose=np.r_[desired_tcp-self.offset,self.quat]
        start=self.park[self.side].copy()
        rotations=Rotation.from_quat(np.stack([start[3:],self.quat])[:,[1,2,3,0]])
        slerp=Slerp([0,1],rotations)
        # Preparation is explicit, excluded from the local-policy horizon and replay.
        for ratio in np.linspace(0,1,45)[1:]:
            q=slerp([ratio]).as_quat()[0][[3,0,1,2]]
            pose=np.r_[start[:3]+ratio*(desired_pose[:3]-start[:3]),q]
            self.dispatch(pose,1.)
        for _ in range(5):self.dispatch(desired_pose,1.)
        preparation_error=float(np.linalg.norm(self.tcp()-desired_tcp))
        if preparation_error>.025:
            raise PregraspError(f'Local reset failed to reach pregrasp: {preparation_error:.3f} m')
        self.previous_tcp=self.tcp()
        self.previous_target=self.point()
        self.start_tcp=self.previous_tcp.copy()
        self.initial_target=self.previous_target.copy()
        self.reference_hold_tcp=None
        self.actual_start_tcp_offset=(self.start_tcp-self.initial_target).tolist()
        self.capture(0)
        return self.state()
    def point(self):return self.cloth.sample_mesh_vertices()[0][self.indices].mean(0)
    def pose(self):return self.env.robot_manager.get_real_endpose(self.robots[self.side])[0].copy()
    def tcp(self):
        p=self.pose();return p[:3]+matrix(p[3:])@np.array([.145,0,0])
    def dispatch(self,pose,grip):
        action={}
        for side in ['left','right']:
            action[side+'_ee_pose']=(pose if side==self.side else self.park[side]).tolist()
            action[side+'_ee_joint_state']=[float(grip if side==self.side else 1.)]
        before=self.env.take_action_cnt[0]
        self.env.take_action(action)
        if self.env.take_action_cnt[0]!=before+1:
            raise RuntimeError('Simulator rejected or skipped a control step; do not log it as physics experience')
        self.grip=float(grip)
    def state(self):
        p=self.tcp();points=self.cloth.sample_mesh_vertices()[0];t=points[self.indices].mean(0)
        s=np.r_[(t-p)/.1,(t[2]-self.initial_target[2])/.1,self.grip,
                (p-self.initial_target)/.1,(p-self.previous_tcp)/.008,(t-self.previous_target)/.008,
                min(self.step_index/80,1),((points[self.patch_indices]-t)/.1).reshape(-1)]
        return s.astype(np.float32)
    def expert(self,state):
        tcp=self.tcp();t=self.point()
        if self.close_step is None:
            desired=t+np.array([0,0,.006])
            delta=desired-tcp
            grip=1.
            if np.linalg.norm(delta)<.009:
                self.close_step=self.step_index;grip=0.
        else:
            grip=0.
            delta=np.zeros(3)
            if self.step_index-self.close_step>=8:
                delta[2]=max(0,min(.006,self.initial_target[2]+.15-tcp[2]))
        return np.r_[np.clip(delta/.008,-1,1),2*grip-1].astype(np.float32)
    def demonstration_action(self):
        """Retarget recorded TRAIN TCP commands, then append a local lift/hold.

        Translation aligns the original closure TCP to this material point.
        No cloth-state labels are imputed into the HDF5. The resulting BC
        observations are measured in physical simulation after executing it.
        """
        if not hasattr(self,'demo_arrays'):
            self.demo_arrays=dict(np.load(self.out/'demonstration.npz'))
        event=self.events[self.target]
        poses=self.demo_arrays[f'action_{self.side}_ee_poses']
        closes=event['close_frame'];t=closes-20+self.step_index
        def tcp(p):return p[:3]+matrix(p[3:])@np.array([.145,0,0])
        if self.step_index<26:
            desired=tcp(poses[t])-tcp(poses[closes])+self.initial_target+[0,0,.006]
            g=float(self.demo_arrays[f'action_{self.side}_ee_joint_states'][t,0]>=.5)
            delta=desired-self.tcp()
        else:
            g=0.
            delta=np.zeros(3)
            if self.step_index>=34:
                delta[2]=max(0,min(.006,self.initial_target[2]+.15-self.tcp()[2]))
        return np.r_[np.clip(delta/.008,-1,1),2*g-1].astype(np.float32)
    def step(self,action):
        action=np.clip(np.asarray(action,dtype=np.float64),-1,1)
        if not np.isfinite(action).all():raise ValueError('Nonfinite policy action')
        self.previous_tcp=self.tcp();self.previous_target=self.point()
        p=self.pose();p[:3]+=action[:3]*.008;p[3:]=self.quat
        # Local movement envelope, enforced independently of the learned network.
        tcp=p[:3]+self.offset
        tcp=np.clip(tcp,self.initial_target+[-.07,-.07,-.005],self.initial_target+[.07,.07,.19])
        p[:3]=tcp-self.offset
        executed=action.copy();executed[:3]=(p[:3]-self.pose()[:3])/.008
        self.dispatch(p,(action[3]+1)/2)
        self.step_index+=1
        s=self.state()
        lift=float(s[3]*.1);gap=float(np.linalg.norm(s[:3])*.1)
        held=lift>=.08 and gap<.055 and self.grip<.35
        if self.step_index%4==0:self.capture(self.step_index)
        return s,executed.astype(np.float32),dict(lift=lift,gap=gap,held=held)
    def capture(self,step):
        obs=self.env.get_obs()
        folder=self.out/'rollouts'/self.episode_id
        folder.mkdir(parents=True,exist_ok=True)
        row=dict(step=step)
        for cam,data in obs['vision'].items():
            image=Image.fromarray(np.asarray(data['color'])).resize((640,480))
            draw=ImageDraw.Draw(image)
            draw.rectangle([8,8,260,30],fill=(16,20,19))
            draw.text((14,13),f'{self.target} | step {step}',fill=(191,231,146))
            path=folder/f'{cam}_{step:04d}.jpg';image.save(path,quality=80)
            row[cam]=str(path.relative_to(self.out))
        self.frames.append(row)
        atomic_json(self.out/'live.json',dict(frames={k:v for k,v in row.items() if k!='step'},
                    target=self.target,step=step,episode=self.episode_id,updated=time.time()))


class PregraspError(RuntimeError):
    pass


def rollout(local,learner,target,seed,episode_id,controller,role='train',noise=0.,jitter=.015,case=None,policy=None):
    s=local.reset(target,seed,episode_id,jitter=jitter,case=case)
    states=[s];actions=[];max_lift=0.;hold=0;max_hold=0;total_reward=0
    for step in range(80):
        a=(local.expert(s) if controller=='demo-guided' else local.demonstration_action()
           if controller=='train-demo-retarget' else learner.action(s))
        if noise:
            a=a.copy();a[:3]+=local.rng.normal(0,noise,3)
        s,a,info=local.step(a)
        states.append(s);actions.append(a)
        max_lift=max(max_lift,info['lift']);hold=hold+1 if info['held'] else 0;max_hold=max(hold,max_hold)
        total_reward+=float(reward(torch.from_numpy(s),torch.from_numpy(a)))
    success=hold>=10  # final continuous retention, not a transient mid-episode lift
    row=dict(id=episode_id,target=target,seed=seed,role=role,controller=controller,
             steps=len(actions),success=success,max_lift_m=max_lift,max_hold_steps=max_hold,
             final_hold_steps=hold,reward=total_reward,frames=local.frames,
             reason='retained' if success else ('抬升后未保持' if max_lift>=.08 else '未达到抬升阈值'),jitter_m=jitter)
    row['actual_start_tcp_offset_m']=local.actual_start_tcp_offset
    if controller=='train-demo-retarget':
        row['demonstration_retarget_steps']=26
        row['lift_hold_extension_steps']=54
    if case:
        row.update(case_id=case['id'],policy=policy,distribution=case['distribution'],
                   position=case['position'],garment=case['garment'],tcp_offset_m=case['tcp_offset_m'])
    np.savez_compressed(local.out/'rollouts'/episode_id/'transitions.npz',states=states,actions=actions)
    from .export import export_run
    export_run(local.out)
    append(local.out/'episodes.jsonl',row)
    print('EPISODE',json.dumps({k:v for k,v in row.items() if k!='frames'}),flush=True)
    return states,actions,row


def run(env,out,args):
    torch.set_num_threads(4)
    demo=json.loads((out/'demo.json').read_text())
    learner=Learner(seed=args.seed,binary_gripper=args.binary_gripper)
    local=LocalGrasp(env,out,demo)
    config=dict(algorithm='SuperMimic-inspired local cloth MBRL + BC anchor',
                world_model_window=8,imagination_horizon=16,episodes=args.episodes,
                actor_parameters=sum(p.numel() for p in learner.actor.parameters()),
                observation_label='特权局部布料几何 + 本体状态',observation='39D local particle state; not RGB',
                action='4D bounded xyz increment + gripper; demonstration-conditioned fixed orientation',
                demo_sha256=demo['sha256'],demonstrations=1,seed=args.seed,
                success='target lift >= 0.08 m, TCP distance < 0.055 m, command closed, held for final 10 steps (0.4 s)',
                evaluation_scope='Same garment, held-out initial TCP offsets; not unseen garments or full fold benchmark',
                demo_usage='Four grasp orientations and local grasp timing seed a simulator exploration controller; HDF5 has no cloth state',
                sources=['https://arxiv.org/pdf/2610.28480'],device=torch.cuda.get_device_name(0))
    config.update(binary_gripper=args.binary_gripper,bc_steps=args.bc_steps,bc_weight=args.bc_weight,
                  policy_updates=args.policy_updates,warmup_episodes=args.warmup_episodes,
                  bootstrap=str(args.bootstrap) if args.bootstrap else None,
                  finite_horizon_mask=True,algorithm_revision='local-cloth-v3')
    snapshot=out/'source';snapshot.mkdir(exist_ok=True)
    source_hashes={}
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,snapshot/path.name)
        source_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    config['source_sha256']=source_hashes
    if args.checkpoint:
        metadata=learner.load(args.checkpoint)
        if args.mode=='evaluate':
            learner.binary_gripper=args.binary_gripper or metadata.get('binary_gripper',False)
            config['binary_gripper']=learner.binary_gripper
            config['algorithm']=metadata.get('algorithm','BC baseline (offline imitation initialization only)')
            config['evaluated_checkpoint_metadata']=metadata
        config['initial_checkpoint']=str(args.checkpoint)
        config['initial_checkpoint_sha256']=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    atomic_json(out/'experiment.json',config)
    sim_steps=0
    initial_results=[]
    if args.mode=='train':
        if (out/'episodes.jsonl').exists():raise RuntimeError('Run already contains episodes; use a fresh run directory to avoid mixing experiments')
        if args.bootstrap:
            bootstrap_demo=json.loads((args.bootstrap/'demo.json').read_text())
            if bootstrap_demo['sha256']!=demo['sha256']:raise ValueError('Bootstrap data must use the same single demonstration')
            loaded=0
            for line in (args.bootstrap/'episodes.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row['role']!='train':continue
                with np.load(args.bootstrap/'rollouts'/row['id']/'transitions.npz') as x:
                    n=int(row['id'].split('-')[-1])
                    learner.replay.add(x['states'],x['actions'],holdout=n in (5,6),
                                       expert=row['controller']=='demo-guided' or row['success'])
                    loaded+=len(x['actions'])
            config['reused_physics_transitions']=loaded
            config['bc_anchor']='demonstration-guided TRAIN episodes plus successful learned TRAIN episodes; no validation episodes'
            atomic_json(out/'experiment.json',config)
            atomic_json(out/'status.json',dict(status='learning',detail='拟合既有训练交互；排除所有验证回合',updated=time.time()))
            if args.bc_steps:learner.warm_start(args.bc_steps)
            learner.fit_world(steps=400)
            learner.save(out/'initial.pt',config)
            if args.evaluate_initial:
                atomic_json(out/'status.json',dict(status='evaluating',detail='训练前冻结的模仿初始化对照；不写入 replay',updated=time.time()))
                for i in range(args.evaluation_episodes):
                    _,_,row=rollout(local,learner,TARGETS[i%4],100000+args.seed+i,
                                    f'baseline-{i+1:03d}','bc-policy',role='baseline_validation',jitter=.02)
                    initial_results.append(row)
                atomic_json(out/'baseline-evaluation.json',dict(total=len(initial_results),successes=sum(r['success'] for r in initial_results)))
        elif args.episodes<args.warmup_episodes or args.warmup_episodes<8:
            raise ValueError('Without bootstrap, require at least 8 warmup episodes and enough total episodes')
        for episode in range(args.episodes):
            target=TARGETS[episode%4]
            controller='demo-guided' if episode<args.warmup_episodes else 'learned-policy'
            atomic_json(out/'status.json',dict(status='collecting',episode=episode+1,sim_steps=sim_steps,
                        detail=f'{target} · {controller}',updated=time.time()))
            states,actions,result=rollout(local,learner,target,args.seed+episode,f'train-{episode+1:03d}',controller,
                                         noise=.06 if episode<args.warmup_episodes else .02)
            sim_steps+=len(actions)
            # Whole episodes 5 and 6 are world-model validation only. Never sampled by the actor.
            holdout=episode in (4,5) and not args.bootstrap
            learner.replay.add(states,actions,holdout=holdout,expert=episode<args.warmup_episodes)
            if episode==args.warmup_episodes-1:
                learner.warm_start(args.bc_steps)
                learner.save(out/'bc-baseline.pt',config)
            if args.bootstrap or episode>=args.warmup_episodes-1:
                atomic_json(out/'status.json',dict(status='learning',episode=episode+1,sim_steps=sim_steps,
                            detail='拟合 8 步动力学，反传 16 步策略回报',updated=time.time()))
                wm_loss=learner.fit_world(steps=200 if episode==7 else 100)
                stats=learner.improve(steps=args.policy_updates,bc_weight=args.bc_weight)
                stats.update(update=learner.updates,sim_steps=sim_steps,wm_train_mse=wm_loss,
                             wm_validation_rmse=learner.validation_rmse(),updated=time.time())
                append(out/'metrics.jsonl',stats)
                learner.save(out/'latest.pt',config)
                learner.save(out/f'update-{learner.updates:03d}.pt',config)
        learner.save(out/'final.pt',config)
    if args.mode=='evaluate' and not args.checkpoint:raise ValueError('evaluate requires --checkpoint')
    # Fresh physical resets and unseen perturbation seeds. No learning during evaluation.
    atomic_json(out/'status.json',dict(status='evaluating',sim_steps=sim_steps,updated=time.time()))
    results=[]
    evaluation_controller='bc-policy' if config['algorithm'].startswith('BC baseline') else 'learned-policy'
    for index in range(args.evaluation_episodes):
        target=TARGETS[index%4]
        _,_,row=rollout(local,learner,target,100000+args.seed+index,f'validation-{index+1:03d}',
                        evaluation_controller,role='validation',jitter=.02)
        results.append(row)
    by_target={t:dict(total=sum(r['target']==t for r in results),
                      successes=sum(r['target']==t and r['success'] for r in results)) for t in TARGETS}
    atomic_json(out/'evaluation.json',dict(total=len(results),successes=sum(r['success'] for r in results),
                by_target=by_target,controller=evaluation_controller,
                description='同一衣物，独立初始夹爪扰动种子；最终抬升 ≥8 cm 并保持 ≥0.4 s。仅局部抓取，不是完整折叠评分。'))
    if initial_results:
        atomic_json(out/'comparisons.json',dict(protocol='同一组 8 个初始偏移种子；模型冻结，无在线更新。小样本探索性对照。',
                    baseline=dict(successes=sum(r['success'] for r in initial_results),total=len(initial_results)),
                    learned=dict(successes=sum(r['success'] for r in results),total=len(results)),
                    pairs=[dict(target=a['target'],seed=a['seed'],baseline=a['success'],learned=b['success'])
                           for a,b in zip(initial_results,results)]))
