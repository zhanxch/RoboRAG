"""Native cloth reference replay and joint-space policy execution, without attachments."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
from roborag.cloth_grasp.training import LocalGrasp, matrix
from roborag.cloth_grasp.formal import register_cases
from .protocol import sha, write, append, rows, freeze, verification as read_verification

MODULE_SOURCE=Path(__file__).read_text()
MODULE_SHA=__import__("hashlib").sha256(MODULE_SOURCE.encode()).hexdigest()
DT=.04

def trajectory(source, target, scale, offset, horizontal_radius=.06):
    demo=json.loads((source/'demo.json').read_text()); arrays=np.load(source/'demonstration.npz')
    e=next(x for x in demo['events'] if x['target']==target); side=e['side']; c=e['close_frame']
    poses=arrays[f'action_{side}_ee_poses']; grip=arrays[f'action_{side}_ee_joint_states'][:,0]
    rotations=Rotation.from_quat(poses[:,[4,5,6,3]])
    tcp=poses[:,:3]+rotations.apply(np.tile([.145,0,0],(len(poses),1)))
    opened=np.flatnonzero(grip[c+1:]>.5); end=c+1+int(opened[0]) if len(opened) else len(poses)
    outside=np.flatnonzero(np.linalg.norm(tcp[c:end,:2]-tcp[c,:2],axis=1)>horizontal_radius)
    local_end=c+int(outside[0]) if len(outside) else end
    peak=c+int(np.argmax(tcp[c:local_end,2]))
    if peak<=c: raise ValueError('No positive recorded lift')
    t=np.r_[np.linspace(c-20,c,25,endpoint=False),np.linspace(c,peak,30),np.repeat(peak,25)]
    xyz=np.stack([np.interp(t,np.arange(len(poses)),tcp[:,i]) for i in range(3)],-1)-tcp[c]
    xyz[:,2]=np.where(xyz[:,2]>0,xyz[:,2]*scale,xyz[:,2]); xyz+=np.array(offset)
    quat=Slerp(np.arange(len(poses)),rotations)(t).as_quat()[:,[3,0,1,2]]
    g=np.interp(t,np.arange(len(grip)),grip)
    return demo,xyz,quat,g,dict(close_frame=c,peak_frame=peak,release_frame=end,local_window_end=local_end,source_times=t.tolist())

class JointCloth(LocalGrasp):
    def begin(self, case):
        self.control_dt=float(self.env.obs_manager.dt)*int(self.env.obs_manager.collect_interval)
        if abs(self.control_dt-DT)>1e-6:raise ValueError(f'Native control period differs from frozen25Hz: {self.control_dt}')
        self.reset(case['target'],case['seed'],case['id'],case=case)
        rm=self.env.robot_manager; self.robot=self.robots[self.side]
        self.key=rm.robot_key[rm.robot_list.index(self.robot)]
        self.q0=rm.get_joint(self.robot,env_idx_list=[0])[0].copy()
        self.qscale=np.full(6,getattr(self,'action_scale',.6)); self.limits=rm.get_joint_limits(self.robot,env_idx_list=[0])[0]
        self.origin=self.initial_target.copy()
        self.ctx=np.r_[self.q0,self.qscale,self.limits[:,0],self.limits[:,1],self.origin,float(self.side=='right')].astype('float32')
        self.last_pose=self.pose(); self.last_points=self.points(); self.prev_action=np.r_[np.zeros(6),1.].astype('float32')
        self.last_open=self.opening(); self.last_aq=self.q0.copy()
        return self.measure(initial=True)
    def points(self):
        v=self.cloth.sample_mesh_vertices()[0]
        return np.r_[v[self.indices].mean(0)[None],v[self.patch_indices]]
    def opening(self):
        raw=float(self.env.robot_manager.get_end_effector_real_val(self.robot,env_idx_list=[0])[0][0])
        g=self.robot.gripper_move; a,b=self.robot.gripper_scale; x=(raw-a)/(b-a)
        return float(np.clip(x if g['sign']==1 else 1-x,0,1))
    def measure(self,initial=False):
        q=self.env.robot_manager.get_joint(self.robot,env_idx_list=[0])[0]
        qv=self.key.data.joint_vel[0,self.robot.arm_joint_indices].detach().cpu().numpy()
        op=self.opening(); p=self.pose(); R=matrix(p[3:]); oldR=matrix(self.last_pose[3:])
        tcp=p[:3]+R@np.array([.145,0,0]); oldtcp=self.last_pose[:3]+oldR@np.array([.145,0,0])
        points=self.points(); cv=(points-self.last_points)/DT
        omega=Rotation.from_matrix(R@oldR.T).as_rotvec()/DT/np.pi
        s=np.r_[(q-self.q0)/self.qscale,2*op-1,qv/5.,(op-self.last_open)*2/DT/5.,
            (tcp-self.origin)/.1,R[:,:2].T.reshape(-1),(tcp-oldtcp)/DT,omega,
            ((points-self.origin)/.1).reshape(-1),cv.reshape(-1),self.prev_action].astype('float32')
        if initial:s[7:14]=0;s[23:29]=0;s[56:83]=0
        self.last_pose=p; self.last_points=points; self.last_open=op
        if s.shape!=(90,): raise RuntimeError(f'Bad state {s.shape}')
        return s
    def joint_step(self, action):
        action=np.clip(np.asarray(action,dtype=float),-1,1)
        q=np.clip(self.q0+self.qscale*action[:6],self.limits[:,0],self.limits[:,1])
        executed=np.r_[(q-self.q0)/self.qscale,action[6]].astype('float32')
        command={}
        for side,r in self.robots.items():
            command[side+'_arm_joint_state']=(q if side==self.side else self.env.robot_manager.get_joint(r,env_idx_list=[0])[0]).tolist()
            command[side+'_ee_joint_state']=[float((action[6]+1)/2 if side==self.side else 1.)]
        before=self.env.take_action_cnt[0]; self.env.take_action(command)
        if self.env.take_action_cnt[0]!=before+1:raise RuntimeError('Rejected physical command')
        self.grip=(action[6]+1)/2; self.prev_action=executed; self.step_index+=1
        s=self.measure(); lift=float(self.point()[2]-self.origin[2]); gap=float(np.linalg.norm(self.point()-self.tcp()))
        done=not np.isfinite(s).all() or np.linalg.norm(self.tcp()-self.origin)>.30 or lift<-.15
        if self.step_index%4==0:self.capture(self.step_index)
        return s,executed,dict(lift=lift,gap=gap,held=lift>=.005 and gap<.055 and self.grip<.35 and self.opening()<.35,terminated=bool(done))
    def pose_action(self,tcp,quat,grip):
        pose=np.r_[tcp-matrix(quat)@np.array([.145,0,0]),quat]
        ik=self.env.robot_manager.solve_ik(pose,env_idx=0,robot=self.robot,trans='world')
        if ik['status']!='Success':raise RuntimeError('Reference IK failed')
        q=np.asarray(ik['joint_value']).reshape(-1)
        a=np.r_[(q-self.q0)/self.qscale,2*grip-1]
        if np.max(np.abs(a[:6]))>1.001:raise RuntimeError(f'Reference exceeds frozen joint envelope: {np.max(np.abs(a[:6]))}')
        return a

def simulator_run(env,out,args):
    torch.set_num_threads(4)
    root=Path(args.mimic_plan).parent; plan=freeze(root)
    source_folder=out/'source'/MODULE_SHA;source_folder.mkdir(parents=True,exist_ok=True);(source_folder/'physics.py').write_text(MODULE_SOURCE)
    phase=args.mimic_phase
    verification=read_verification(root)
    horizon=verification['confirmation_horizon'] if phase=='confirm' else 80
    hold_steps=verification['confirmation_final_hold_steps'] if phase=='confirm' else 25
    candidates=plan['candidates']
    if phase=='reference':jobs=candidates[args.case_start:args.case_start+args.case_count]
    else:
        selected=json.loads((root/'references.json').read_text())['selected']
        jobs=[]
        repeats=8 if phase=='confirm' else 4 if phase=='probe' else 1
        for repeat in range(repeats):
            for ref in selected:
                case=dict(ref['case']);case.update(id=f'{phase}-{args.focus_round:05d}-{repeat}-{case["target"]}',reference_id=ref['id'],reference_path=ref['path'],seed=case['seed'])
                position_index=(1 if phase=='check' else args.focus_round%2 if phase=='collect' else repeat%2)
                case['tcp_offset_m']=verification['positions'][position_index]
                case['position']=f'seen-position-{position_index}'
                jobs.append(case)
    register_cases(env,out,dict(cases=jobs))
    completed={x['id'] for x in rows(out/'episodes.jsonl')}
    if phase not in ['reference','check']:
        policy=torch.jit.load(str(root/'policy.ts'),map_location='cpu').eval()
        policy_hash=sha(root/'policy.ts')
    for case in jobs:
        if case['id'] in completed:continue
        if time.time()>=plan['deadline']:raise TimeoutError('Hard deadline')
        source=Path(plan['sources'][case['source_index']]['folder'])
        demo=json.loads((source/'demo.json').read_text());local=JointCloth(env,out,demo)
        local.action_scale=plan['adapter']['action_scale_rad']
        s=local.begin(case);ctx=local.ctx.copy(); states=[s]; actions=[]; terms=[];infos=[]
        refmeta={}; start=0
        rng=np.random.default_rng(plan['seed']+args.focus_round*100+jobs.index(case))
        if phase=='reference':
            _,xyz,quat,g,refmeta=trajectory(source,case['target'],case['vertical_scale'],case['contact_offset_m'])
        else:
            ref=np.load(case['reference_path']); reference=ref['states']; reference_actions=ref['actions']
            # Canonical per-reference action coordinates, including references captured before an adapter amendment.
            recorded=ref['context'].astype('float32')
            reset_delta=ctx[24:27]-recorded[24:27]
            append(root/'reset-diagnostics.jsonl',dict(id=case['id'],reference_id=case['reference_id'],actual_context=ctx.tolist(),reference_context=recorded.tolist(),origin_delta_m=reset_delta.tolist(),norm_m=float(np.linalg.norm(reset_delta)),tcp=local.tcp().tolist(),case=case,updated=time.time()))
            if np.linalg.norm(reset_delta)>.05:raise ValueError(f'Reference reset exceeds measured local cloth variability envelope: {reset_delta.tolist()}')
            local.ctx=recorded.copy();local.q0=recorded[:6].copy();local.qscale=recorded[6:12].copy();local.origin=recorded[24:27].copy()
            ctx=recorded;s=local.measure(initial=True);states=[s]
            prefix_history=[np.r_[s[:14],s[83:]]]
            if phase=='collect':
                weights=json.loads((root/'reset-weights.json').read_text()) if (root/'reset-weights.json').exists() else {}
                probs=weights.get(case['reference_id'],{}).get('probabilities',[1/3]*3)
                start=int(rng.choice([0,25,50],p=probs))
                for k in range(start):
                    s,_,info=local.joint_step(reference_actions[k])
                    prefix_history.append(np.r_[s[:14],s[83:]])
                    if info['terminated']:raise RuntimeError('Reference prefix terminated')
                states=[s]
        history=np.repeat(np.r_[s[:14],s[83:]][None],10,axis=0)
        if phase!='reference':
            history=np.asarray(([prefix_history[0]]*10+prefix_history)[-10:])
        histories=[history.copy()]
        for t in range(start,horizon):
            if phase=='reference':a=local.pose_action(local.origin+xyz[t],quat[t],g[t])
            elif phase=='check':a=reference_actions[t]
            else:
                indices=np.minimum(np.arange(t+1,t+11),len(reference)-1)
                with torch.no_grad():a=policy(torch.tensor(history[None],dtype=torch.float32),torch.tensor(reference[indices,:83][None],dtype=torch.float32))[0].numpy()
                if phase=='collect':a=a+rng.normal(0,plan['method']['action_noise'],7)
            s,a,info=local.joint_step(a); states.append(s);actions.append(a);infos.append(info);terms.append(float(info['terminated']))
            history=np.concatenate([history[1:],np.r_[s[:14],s[83:]][None]],0); histories.append(history.copy())
            if info['terminated']:break
        folder=out/'rollouts'/case['id'];folder.mkdir(parents=True,exist_ok=True)
        path=folder/'transitions.npz'
        np.savez_compressed(path,states=np.array(states),actions=np.array(actions),terminated=np.array(terms),
            histories=np.array(histories),context=ctx,start_frame=start,reference_id=case.get('reference_id',case['id']))
        from .metrics import grasp_metrics
        verified=grasp_metrics(states,actions,start,horizon=horizon,hold_steps=hold_steps,terminated=any(terms))
        row=dict(id=case['id'],role='reference_feasibility' if phase=='reference' else 'reference_reset_check' if phase=='check' else ('train' if phase=='collect' else ('seen_confirmation' if phase=='confirm' else 'seen_development')),
            target=case['target'],case=case,steps=len(actions),start_frame=start,control_dt_s=local.control_dt,
            **verified,frames=local.frames,verification_horizon=horizon,required_hold_steps=hold_steps,verification_protocol_sha256=verification['sha256'],
            transition_path=str(path),transition_sha256=sha(path),scene_sha256=sha(out/'test-layouts'/f'{case["id"]}.json'),
            source_sha256=plan['sources'][case['source_index']]['hdf5_sha256'],protocol_sha256=plan['sha256'],
            reference_processing=refmeta,physics_source_sha256=MODULE_SHA,policy_sha256=None if phase in ['reference','check'] else policy_hash,updated=time.time())
        state_array=np.asarray(states)
        write(folder/'geometry.json',dict(tcp=(state_array[:,14:17]*.1).tolist(),target=(state_array[:,29:32]*.1).tolist(),patch=(state_array[:,32:56].reshape(-1,8,3)*.1).tolist()))
        append(out/'episodes.jsonl',row)
        write(root/'progress.json',dict(stage=phase,last_episode={k:v for k,v in row.items() if k!='frames'},updated=time.time()))
        print('MIMIC_EPISODE',json.dumps({k:v for k,v in row.items() if k not in ['frames','reference_processing','case']}),flush=True)
