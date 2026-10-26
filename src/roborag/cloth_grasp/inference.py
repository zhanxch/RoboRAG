"""CPU-capable TorchScript inference; no simulator installation required.

The input requires measured/estimated material geometry. RGB pixels are not a
valid substitute. The robot adapter must supply the demonstrated orientation,
TCP convention, IK and workspace limits used by the training environment.
"""
import numpy as np
import torch


def encode_state(*, tcp, target, initial_target, previous_tcp, previous_target,
                 gripper_command, step, patch):
    tcp,target,origin,prev_tcp,prev_target=[np.asarray(v,dtype=np.float32) for v in
        [tcp,target,initial_target,previous_tcp,previous_target]]
    points=np.asarray(patch,dtype=np.float32)
    if any(v.shape!=(3,) for v in [tcp,target,origin,prev_tcp,prev_target]) or points.shape!=(8,3):
        raise ValueError('Expected five xyz vectors and eight persistent material points')
    if not 0<=gripper_command<=1 or not 0<=step<80:
        raise ValueError('Gripper command must be [0,1]; call the policy only within its 80-step horizon')
    state=np.r_[(target-tcp)/.1,(target[2]-origin[2])/.1,gripper_command,
                (tcp-origin)/.1,(tcp-prev_tcp)/.008,(target-prev_target)/.008,
                step/80,((points-target)/.1).reshape(-1)].astype(np.float32)
    if not np.isfinite(state).all():raise ValueError('Nonfinite observation')
    return state


class ClothPolicy:
    def __init__(self,model_path,device='cpu'):
        self.device=torch.device(device)
        self.model=torch.jit.load(str(model_path),map_location=self.device).eval()
    def __call__(self,state):
        value=torch.as_tensor(state,dtype=torch.float32,device=self.device)
        if value.shape[-1:]!=(39,) or value.ndim not in (1,2) or not torch.isfinite(value).all():
            raise ValueError('Expected a finite 39D state or B x 39 batch')
        single=value.ndim==1
        if single:value=value.unsqueeze(0)
        with torch.inference_mode():
            result=self.model(value).cpu().numpy()
        return result[0] if single else result
    def local_command(self,state):
        action=self(state)
        return dict(delta_xyz_m=action[...,:3]*.008,gripper_open=(action[...,3]+1)/2)
