"""SuperMimic Algorithm 1: multistep dynamics, analytic actor gradients, lambda critic.

The robot/cloth state and analytic tracking heads are adapter choices. There is
no supervised action loss, teacher action mixing, PPO, or target critic.
"""
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

S=90; O=21; G=83; C=28; A=7

def mlp(inp,widths,out):
    layers=[]
    for width in widths: layers.extend([nn.Linear(inp,width),nn.ELU()]);inp=width
    layers.append(nn.Linear(inp,out))
    return nn.Sequential(*layers)

def observation(s):return torch.cat([s[...,:14],s[...,83:]],-1)

class Actor(nn.Module):
    def __init__(self,config):
        super().__init__();self.encoder=mlp(10*G,config['encoder_widths'],32)
        self.decoder=mlp(10*O+32,config['decoder_widths'],A)
        nn.init.uniform_(self.decoder[-1].weight,-1e-3,1e-3);nn.init.zeros_(self.decoder[-1].bias)
    def forward(self,history,goal):
        z=torch.tanh(self.encoder(goal.flatten(-2)))
        return torch.tanh(self.decoder(torch.cat([history.flatten(-2),z],-1)))

class World(nn.Module):
    def __init__(self,config):
        super().__init__();self.net=mlp(S+A+C,config['world_widths'],84)
        nn.init.uniform_(self.net[-1].weight,-1e-3,1e-3);nn.init.zeros_(self.net[-1].bias)
        # Low initial hazard, not a hardcoded nontermination outcome.
        with torch.no_grad():self.net[-1].bias[-1]=-5.
    def forward(self,s,a,context):
        x=self.net(torch.cat([s,a,context],-1))
        return torch.cat([s[...,:83]+x[...,:83],a],-1),x[...,-1]

class Critic(nn.Module):
    def __init__(self,config):
        super().__init__();self.net=mlp(S+10*G+C,config['critic_widths'],1)
    def forward(self,s,g,c):return self.net(torch.cat([s,g.flatten(-2),c],-1)).squeeze(-1)

def rotmat(x):
    u=F.normalize(x[...,:3],dim=-1,eps=1e-6)
    v=F.normalize(x[...,3:]-u*(u*x[...,3:]).sum(-1,keepdim=True),dim=-1,eps=1e-6)
    return torch.stack([u,v,torch.cross(u,v,dim=-1)],-1)

def tracking(next_state,reference,action,previous_action,context):
    """Analytic cloth-adapter motion tracking, explicitly dimensioned in SI."""
    s,g=next_state,reference
    p=torch.cat([s[...,14:17],s[...,29:56]],-1)*.1
    gp=torch.cat([g[...,14:17],g[...,29:56]],-1)*.1
    position=torch.exp(-((p-gp).reshape(*p.shape[:-1],10,3).square().sum(-1)).mean(-1)/.03**2)
    r=rotmat(s[...,17:23]).transpose(-1,-2)@rotmat(g[...,17:23])
    # atan2 avoids acos' singular derivative at perfect tracking.
    skew=torch.stack([r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]],-1)*.5
    angle=torch.atan2(torch.linalg.vector_norm(skew,dim=-1),(r.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5)
    orientation=torch.exp(-angle.square()/.4**2)
    v=torch.cat([s[...,23:26],s[...,56:83]],-1);gv=torch.cat([g[...,23:26],g[...,56:83]],-1)
    linear=torch.exp(-(v-gv).square().reshape(*v.shape[:-1],10,3).sum(-1).mean(-1))
    angular=torch.exp(-(s[...,26:29]-g[...,26:29]).square().sum(-1))
    scale=torch.cat([context[...,6:12],torch.full_like(context[...,:1],.088)],-1)
    joint=torch.exp(-((s[...,:7]-g[...,:7])*scale).square().mean(-1)/.4**2)
    physical_q=context[...,:6]+context[...,6:12]*s[...,:6]
    excess=F.relu(context[...,12:18]-physical_q)+F.relu(physical_q-context[...,18:24])
    # Paper reward terms are integrated over the control period; cloth runs at25Hz.
    return .04*(position+orientation+linear+angular+joint-.1*(action-previous_action).square().sum(-1)-2*excess.sum(-1))

def lambda_returns(rewards,terminated,values,gamma=.99,lam=.95):
    """Eq.3, last axis is time; values has one additional bootstrap element."""
    ret=values[...,-1];out=[]
    for t in reversed(range(rewards.shape[-1])):
        ret=rewards[...,t]+gamma*(1-terminated[...,t])*((1-lam)*values[...,t+1]+lam*ret)
        out.append(ret)
    return torch.stack(out[::-1],-1)

class Replay:
    def __init__(self,capacity=1000,seed=0):
        self.capacity=capacity;self.episodes=[];self.ids=[];self.rng=np.random.default_rng(seed)
    def add(self,episode,identity):
        if identity in self.ids:return
        if len(episode['actions'])<1:raise ValueError('Empty physical trajectory')
        self.episodes.append(episode);self.ids.append(identity)
        while sum(len(e['actions']) for e in self.episodes)>self.capacity:
            excess=sum(len(e['actions']) for e in self.episodes)-self.capacity
            first=self.episodes[0];n=len(first['actions'])
            if n-excess<1:self.episodes.pop(0);continue
            first=dict(first)
            for k in ['states','actions','terminated','histories']:first[k]=first[k][excess:]
            first['start_frame']=int(first['start_frame'])+excess;self.episodes[0]=first
    def sample(self,batch,horizon,refs,device):
        # Uniform over valid transitions, not over variable-length episodes.
        counts=np.array([len(e['actions']) for e in self.episodes]); total=int(counts.sum())
        if total<=0:raise ValueError('No valid replay windows')
        pick=self.rng.integers(total,size=batch);ends=np.cumsum(counts)
        samples=[]
        for p in pick:
            i=int(np.searchsorted(ends,p,side='right'));e=self.episodes[i]
            t=int(p-(ends[i-1] if i else 0));phase=int(e['start_frame'])+t
            ref=refs[str(e['reference_id'])]
            # Future frames always refer to a single motion; clamp final pose.
            goal=ref[np.minimum(np.arange(phase+1,phase+43),len(ref)-1),:83]
            available=min(horizon,len(e['actions'])-t)
            si=np.minimum(np.arange(t,t+horizon+1),len(e['states'])-1)
            ai=np.minimum(np.arange(t,t+horizon),len(e['actions'])-1)
            valid=np.arange(horizon)<available
            samples.append((e['states'][si],e['actions'][ai],e['terminated'][ai],e['histories'][t],e['context'],goal,80-phase,valid))
        arrays=[np.stack([s[i] for s in samples]) for i in range(8)]
        return [torch.as_tensor(a,dtype=torch.float32,device=device) for a in arrays]

class Learner:
    def __init__(self,config,seed=41001,device='cuda'):
        self.cfg=config;self.device=device;torch.manual_seed(seed);np.random.seed(seed)
        if str(device).startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32=True
            torch.set_float32_matmul_precision('high')
        self.actor=Actor(config).to(device);self.world=World(config).to(device);self.critic=Critic(config).to(device)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=config['lr'],betas=tuple(config['actor_betas']))
        self.world_opt=torch.optim.Adam(self.world.parameters(),lr=config['lr'])
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=config['lr'])
        self.replay=Replay(config['replay_capacity_steps'],seed+1);self.updates=0;self.microbatch=config['microbatch']
    def schedule(self):
        fraction=min(self.updates/self.cfg['cosine_updates'],1.)
        mult=self.cfg['lr_final_fraction']+(1-self.cfg['lr_final_fraction'])*.5*(1+math.cos(math.pi*fraction))
        for opt in [self.world_opt,self.actor_opt,self.critic_opt]:
            for group in opt.param_groups:group['lr']=self.cfg['lr']*mult
    def fit_world(self,refs):
        cfg=self.cfg;losses=[]
        for _ in range(cfg['world_batches']):
            self.world_opt.zero_grad(set_to_none=True);total=0.
            for offset in range(0,cfg['world_batch'],self.microbatch):
                b=min(self.microbatch,cfg['world_batch']-offset)
                states,actions,terms,_,c,_,_,valid=self.replay.sample(b,8,refs,self.device)
                s=states[:,0];loss=torch.zeros(b,device=self.device)
                for t in range(8):
                    s,logit=self.world(s,actions[:,t],c)
                    loss=loss+valid[:,t]*((s-states[:,t+1]).square().mean(-1)+F.binary_cross_entropy_with_logits(logit,terms[:,t],reduction='none'))
                loss=(loss/valid.sum(-1).clamp_min(1)).mean(); (loss*b/cfg['world_batch']).backward();total+=float(loss.detach())*b/cfg['world_batch']
            norm=nn.utils.clip_grad_norm_(self.world.parameters(),1.,error_if_nonfinite=True)
            self.world_opt.step();losses.append(total)
        return float(np.mean(losses))
    def improve(self,refs):
        cfg=self.cfg;batch=cfg['actor_batch'];h=cfg['imagination_horizon'];gamma=cfg['gamma']
        self.actor_opt.zero_grad(set_to_none=True);dataset=[];meanreturn=0.
        for net in [self.world,self.critic]:net.requires_grad_(False)
        try:
            for off in range(0,batch,self.microbatch):
                b=min(self.microbatch,batch-off)
                ss,_,_,history,c,goals,remaining,_=self.replay.sample(b,1,refs,self.device)
                s=ss[:,0];survival=torch.ones(b,device=self.device);objective=torch.zeros_like(survival)
                states=[];gseq=[];rewards=[];dones=[];values=[];valid=[]
                for t in range(h):
                    goal=goals[:,t:t+10];v=self.critic(s,goal,c)
                    states.append(s.detach());gseq.append(goal.detach());values.append(v.detach());valid.append((remaining>t).detach())
                    a=self.actor(history,goal);nxt,logit=self.world(s,a,c)
                    r=tracking(nxt,goals[:,t],a,s[:,83:],c)
                    # Time truncation at the reference end is analytic; hard state failures are learned.
                    done=torch.where(remaining<=t+1,torch.ones_like(logit),torch.sigmoid(logit))
                    r=r*(remaining>t)
                    objective=objective+(gamma**t)*survival*r
                    survival=survival*(1-done)
                    rewards.append(r.detach());dones.append(done.detach())
                    history=torch.cat([history[:,1:],observation(nxt)[:,None]],1);s=nxt
                tail=self.critic(s,goals[:,h:h+10],c)
                objective=objective+(gamma**h)*survival*tail
                (-objective.mean()*b/batch).backward();meanreturn+=float(objective.detach().mean())*b/batch
                values.append(tail.detach())
                target=lambda_returns(torch.stack(rewards,1),torch.stack(dones,1),torch.stack(values,1),gamma,cfg['gae_lambda'])
                valid=torch.stack(valid,1)
                dataset.append(tuple(x[valid].cpu() for x in (torch.stack(states,1),torch.stack(gseq,1),c[:,None].expand(-1,h,-1),target)))
        finally:
            for net in [self.world,self.critic]:net.requires_grad_(True)
        norm=nn.utils.clip_grad_norm_(self.actor.parameters(),1.,error_if_nonfinite=True)
        for p in self.actor.parameters():
            if p.grad is not None:p.grad.clamp_(-.1,.1)
        self.actor_opt.step()
        # Frozen targets from this rollout, no EMA target critic.
        data=[torch.cat([d[i] for d in dataset]) for i in range(4)];n=len(data[0]);criticloss=0.
        for _ in range(cfg['critic_epochs']):
            order=torch.randperm(n)
            for indices in order.tensor_split(cfg['critic_minibatches']):
                self.critic_opt.zero_grad(set_to_none=True)
                for ix in indices.split(self.microbatch):
                    s,g,c,target=[x[ix].to(self.device) for x in data]
                    loss=F.mse_loss(self.critic(s,g,c),target)
                    (loss*len(ix)/len(indices)).backward();criticloss+=float(loss.detach())*len(ix)/n/cfg['critic_epochs']
                nn.utils.clip_grad_norm_(self.critic.parameters(),1.,error_if_nonfinite=True);self.critic_opt.step()
        self.updates+=1
        return dict(imagined_return=meanreturn,actor_grad_norm=float(norm),critic_loss=criticloss)
    def save(self,path,metadata):
        payload=dict(metadata=metadata,updates=self.updates,microbatch=self.microbatch,
            replay=self.replay,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],numpy_rng=np.random.get_state())
        for name in ['actor','world','critic','actor_opt','world_opt','critic_opt']:payload[name]=getattr(self,name).state_dict()
        tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
    def load(self,path,protocol_sha):
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p['metadata']['protocol_sha256']!=protocol_sha:raise ValueError('Checkpoint protocol mismatch')
        for name in ['actor','world','critic','actor_opt','world_opt','critic_opt']:getattr(self,name).load_state_dict(p[name])
        # Optimizer state must follow parameter devices after loading through CPU.
        for opt in [self.actor_opt,self.world_opt,self.critic_opt]:
            for state in opt.state.values():
                for k,v in state.items():
                    if torch.is_tensor(v) and k!='step':state[k]=v.to(self.device)
        self.replay=p['replay'];self.updates=p['updates'];self.microbatch=p['microbatch']
        torch.set_rng_state(p['torch_rng']);np.random.set_state(p['numpy_rng'])
        if p['cuda_rng']:torch.cuda.set_rng_state_all(p['cuda_rng'])
        return p['metadata']
    def export(self,path):
        actor=deepcopy(self.actor).cpu().eval()
        ts=torch.jit.trace(actor,(torch.zeros(1,10,O),torch.zeros(1,10,G)))
        tmp=path.with_suffix('.tmp');ts.save(str(tmp));tmp.replace(path)
