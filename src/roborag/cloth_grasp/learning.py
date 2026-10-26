"""Short-horizon analytic policy gradients through learned cloth dynamics.

Adapted from SuperMimic's collect / multi-step fit / BPTT loop. Smaller networks,
local state, analytic grasp reward and a demonstration-derived BC anchor are
explicit differences. The physics simulator is never differentiated.
"""
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STATE_DIM = 39
ACTION_DIM = 4


def mlp(inp, widths, out):
    layers = []
    for width in widths:
        layers += [nn.Linear(inp, width), nn.ELU()]
        inp = width
    layers.append(nn.Linear(inp, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(STATE_DIM, [256, 256, 128], ACTION_DIM)
    def forward(self, state):
        return self.net(state).tanh()


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(STATE_DIM + ACTION_DIM, [256, 256, 256], STATE_DIM)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, state, action):
        pred = state + .2 * self.net(torch.cat([state, action], -1))
        # Commanded opening and the local horizon clock are known, not learned.
        return torch.cat([pred[..., :4], (action[..., 3:4] + 1) / 2,
                          pred[..., 5:14], (state[..., 14:15] + 1/80).clamp(max=1),
                          pred[..., 15:]], -1)


def reward(next_state, action):
    """Smooth reward; success is measured separately in real physics."""
    distance = next_state[..., :3].square().sum(-1)
    near = torch.exp(-distance / .12)
    height = next_state[..., 3].clamp(-.3, 1.5)
    closed = 1 - next_state[..., 4].clamp(0, 1)
    return .25*near + 2*height*near*closed - .01*action[..., :3].square().mean(-1)


@contextmanager
def frozen(*modules):
    params = [p for module in modules for p in module.parameters()]
    original = [p.requires_grad for p in params]
    try:
        for p in params: p.requires_grad_(False)
        yield
    finally:
        for p, value in zip(params, original): p.requires_grad_(value)


class Replay:
    """Complete episodes preserve time order and prevent cross-reset windows."""
    def __init__(self, device, seed=0):
        self.episodes = []
        self.device = device
        self.rng = np.random.default_rng(seed)
    def add(self, states, actions, *, holdout=False, expert=False):
        s, a = np.asarray(states, np.float32), np.asarray(actions, np.float32)
        if len(s) != len(a)+1 or not np.isfinite(s).all() or not np.isfinite(a).all():
            raise ValueError("Replay must contain finite, complete T+1 state / T action episodes")
        self.episodes.append(dict(s=torch.as_tensor(s,device=self.device),
                                 a=torch.as_tensor(a,device=self.device),
                                 holdout=holdout, expert=expert))
    def sample(self, batch=128, horizon=8, holdout=False, expert_only=False):
        eligible=[e for e in self.episodes if e['holdout']==holdout and len(e['a'])>=horizon
                  and (not expert_only or e['expert'])]
        if not eligible: raise ValueError("No eligible replay episodes")
        ss, aa = [], []
        for _ in range(batch):
            e=eligible[int(self.rng.integers(len(eligible)))];t=int(self.rng.integers(len(e['a'])-horizon+1))
            ss.append(e['s'][t:t+horizon+1]);aa.append(e['a'][t:t+horizon])
        return torch.stack(ss),torch.stack(aa)


def advance_replay_rng(replay, eligible_count, horizon, calls, batch=128, episode_length=80):
    """Recover legacy sampler state without allocating samples or updating weights."""
    for _ in range(calls*batch):
        replay.rng.integers(eligible_count)
        replay.rng.integers(episode_length-horizon+1)


class Learner:
    def __init__(self, device='cuda', seed=0, binary_gripper=False):
        torch.manual_seed(seed)
        self.device=device
        self.actor=Actor().to(device)
        self.world=WorldModel().to(device)
        self.critic=mlp(STATE_DIM,[128,128],1).to(device)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=1e-4)
        self.world_opt=torch.optim.Adam(self.world.parameters(),lr=3e-4)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=3e-4)
        self.replay=Replay(device,seed)
        self.updates=0
        self.gamma=.97
        self.binary_gripper=binary_gripper
    def actuate(self,action):
        if not self.binary_gripper:return action
        g=action[...,3:4]
        # Hard forward decision, straight-through backward estimator. This is
        # an explicit hybrid-control extension, not SuperMimic's exact gradient.
        hard=torch.where(g>=0,torch.ones_like(g),-torch.ones_like(g))
        return torch.cat([action[...,:3],g+(hard-g).detach()],-1)
    def action(self,state):
        with torch.no_grad(): return self.actuate(self.actor(torch.as_tensor(state,dtype=torch.float32,device=self.device))).cpu().numpy()
    def warm_start(self,steps=400):
        loss=None
        for _ in range(steps):
            s,a=self.replay.sample(horizon=1,expert_only=True)
            loss=F.mse_loss(self.actor(s[:,0]),a[:,0])
            self.actor_opt.zero_grad();loss.backward();self.actor_opt.step()
        return float(loss.detach())
    def fit_world(self,steps=100,horizon=8):
        losses=[]
        for _ in range(steps):
            states,actions=self.replay.sample(horizon=horizon)
            pred=states[:,0];loss=0
            for t in range(horizon):
                pred=self.world(pred,actions[:,t]);loss=loss+F.mse_loss(pred,states[:,t+1])
            loss=loss/horizon
            self.world_opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(self.world.parameters(),1.0);self.world_opt.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))
    @torch.no_grad()
    def validation_rmse(self,horizon=8):
        if not any(e['holdout'] for e in self.replay.episodes):return None
        s,a=self.replay.sample(batch=128,horizon=horizon,holdout=True)
        pred=s[:,0];loss=0
        for t in range(horizon):
            pred=self.world(pred,a[:,t]);loss+=F.mse_loss(pred,s[:,t+1])
        return float((loss/horizon).sqrt())
    def improve(self,steps=30,horizon=16,bc_weight=2.0):
        rows=[]
        for _ in range(steps):
            states,_=self.replay.sample(horizon=1)
            s=states[:,0].detach();chain=[];rewards=[];actions=[];continuations=[]
            with frozen(self.world,self.critic):
                survival=torch.ones(s.shape[0],device=s.device)
                objective=torch.zeros_like(survival)
                for _t in range(horizon):
                    chain.append(s);a=self.actuate(self.actor(s));actions.append(a)
                    s=self.world(s,a);r=reward(s,a);rewards.append(r)
                    objective=objective+(self.gamma**_t)*survival*r
                    continuation=(s[...,14]<1-1e-6).float();continuations.append(continuation)
                    survival=survival*continuation
                objective=objective+(self.gamma**horizon)*survival*self.critic(s).squeeze(-1)
                # BC is a stabilization term, and is reported separately from RL.
                es,ea=self.replay.sample(horizon=1,expert_only=True)
                bc=F.mse_loss(self.actor(es[:,0]),ea[:,0])
                loss=-objective.mean()/horizon+bc_weight*bc
                self.actor_opt.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),1.0);self.actor_opt.step()
            with torch.no_grad():
                target=self.critic(s.detach()).squeeze(-1)
                targets=[]
                for r,c in reversed(list(zip(rewards,continuations))):
                    target=r.detach()+self.gamma*c*target
                    targets.append(target)
                targets=torch.stack(list(reversed(targets)))
            pred=self.critic(torch.stack(chain).detach()).squeeze(-1)
            critic_loss=F.mse_loss(pred,targets)
            self.critic_opt.zero_grad();critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(),1.0);self.critic_opt.step()
            rows.append((float(objective.detach().mean()/horizon),float(bc.detach()),float(critic_loss.detach())))
        self.updates+=1
        return dict(zip(['imagined_return','bc_loss','critic_loss'],np.mean(rows,axis=0).tolist()))
    def improve_conservative(self,reference,steps=50,horizon=8,bc_weight=20.,trust_weight=100.):
        """Ablation: short model rollouts, BC action trust penalty, no value bootstrap.

        The reference is a separately frozen, physically evaluated policy. The
        trust term constrains all four raw outputs on imagined states, including
        the gripper logit before its discontinuous forward decision. No grip
        latch or change to simulator actions/success criteria is introduced.
        """
        rows=[]
        for _ in range(steps):
            states,_=self.replay.sample(horizon=1)
            s=states[:,0].detach()
            with frozen(self.world,reference):
                survival=torch.ones(s.shape[0],device=s.device)
                objective=torch.zeros_like(survival);trust=0.
                for t in range(horizon):
                    raw=self.actor(s)
                    trust=trust+F.mse_loss(raw,reference(s))
                    action=self.actuate(raw);s=self.world(s,action)
                    objective=objective+self.gamma**t*survival*reward(s,action)
                    survival=survival*(s[...,14]<1-1e-6).float()
                es,ea=self.replay.sample(horizon=1,expert_only=True)
                bc=F.mse_loss(self.actor(es[:,0]),ea[:,0])
                trust=trust/horizon
                loss=-objective.mean()/horizon+bc_weight*bc+trust_weight*trust
                self.actor_opt.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),1.);self.actor_opt.step()
            rows.append((float(objective.detach().mean()/horizon),float(bc.detach()),float(trust.detach())))
        self.updates+=1
        return dict(zip(['imagined_return','bc_loss','reference_action_mse'],np.mean(rows,axis=0).tolist()))
    def save(self,path,metadata):
        torch.save(dict(actor=self.actor.state_dict(),world=self.world.state_dict(),critic=self.critic.state_dict(),
                        actor_opt=self.actor_opt.state_dict(),world_opt=self.world_opt.state_dict(),
                        critic_opt=self.critic_opt.state_dict(),updates=self.updates,metadata=metadata,
                        replay_rng_state=self.replay.rng.bit_generator.state,
                        torch_rng_state=torch.get_rng_state(),
                        cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),path)
    def load(self,path):
        x=torch.load(path,map_location=self.device,weights_only=False)
        for name in ['actor','world','critic']:
            getattr(self,name).load_state_dict(x[name])
        self.updates=x['updates']
        return x['metadata']
