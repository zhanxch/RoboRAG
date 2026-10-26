"""Numerical invariants for the paper-based learner, independent of Isaac Sim."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from roborag.supermimic_cloth.core import Actor,Learner,Replay,lambda_returns,tracking

CFG=dict(encoder_widths=[24,16],decoder_widths=[24,16],world_widths=[32,32],critic_widths=[24,16],
    lr=2e-4,actor_betas=[.9,.95],gamma=.99,gae_lambda=.95,world_batches=1,world_batch=8,
    actor_batch=8,microbatch=4,critic_epochs=1,critic_minibatches=2,replay_capacity_steps=1000,
    imagination_horizon=32,cosine_updates=250000,lr_final_fraction=.03)

def episode(n=80,start=0):
    states=np.zeros((n+1,90),dtype='float32');states[:,17]=1;states[:,21]=1
    # Stable identity rotation and nonzero reference error make actor gradients observable.
    states[:,6]=1
    ctx=np.r_[np.zeros(6),np.full(6,.6),np.full(6,-3),np.full(6,3),np.zeros(4)].astype('float32')
    hist=np.zeros((n+1,10,21),dtype='float32');hist[:,:,6]=1
    ref=states.copy();ref[:,0]=.4;ref[:,14]=.2
    return dict(states=states,actions=np.zeros((n,7),dtype='float32'),terminated=np.zeros(n,dtype='float32'),histories=hist,context=ctx,start_frame=start,reference_id='r'),{'r':ref}

class SuperMimicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_lambda_terminal_blocks_bootstrap(self):
        got=lambda_returns(torch.tensor([[1.,2.,3.]]),torch.tensor([[0.,1.,0.]]),torch.tensor([[0.,10.,20.,30.]]),gamma=.9,lam=.8)
        self.assertAlmostEqual(got[0,1].item(),2.)
        self.assertAlmostEqual(got[0,0].item(),1+.9*(.2*10+.8*2),places=5)
    def test_actor_uses_future_and_history_continuously(self):
        torch.manual_seed(1);a=Actor(CFG)
        h=torch.randn(3,10,21,requires_grad=True);g=torch.randn(3,10,83,requires_grad=True)
        y=a(h,g);y.sum().backward()
        self.assertGreater(h.grad.abs().sum().item(),0);self.assertGreater(g.grad.abs().sum().item(),0)
        self.assertTrue(bool((y.abs()<1).all()));self.assertTrue(bool((y[:,6].abs()<.9).all()))
    def test_replay_windows_never_cross_episodes(self):
        e,refs=episode(12);e['states'][:,0]=np.arange(13)
        r=Replay(20,0);r.add(e,'a');r.add(e,'a');self.assertEqual(len(r.episodes),1)
        e2,_=episode(12);e2['states'][:,0]=100+np.arange(13);r.add(e2,'b')
        s,*_=r.sample(200,8,refs,'cpu')
        delta=s[:,1:,0]-s[:,:-1,0]
        self.assertTrue(bool(((delta==1)|(delta==0)).all()))
        self.assertLessEqual(sum(len(x['actions']) for x in r.episodes),20)
    def test_future_window_clamps_within_motion(self):
        e,refs=episode(8,start=72);refs['r']=np.tile(np.arange(81,dtype='float32')[:,None],(1,90))
        r=Replay();r.add(e,'a');*_,goals,remaining,valid=r.sample(1,8,refs,'cpu')
        self.assertGreaterEqual(goals[0,0,0],73);self.assertEqual(goals[0,-1,0],80);self.assertGreaterEqual(remaining.item(),1)
    def test_short_terminal_episode_is_retained_and_padding_masked(self):
        e,refs=episode(1);e['terminated'][0]=1
        r=Replay();r.add(e,'terminal');s,a,d,h,c,g,remain,valid=r.sample(3,8,refs,'cpu')
        self.assertTrue(bool((valid.sum(-1)==1).all()));self.assertTrue(bool((d[:,0]==1).all()))
    def test_tracking_gradient_finite_at_identical_rotation(self):
        e,refs=episode();s=torch.tensor(e['states'][:2],requires_grad=True);a=torch.zeros(2,7,requires_grad=True);c=torch.tensor(e['context']).expand(2,-1)
        r=tracking(s,torch.tensor(refs['r'][:2,:83]),a,s[:,83:],c);r.sum().backward()
        self.assertTrue(bool(torch.isfinite(s.grad).all()))
    def test_joint_update_and_exact_resume(self):
        e,refs=episode();l=Learner(CFG,19,'cpu');l.replay.add(e,'a')
        old={k:v.clone() for k,v in l.actor.state_dict().items()};l.fit_world(refs);stats=l.improve(refs)
        self.assertGreater(stats['actor_grad_norm'],0)
        self.assertTrue(any(not torch.equal(old[k],v) for k,v in l.actor.state_dict().items()))
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'m.pt';l.save(p,dict(protocol_sha256='test'))
            loss=l.fit_world(refs);result=l.improve(refs)
            other=Learner(CFG,99,'cpu');other.load(p,'test');loss2=other.fit_world(refs);result2=other.improve(refs)
            self.assertEqual(loss,loss2);self.assertEqual(result,result2)
            for x,y in zip(l.actor.parameters(),other.actor.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    def test_stability_rejects_stationary_touch_and_slipping(self):
        from roborag.supermimic_cloth.metrics import grasp_metrics
        e,_=episode();s=e['states'];a=e['actions'];a[:,6]=-1;s[:,6]=-1
        self.assertFalse(grasp_metrics(s,a)['success'])
        height=np.r_[np.linspace(0,.4,56),np.repeat(.4,25)]
        s[:,16]=height;s[:,31]=height
        self.assertTrue(grasp_metrics(s,a)['success'])
        s[-10:,29]=np.linspace(0,.2,10)
        self.assertFalse(grasp_metrics(s,a)['success'])
    def test_partial_reference_reset_not_claimed_full_grasp(self):
        from roborag.supermimic_cloth.metrics import grasp_metrics
        e,_=episode(30,start=50);s=e['states'];a=e['actions'];s[:,6]=-1;a[:,6]=-1
        s[:,16]=np.linspace(.1,.4,31);s[:,31]=s[:,16]
        m=grasp_metrics(s,a,start_frame=50)
        self.assertTrue(m['stable_retention']);self.assertFalse(m['success'])
    def test_export_shapes(self):
        l=Learner(CFG,0,'cpu')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'policy.ts';l.export(p);net=torch.jit.load(str(p))
            h=torch.randn(5,10,21);g=torch.randn(5,10,83)
            torch.testing.assert_close(net(h,g),l.actor(h,g))

if __name__=='__main__':unittest.main()
