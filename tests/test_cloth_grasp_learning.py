"""Algorithm invariants, not task-success substitutes. Run with the policy Python."""
import unittest
try:
    import numpy as np
    import torch
    from roborag.cloth_grasp.learning import Actor, Learner, Replay, STATE_DIM, frozen, advance_replay_rng
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional cloth training dependencies are not installed')
class ClothLearningTests(unittest.TestCase):
    def test_legacy_rng_recovery_matches_real_sampler(self):
        a,b=Replay('cpu',seed=3042),Replay('cpu',seed=3042)
        for i in range(19):a.add(np.zeros((81,STATE_DIM)),np.zeros((80,4)),holdout=i in (4,5),expert=i<12)
        for horizon,holdout,expert,count,calls in [(1,False,True,10,7),(8,False,False,17,9),(1,False,False,17,3),(8,True,False,2,2)]:
            for _ in range(calls):a.sample(horizon=horizon,holdout=holdout,expert_only=expert)
            advance_replay_rng(b,count,horizon,calls)
        self.assertEqual(a.rng.bit_generator.state,b.rng.bit_generator.state)

    def test_replay_does_not_cross_episode_or_holdout(self):
        replay=Replay('cpu',seed=2)
        for value,holdout in [(1,False),(2,False),(99,True)]:
            replay.add(np.full((13,STATE_DIM),value),np.zeros((12,4)),holdout=holdout)
        s,_=replay.sample(batch=100,horizon=8)
        self.assertTrue(torch.all(s[:,:,0]==s[:,0:1,0]))
        self.assertFalse(torch.any(s==99))
        h,_=replay.sample(batch=4,horizon=8,holdout=True)
        self.assertTrue(torch.all(h==99))

    def test_replay_rejects_incomplete_or_nonfinite_trajectory(self):
        replay=Replay('cpu')
        with self.assertRaises(ValueError):replay.add(np.zeros((3,STATE_DIM)),np.zeros((3,4)))
        with self.assertRaises(ValueError):replay.add(np.full((4,STATE_DIM),np.nan),np.zeros((3,4)))

    def test_actor_bounded_and_fixed_dimension(self):
        actor=Actor()
        action=actor(torch.randn(40,STATE_DIM)*50)
        self.assertEqual(tuple(action.shape),(40,4))
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue((action.abs()<=1).all())

    def test_binary_gripper_has_hard_forward_and_surrogate_gradient(self):
        learner=Learner('cpu',binary_gripper=True)
        a=torch.tensor([[.1,.2,.3,-.01],[.1,.2,.3,.01]],requires_grad=True)
        actual=learner.actuate(a)
        self.assertEqual(actual[:,3].tolist(),[-1,1])
        actual.sum().backward()
        self.assertTrue(torch.equal(a.grad,torch.ones_like(a)))

    def test_frozen_model_keeps_action_gradient(self):
        learner=Learner('cpu',seed=3)
        # A fitted nonzero world-model Jacobian must carry gradients to actions.
        torch.nn.init.normal_(learner.world.net[-1].weight,std=.01)
        s=torch.randn(8,STATE_DIM)
        before=[p.detach().clone() for p in learner.world.parameters()]
        with frozen(learner.world):
            a=learner.actor(s)
            pred=learner.world(s,a)
            loss=pred[:,:4].square().mean()
            loss.backward()
            self.assertTrue(all(p.grad is None for p in learner.world.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in learner.actor.parameters()),0)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,learner.world.parameters())))

    def test_training_update_has_finite_losses_and_changes_actor(self):
        learner=Learner('cpu',seed=4)
        rng=np.random.default_rng(4)
        for holdout in [False,True]:
            s=rng.normal(0,.1,(25,STATE_DIM)).astype(np.float32)
            s[:,14]=np.arange(25)/80
            learner.replay.add(s,np.zeros((24,4)),holdout=holdout,expert=True)
        learner.fit_world(steps=2,horizon=3)
        before=[p.detach().clone() for p in learner.actor.parameters()]
        stats=learner.improve(steps=2,horizon=3)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,learner.actor.parameters())))
        self.assertTrue(np.isfinite(learner.validation_rmse(horizon=3)))


if __name__=='__main__':unittest.main()
