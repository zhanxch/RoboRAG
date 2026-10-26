import json
from pathlib import Path
import tempfile
import unittest
try:
    import numpy as np
    import torch
    from roborag.cloth_grasp.focus import cases,score,train_update,write,sha,restore,amendment_for_round
    from roborag.cloth_grasp.protocol import TARGETS,digest
    from roborag.cloth_grasp.learning import Learner
except ImportError:
    np=None


@unittest.skipIf(np is None,'Optional policy dependencies are not installed')
class FocusTrainingTests(unittest.TestCase):
    def fixture(self,root):
        pilot=root/'pilot';pilot.mkdir()
        l=Learner('cpu',seed=42,binary_gripper=True);l.save(pilot/'bc-baseline.pt',{})
        torch.jit.trace(l.actor,torch.zeros(1,39)).save(str(pilot/'bc-policy.ts'))
        write(pilot/'checkpoint-freeze.json',dict(bc_sha256=sha(pilot/'bc-baseline.pt')))
        p=dict(pilot_train=str(pilot),seed=42,garment=9,positions=[[0,0,.065],[.015,-.015,.065]],fixed_garment_position=[0,-.12,.8441],
            learning=dict(world_initial_steps=1,world_steps_per_round=1,policy_updates_per_round=2,
                world_window=3,imagination_horizon=3,bc_weight=20,rl_learning_rate=2e-5),
            effectiveness_gate=dict(probe_trials=16,confirmation_trials=32,success_rate=.9,per_target_minimum_rate=.75))
        p['sha256']=digest(p);write(root/'training-protocol.json',p)
        rng=np.random.default_rng(3)
        def episode(base,identity):
            folder=base/'rollouts'/identity;folder.mkdir(parents=True)
            s=rng.normal(0,.05,(81,39)).astype('float32');s[:,14]=np.arange(81)/80
            a=rng.normal(0,.05,(80,4)).astype('float32');a[:,3]=-1
            np.savez(folder/'transitions.npz',states=s,actions=a)
        pilot_rows=[dict(id='t',role='train',controller='train-demo-retarget'),dict(id='h',role='world_holdout',controller='train-demo-retarget'),dict(id='NO_TEST_FILE',role='test')]
        for identity in ['t','h']:episode(pilot,identity)
        (pilot/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in pilot_rows))
        train=root/'train';train.mkdir();r=[]
        for i in range(8):
            identity=f'train-0001-{i:02d}';episode(train,identity)
            r.append(dict(id=identity,role='train',controller='train-demo-retarget' if i<4 else 'learned-policy'))
        (train/'episodes.jsonl').write_text('\n'.join(json.dumps(x) for x in r))
        return p
    def test_real_update_resume_and_test_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            before=train_update(root,0,device='cpu');after=train_update(root,1,device='cpu')
            self.assertEqual(before['train_transitions'],80)
            self.assertEqual(after['train_transitions'],9*80)
            self.assertEqual(after['holdout_transitions'],80)
            self.assertEqual(after['total_rl_gradient_updates'],2)
            self.assertFalse(any('NO_TEST_FILE' in x for x in after['replay_episode_ids']))
            checkpoint=root/'checkpoints/round-0001.pt';old=sha(checkpoint)
            train_update(root,1,device='cpu');self.assertEqual(old,sha(checkpoint))
            l=Learner('cpu',binary_gripper=True);restore(l,checkpoint)
            self.assertTrue(l.actor_opt.state)
            exported=torch.jit.load(str(root/'policy.ts'))(torch.zeros(2,39)).detach().numpy()
            np.testing.assert_allclose(exported,l.action(np.zeros((2,39))),atol=2e-6)
    def test_seen_cases_and_strict_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.fixture(Path(d))
            train=cases(p,'collect',1);test=cases(p,'probe',5);confirm=cases(p,'confirm',10)
            self.assertTrue({tuple(c['tcp_offset_m']) for c in test}<={tuple(c['tcp_offset_m']) for c in train})
            self.assertTrue({c['seed'] for c in train}.isdisjoint(c['seed'] for c in test))
            self.assertTrue({c['seed'] for c in test}.isdisjoint(c['seed'] for c in confirm))
            values=[dict(id=c['id'],target=c['subtask'],success=True) for c in test]
            values[0]['success']=False
            self.assertTrue(score(values,16,p['effectiveness_gate'])['passed'])
            values[1]['success']=False
            self.assertFalse(score(values,16,p['effectiveness_gate'])['passed'])
            with self.assertRaises(ValueError):score(values[:-1],16,p['effectiveness_gate'])
    def test_versioned_reset_conservative_update_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=self.fixture(root);train_update(root,0,device='cpu')
            (root/'amendments').mkdir()
            amendment=dict(id='test-v2',effective_round=1,base_protocol_sha256=plan['sha256'],
                reference_checkpoint='bc-baseline.pt',reference_sha256=sha(root/'bc-baseline.pt'),reset_actor_to_reference=True,
                learning=dict(update_method='conservative',trust_weight=100.,imagination_horizon=3))
            amendment['sha256']=digest(amendment);write(root/'amendments/v2.json',amendment)
            self.assertIsNone(amendment_for_round(root,plan,0))
            before=torch.load(root/'checkpoints/round-0000.pt',weights_only=False)
            metadata=train_update(root,1,device='cpu')
            self.assertEqual(metadata['learning_revision'],'test-v2')
            self.assertEqual(metadata['rl_updates_since_actor_reset'],2)
            self.assertTrue(np.isfinite(metadata['metrics']['reference_action_mse']))
            checkpoint=root/'checkpoints/round-0001.pt'
            after=torch.load(checkpoint,weights_only=False)
            self.assertTrue(all(torch.equal(v,after['critic'][k]) for k,v in before['critic'].items()))
            l=Learner('cpu',binary_gripper=True);restore(l,checkpoint)
            rng_before=torch.rand(8);replay_before=l.replay.rng.random(8)
            torch.manual_seed(999);l.replay.rng.random(100)
            restore(l,checkpoint)
            self.assertTrue(torch.equal(rng_before,torch.rand(8)))
            np.testing.assert_array_equal(replay_before,l.replay.rng.random(8))
            saved=sha(checkpoint);train_update(root,1,device='cpu');self.assertEqual(saved,sha(checkpoint))
            amendment['learning']['trust_weight']=1.
            write(root/'amendments/v2.json',amendment)
            with self.assertRaises(ValueError):amendment_for_round(root,plan,1)


if __name__=='__main__':unittest.main()
