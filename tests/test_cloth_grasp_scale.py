import json
from pathlib import Path
import tempfile
import unittest
try:
    import numpy as np
    import torch
    from roborag.cloth_grasp.scale import fit_offline
    from roborag.cloth_grasp.protocol import digest
except ImportError:
    np=None


@unittest.skipIf(np is None,'Optional policy dependencies are not installed')
class ExpandedTrainingTests(unittest.TestCase):
    def fixture(self,root):
        physics=root/'physics';physics.mkdir()
        rows=[dict(id='a',role='train'),dict(id='b',role='train'),dict(id='h',role='world_holdout')]
        (physics/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        rng=np.random.default_rng(11)
        for row in rows:
            folder=physics/'rollouts'/row['id'];folder.mkdir(parents=True)
            states=rng.normal(0,.01,(81,39)).astype('float32');states[:,14]=np.arange(81)/80
            actions=rng.normal(0,.01,(80,4)).astype('float32');actions[:,3]=-1
            np.savez(folder/'transitions.npz',states=states,actions=actions)
        p=dict(collection=[dict(id='a',source_index=0),dict(id='b',source_index=1),dict(id='h',source_index=0)],
               learning=dict(bc_steps=2,world_steps=2,policy_updates=1,bc_weight=20,rl_learning_rate=2e-5))
        p['sha256']=digest(p);(root/'development-protocol.json').write_text(json.dumps(p))
    def test_data_subset_holdout_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);fit_offline(root,1,42,device='cpu')
            out=root/'models/demos-1-seed-42';m=json.loads((out/'experiment.json').read_text())
            self.assertEqual(m['physics_episode_ids'],['a','h'])
            self.assertEqual(m['train_transitions'],80)
            self.assertEqual(m['world_holdout_transitions'],80)
            result=torch.jit.load(str(out/'policy.ts'))(torch.zeros(3,39))
            self.assertEqual(tuple(result.shape),(3,4));self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(torch.all(result[:,3].abs()==1))
    def test_rejects_evaluation_trajectory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            with (root/'physics/episodes.jsonl').open('a') as f:f.write('\n'+json.dumps(dict(id='test-x',role='test')))
            with self.assertRaisesRegex(ValueError,'Only TRAIN'):fit_offline(root,1,42,device='cpu')


if __name__=='__main__':unittest.main()
