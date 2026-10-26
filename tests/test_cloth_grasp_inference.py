import unittest
import tempfile
from pathlib import Path
try:
    import numpy as np
    import torch
    from roborag.cloth_grasp.inference import encode_state, ClothPolicy
except ImportError:
    np=None


@unittest.skipIf(np is None,'Optional policy dependencies are not installed')
class ClothInferenceTests(unittest.TestCase):
    def sample(self):
        return dict(tcp=[0,0,1.1],target=[0,0,1.08],initial_target=[0,0,1],
                    previous_tcp=[0,0,1.092],previous_target=[0,0,1.072],
                    gripper_command=0,step=40,patch=np.tile([0,0,1.08],(8,1)))
    def test_units_velocity_and_clock(self):
        state=encode_state(**self.sample())
        self.assertEqual(state.shape,(39,))
        np.testing.assert_allclose(state[[2,3,7,10,13,14]],[-.2,.8,1,1,1,.5],atol=1e-5)
        np.testing.assert_array_equal(state[15:],0)
    def test_rejects_unusable_observations(self):
        for key,value in [('step',80),('gripper_command',2),('target',[0,0,np.nan]),('patch',np.zeros((7,3)))]:
            args=self.sample();args[key]=value
            with self.assertRaises(ValueError):encode_state(**args)
    def test_wrapper_accepts_single_state_for_batched_torchscript(self):
        class Batched(torch.nn.Module):
            def forward(self,x):return x[:, :4]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.ts'
            torch.jit.trace(Batched(),torch.zeros(1,39)).save(str(path))
            policy=ClothPolicy(path)
            state=encode_state(**self.sample())
            single=policy(state);batch=policy(np.stack([state,state]))
            self.assertEqual(single.shape,(4,))
            self.assertEqual(batch.shape,(2,4))
            np.testing.assert_array_equal(single,batch[0])


if __name__=='__main__':unittest.main()
