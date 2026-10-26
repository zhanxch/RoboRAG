"""Export recorded local geometry for the lightweight browser viewer."""
import argparse
import json
from pathlib import Path
import numpy as np


def export_run(run):
    for path in Path(run).glob('rollouts/*/transitions.npz'):
        destination=path.with_name('geometry.json')
        if destination.exists():continue
        with np.load(path) as data:
            s=data['states']
            tcp=s[:,5:8]*.1
            target=tcp+s[:,:3]*.1
            patch=s[:,15:].reshape(-1,8,3)*.1+target[:,None,:]
            value=dict(frame='world axes, origin at the initial target point',units='m',
                       tcp=tcp.tolist(),target=target.tolist(),patch=patch.tolist(),gripper_command=s[:,4].tolist())
        destination.write_text(json.dumps(value,separators=(',',':')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();export_run(a.run)
