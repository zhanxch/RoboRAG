#!/usr/bin/env python3
"""Audit a completed experiment and export data, scientific plot and model bundle."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    out=args.run.resolve()
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from roborag.cloth_grasp.protocol import read,summarize,TARGETS
    from roborag.cloth_grasp.inference import ClothPolicy
    protocol=read(out/'benchmark.json')
    rows=[json.loads(x) for x in (out/'episodes.jsonl').read_text().splitlines()]
    result=summarize(protocol,rows)
    saved=json.loads((out/'benchmark-results.json').read_text())
    assert result==saved
    freeze=json.loads((out/'checkpoint-freeze.json').read_text())
    for key,name in [('bc','bc-baseline.pt'),('rl','final.pt'),('policy','policy.ts')]:
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==freeze[key+'_sha256']
    train=[r for r in rows if r['role'] in ('train','world_holdout')]
    test=[r for r in rows if r['role']=='test']
    assert len(train)==20 and len(test)==48
    assert {r['seed'] for r in train}.isdisjoint(r['seed'] for r in test)
    scenes=[json.loads(x) for x in (out/'test-scene-audit.jsonl').read_text().splitlines()]
    scene_keys=[(x['policy'],x['case_id']) for x in scenes]
    assert len(scene_keys)==48 and len(set(scene_keys))==48
    cases={c['id']:c for c in protocol['cases']}
    assert all(x['asset_id']==cases[x['case_id']]['garment'] and x['vertices']>0 for x in scenes)
    policy=ClothPolicy(out/'policy.ts')
    checkpoint=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    from roborag.cloth_grasp.learning import Learner
    learner=Learner('cpu',binary_gripper=True)
    learner.load(out/'final.pt')
    diffs=[]
    failure_modes={'bc':{},'rl':{}}
    for row in rows:
        path=out/'rollouts'/row['id']/'transitions.npz'
        if row.get('reset_failed'):continue
        data=np.load(path);s,a=data['states'],data['actions']
        assert s.shape==(81,39) and a.shape==(80,4)
        assert np.isfinite(s).all() and np.isfinite(a).all()
        assert np.max(np.abs(a))<1.001
        held=(s[1:,3]>=.8)&(np.linalg.norm(s[1:,:3],axis=1)<.55)&(s[1:,4]<.35)
        hold=0
        for value in held:hold=hold+1 if value else 0
        assert hold==row['final_hold_steps'] and (hold>=10)==row['success']
        if row['role']=='test' and not row['success']:
            category=('lifted_then_not_retained' if np.max(s[1:,3])>=.8 else 'never_reached_lift_threshold')
            group=failure_modes[row['policy']]
            group[category]=group.get(category,0)+1
        if row['role']=='test' and row['policy']=='rl':
            predicted=policy(s[:-1]);reference=learner.action(s[:-1])
            diffs.append(float(np.max(np.abs(predicted-reference))))
            np.testing.assert_allclose(predicted,reference,atol=2e-6,rtol=2e-6)
    sample=np.load(out/'rollouts'/'train-001'/'transitions.npz')['states'][0]
    torch.set_num_threads(1)
    for _ in range(100):policy(sample)
    start=time.perf_counter()
    latency_samples=1000
    for _ in range(latency_samples):policy(sample)
    latency_ms=(time.perf_counter()-start)*1000/latency_samples
    audit=dict(protocol_hash_verified=True,checkpoints_hash_verified=True,complete_pairs=24,
               disjoint_training_and_test_seeds=True,outcomes_recomputed_from_physics=True,
               torchscript_max_absolute_error=max(diffs,default=0),
               loaded_scene_assets_verified=True,failure_modes=failure_modes,
               cpu_single_observation_mean_ms=latency_ms,latency_samples=latency_samples,
               latency_scope='TorchScript policy with input checks; excludes state estimation, IK and robot execution',
               total_local_control_steps=sum(r['steps'] for r in rows),
               excluded_costs='reset settling, preposition, rendering and development runs')
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    fields=['case_id','policy','target','garment','distribution','position','success','max_lift_m','final_hold_steps','reason']
    with (out/'benchmark.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(test)
    config=json.loads((out/'experiment.json').read_text())
    assert protocol['frozen_at']<config['started_at']<config['training_completed_at']<=freeze['time']
    card=dict(name='Local cloth-grasp policy',parameters=config['actor_parameters'],
              torchscript_input_shape=['batch',39],torchscript_output_shape=['batch',4],
              wrapper_input_shapes=[[39],['batch',39]],wrapper_output_shapes=[[4],['batch',4]],
              input_schema={'0:3':'(target-TCP)/0.1m','3':'target height above initial /0.1m','4':'previous commanded gripper opening [0,1]',
                            '5:8':'(TCP-initial_target)/0.1m','8:11':'TCP displacement/0.008m','11:14':'target displacement/0.008m',
                            '14':'step/80','15:39':'8 persistent material points relative to target /0.1m, flattened xyz'},
              output_schema={'0:3':'xyz increment normalized by 8mm per axis','3':'binary gripper: -1 closed, +1 open'},
              domain='PhysX local cloth pickup with an already nearby gripper; privileged material geometry',
              fixed_orientation='Use the target-side quaternion in demo.json; not predicted by actor',
              tcp='link6 + rotated [0.145,0,0] m',frequency_hz=25,horizon=80,
              workspace_m={'lower':[-.07,-.07,-.005],'upper':[.07,.07,.19],'origin':'initial target point'},
              training=config,benchmark=result,audit=audit,
              limitations=['Single seed and one source demonstration','Garment 04 is held out from simulator-training interactions; the original HDF5 does not identify its garment asset','No RGB perception or target selection model','No full folding or real-robot validation','Binary gripper uses a straight-through training estimator'])
    (out/'model-card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2))
    baseline=Learner('cpu',binary_gripper=True);baseline.load(out/'bc-baseline.pt')
    class BinaryActor(torch.nn.Module):
        def __init__(self,actor):super().__init__();self.actor=actor
        def forward(self,state):
            action=self.actor(state)
            return torch.cat([action[...,:3],torch.where(action[...,3:4]>=0,1.,-1.)],-1)
    torch.jit.trace(BinaryActor(baseline.actor.eval()),torch.zeros(1,39)).save(str(out/'bc-policy.ts'))
    np.testing.assert_allclose(ClothPolicy(out/'bc-policy.ts')(sample),baseline.action(sample),atol=2e-6,rtol=2e-6)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    colors={'bc':'#647b8c','rl':'#548a37'}
    axes[0].set_title('Frozen benchmark: 24 paired cases')
    for i,name in enumerate(['bc','rl']):
        model=result['models'][name];lo,hi=model['wilson95'];v=model['rate']
        axes[0].bar(i,100*v,color=colors[name],width=.6)
        axes[0].errorbar(i,100*v,yerr=[[100*(v-lo)],[100*(hi-v)]],fmt='none',color='black',capsize=4)
        axes[0].text(i,100*hi+3,f"{model['successes']}/24",ha='center')
    axes[0].set_xticks([0,1],['BC','BC + RL']);axes[0].set_ylim(0,110);axes[0].set_ylabel('Success (%)')
    x=np.arange(4)
    for i,name in enumerate(['bc','rl']):
        values=result['models'][name]['groups']['target']
        axes[1].bar(x+(i-.5)*.34,[100*values[t]['successes']/values[t]['total'] for t in TARGETS],width=.34,color=colors[name],label=name.upper())
    axes[1].set_xticks(x,['L sleeve','R sleeve','L hem','R hem']);axes[1].set_ylim(0,105);axes[1].set_title('Per target (6 cases each)');axes[1].legend()
    for i,name in enumerate(['bc','rl']):
        g=result['models'][name]['groups']['distribution']
        axes[2].bar(np.arange(2)+(i-.5)*.34,[100*g[t]['successes']/g[t]['total'] for t in ['seen_garment','unseen_garment']],width=.34,color=colors[name],label=name.upper())
    axes[2].set_xticks([0,1],['Training garment 09','Sim-held-out garment 04']);axes[2].set_ylim(0,105);axes[2].set_title('Garment transfer (12 cases each)')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Local cloth pickup | lift >= 8 cm, retained during final 0.4 s',fontsize=12)
    fig.savefig(out/'benchmark-summary.png',dpi=180);fig.savefig(out/'benchmark-summary.pdf');plt.close(fig)
    files=['policy.ts','policy.pt','bc-policy.ts','bc-baseline.pt','final.pt','model-card.json','benchmark.json','benchmark-results.json','benchmark.csv','benchmark-summary.png','benchmark-summary.pdf','audit.json','demo.json','experiment.json','metrics.jsonl','checkpoint-freeze.json','test-scene-audit.jsonl','episodes.jsonl']
    with zipfile.ZipFile(out/'cloth-grasp-model-and-results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(out/name,name)
        if (out/'coverage.json').exists():z.write(out/'coverage.json','coverage.json')
        z.write(ROOT/'src/roborag/cloth_grasp/inference.py','inference.py')
        z.write(ROOT/'docs/cloth-grasp.md','README.md')
        for path in (out/'source').glob('*.py'):z.write(path,'training-source/'+path.name)
        for path in (out/'recovery-source').glob('*.py'):z.write(path,'recovery-source/'+path.name)
        for path in (out/'evaluation-source').glob('*.py'):z.write(path,'evaluation-source/'+path.name)
    print(json.dumps(dict(results=result,audit=audit,bundle_bytes=(out/'cloth-grasp-model-and-results.zip').stat().st_size),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
