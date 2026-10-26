"""Offline learner process, sequenced after each owned physical collection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .protocol import freeze,rows,write,append,sha,verification
from .core import Learner,tracking

def main(root,round_number):
    torch.set_num_threads(4)
    plan=freeze(root);cfg=plan['method']; learner=Learner(cfg,plan['seed'])
    selected=json.loads((root/'references.json').read_text())
    if not selected['gate_passed']:raise RuntimeError('Physical reference gate not passed')
    refs={}
    for r in selected['selected']:
        if sha(r['path'])!=r['sha256']:raise ValueError('Reference hash changed')
        refs[r['id']]=np.load(r['path'])['states']
    metadata=dict(verification_protocol_sha256=verification(root)['sha256'],protocol_sha256=plan['sha256'],algorithm='SuperMimic paper-based cloth reproduction',
        source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},trained=False,
        actor_parameters=sum(p.numel() for p in learner.actor.parameters()),reference_hashes={r['id']:r['sha256'] for r in selected['selected']})
    if (root/'last.pt').exists():
        current_sources=metadata['source_hashes'];metadata=learner.load(root/'last.pt',plan['sha256'])
        if metadata['source_hashes']!=current_sources:
            metadata.setdefault('source_revisions',[]).append(dict(at_update=learner.updates,previous=metadata['source_hashes'],current=current_sources,time=time.time()))
            metadata['source_hashes']=current_sources
    else:
        learner.save(root/'last.pt',metadata);learner.export(root/'policy.ts');write(root/'model-card.json',metadata)
    if round_number==0:return
    train=rows(root/'train/episodes.jsonl'); totalsteps=0; resets={}
    for r in train:
        if r['role']!='train':raise ValueError('Non-TRAIN data in collection manifest')
        if r['protocol_sha256']!=plan['sha256'] or sha(r['transition_path'])!=r['transition_sha256']:raise ValueError('Training trajectory hash mismatch')
        totalsteps+=r['steps']
        with np.load(r['transition_path']) as x:e={k:x[k] for k in x.files}
        e['reference_id']=str(e['reference_id']);e['start_frame']=int(e['start_frame'])
        learner.replay.add(e,r['id'])
        # Physical reference tracking return for return-weighted resets, never imagined return.
        n=len(e['actions']); phase=e['start_frame'];ref=refs[e['reference_id']]
        with torch.no_grad():
            val=tracking(torch.tensor(e['states'][1:]),torch.tensor(ref[np.minimum(np.arange(phase+1,phase+n+1),len(ref)-1),:83]),torch.tensor(e['actions']),torch.tensor(e['states'][:-1,83:]),torch.tensor(e['context']).expand(n,-1)).mean().item()
        resets.setdefault(e['reference_id'],{}).setdefault(phase//25,[]).append(val)
    oldweights=json.loads((root/'reset-weights.json').read_text()) if (root/'reset-weights.json').exists() else {}
    newweights={}
    for ref in refs:
        returns=np.array([np.mean(resets.get(ref,{}).get(i,[0.])) for i in range(3)])
        logits=-25*returns/max(float(returns.max()),1e-6);logits-=logits.max();soft=np.exp(logits);soft/=soft.sum()
        hist=np.array(oldweights.get(ref,{}).get('history',[1/3]*3));p=.5/3+.4*soft+.1*hist
        newweights[ref]=dict(probabilities=p.tolist(),history=(.9*hist+.1*soft).tolist(),physical_returns=returns.tolist())
    write(root/'reset-weights.json',newweights)
    target=int(totalsteps*cfg['updates_per_real_step'])
    metadata.update(trained=True,round=round_number,physical_steps=totalsteps,training_ids=list(learner.replay.ids))
    while learner.updates<target:
        if time.time()>=plan['deadline']:break
        started=time.time();learner.schedule()
        # Atomic checkpoint after every completed joint update preserves exact RNG/replay/optimizers.
        world=learner.fit_world(refs);stats=learner.improve(refs)
        stats.update(update=learner.updates,round=round_number,world_train_loss=world,physical_steps=totalsteps,
            elapsed_s=time.time()-started,peak_cuda_gb=torch.cuda.max_memory_allocated()/1e9,updated=time.time())
        learner.save(root/'last.pt',metadata);learner.export(root/'policy.ts')
        write(root/'model-card.json',dict(metadata,updates=learner.updates,policy_sha256=sha(root/'policy.ts')))
        append(root/'metrics.jsonl',stats);write(root/'progress.json',stats)
        print(json.dumps(stats),flush=True)
    write(root/'model-card.json',dict(metadata,updates=learner.updates,last_sha256=sha(root/'last.pt'),policy_sha256=sha(root/'policy.ts')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--round',type=int,required=True);a=p.parse_args();main(a.run,a.round)
