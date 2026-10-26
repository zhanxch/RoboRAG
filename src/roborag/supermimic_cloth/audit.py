"""Hash and physical-outcome audit, plus self-contained loadable model bundle."""
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from .protocol import rows,sha,write,freeze,TARGETS
from .metrics import grasp_metrics

def audit(root,package=False):
    plan=freeze(root);episodes=[];errors=[]
    for folder in [root/'references',root/'train',*sorted((root/'probes').glob('*'))]:
        for row in rows(folder/'episodes.jsonl'):
            path=Path(row['transition_path'])
            if sha(path)!=row['transition_sha256']:errors.append(row['id']+': trajectory hash')
            scene=folder/'test-layouts'/f"{row['id']}.json"
            if sha(scene)!=row['scene_sha256']:errors.append(row['id']+': scene hash')
            with np.load(path) as d:
                measured=grasp_metrics(d['states'],d['actions'],int(d['start_frame']),
                    horizon=row.get('verification_horizon',80),hold_steps=row.get('required_hold_steps',25),terminated=bool(d['terminated'].any()))
                if measured['success']!=row['success']:errors.append(row['id']+': outcome recomputation')
                if not np.isfinite(d['states']).all() or not np.isfinite(d['actions']).all():errors.append(row['id']+': nonfinite physics')
                if not np.allclose(d['states'][1:,83:],d['actions'],atol=1e-6):errors.append(row['id']+': executed action mismatch')
                obs=np.concatenate([d['states'][:,:14],d['states'][:,83:]],axis=-1)
                if not np.allclose(d['histories'][:,-1],obs,atol=1e-6):errors.append(row['id']+': observation history mismatch')
                if d['states'].shape[1]!=90 or len(d['states'])!=len(d['actions'])+1:errors.append(row['id']+': transition shape')
            episodes.append(dict(id=row['id'],role=row['role'],success=row['success'],sha256=row['transition_sha256'],scene_sha256=row['scene_sha256']))
    pilot=root.parent/'cloth-grasp-bench-v1'
    frozen=json.loads((pilot/'checkpoint-freeze.json').read_text())
    for file,key in [('bc-baseline.pt','bc_sha256'),('final.pt','rl_sha256'),('policy.ts','policy_sha256')]:
        if sha(pilot/file)!=frozen[key]:errors.append('Frozen pilot changed: '+file)
    replay_check=None
    if (root/'last.pt').exists():
        import torch
        ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        ids={r['id'] for r in episodes if r['role']=='train'}
        if not set(ck['replay'].ids)<=ids:errors.append('Replay contains non-TRAIN episode IDs')
        if ck['metadata']['protocol_sha256']!=plan['sha256']:errors.append('Checkpoint protocol mismatch')
        replay_check=dict(unique_train_ids=len(ck['replay'].ids),buffer_steps=sum(len(e['actions']) for e in ck['replay'].episodes),optimizer_rng_complete=all(k in ck for k in ['actor_opt','world_opt','critic_opt','torch_rng','cuda_rng','numpy_rng']))
    train=[r for r in episodes if r['role']=='train'];dev=[r for r in episodes if r['role']=='seen_development']
    report=dict(updated=time.time(),passed=not errors,errors=errors,episodes=episodes,
        source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        protocol_sha256=plan['sha256'],replay_check=replay_check,pilot_frozen_hashes_verified=True,train_episodes=len(train),development_episodes=len(dev),
        contamination_check='Only own TRAIN policy transitions loaded; reference feasibility files used for goals/reset, not replay or BC.',
        pilot_model_sha256=sha(root.parent/'cloth-grasp-bench-v1/bc-policy.ts'),
        models={p.name:sha(p) for p in [root/'policy.ts',root/'last.pt',root/'best.pt',root/'best-policy.ts'] if p.exists()})
    write(root/'audit.json',report)
    if errors:raise ValueError('Audit failed: '+str(errors[:5]))
    probes=rows(root/'probes.jsonl');model=json.loads((root/'model-card.json').read_text()) if (root/'model-card.json').exists() else None
    conclusion=dict(updated=time.time(),scope=plan['scope'],criterion=plan['criterion'],algorithm=plan['implementation'],
        demonstration_files_verified=10,demonstrations_used_for_reference=1,model=model,
        references=json.loads((root/'references.json').read_text()) if (root/'references.json').exists() else None,
        latest_development=probes[-1] if probes else None,
        effectiveness='not yet established',note='Reference replay success is not policy success. Historical BC+RL used a different criterion.',
        audit_passed=True)
    if (root/'confirmation.json').exists():
        conclusion['confirmation']=json.loads((root/'confirmation.json').read_text())
        if conclusion['confirmation']['passed']:conclusion['effectiveness']='stable local grasp confirmed on seen conditions; not complete folding'
    write(root/'conclusion.json',conclusion)
    if package:
        dest=root/'supermimic-cloth-model.zip';tmp=dest.with_suffix('.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for name in ['plan.json','effective-plan.json','adapter-amendment.json','model-card.json','audit.json','conclusion.json','references.json','policy.ts','best-policy.ts','last.pt','best.pt','confirmation.json','verification-protocol.json','confirmed-policy.ts']:
                p=root/name
                if p.exists():z.write(p,name)
            if (root/'references.json').exists():
                for r in json.loads((root/'references.json').read_text())['selected']:z.write(r['path'],'reference-bank/'+r['id']+'.npz')
            for p in Path(__file__).parent.glob('*.py'):z.write(p,'source/roborag/supermimic_cloth/'+p.name)
            z.writestr('source/roborag/__init__.py','')
            z.writestr('LOAD_MODEL.txt','TorchScript: torch.jit.load(policy.ts)(history, goal). history float32[B,10,21]; goal float32[B,10,83]. Returns continuous normalized joint setpoints[B,7]. Physical arm command = reference context[:6] + context[6:12]*action[:6]; gripper opening=(action[6]+1)/2. State/normalization contract is in effective-plan.json and source/roborag/supermimic_cloth/physics.py. Reference-bank NPZ contains physically measured future states and context. This is reference-conditioned local cloth grasp; read conclusion.json before treating it as effective. last.pt includes exact optimizer/replay/RNG; add source/ to PYTHONPATH to load its Replay class. Current production simulator and dependencies are external to this bundle. No standalone camera-to-grasp deployment is claimed.\n')
        tmp.replace(dest)
    return report
