#!/usr/bin/env python3
"""Pin and verify public TRAIN sources for the authorized 72-hour experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO='RoboDojo-Benchmark/RoboDojo'
REVISION='269457a3b42d8d0ae69898a601f2c1487c2f826e'
TASKS=['fold_clothes','pack_objects_into_box','stack_blocks','classify_objects','fill_pen_holder','hang_mugs']


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def initialize(out):
    planpath=out/'plan.json'
    if planpath.exists():return json.loads(planpath.read_text())
    now=time.time()
    value=dict(schema='grasp.scale.v1',started_at=now,deadline=now+72*3600,
        authorized_hours=72,compute='Local RTX 4090; one owned simulator at a time',
        repository=REPO,revision=REVISION,tasks=TASKS,
        source_allocation='Public demonstration HDF5 files allocated to TRAIN; official evaluation layouts and all pilot test trajectories are excluded from fitting.',
        initial_download_counts={t:10 if t=='fold_clothes' else 1 for t in TASKS},
        later_data_curve=[1,10,50],training_seeds=[31001,31002,31003],
        initial_focus='Multi-demonstration cloth BC versus BC+RL development gate; other task data are preparation only.',
        dev_gate=dict(minimum_rl_rate=.70,minimum_rl_minus_bc=.10,all_training_seeds_nonnegative=True,
            interpretation='Engineering expansion gate on development data, not statistical proof. Report every seed and unsuccessful candidate.'),
        reserved_sim_test_garments=[1,6,11],development_garments=[9,7],
        train_garments=[9],pilot_test_garment_excluded=4,
        final_test='Before any new official claim, freeze fresh object/layout/offset cases and evaluate every seed; never select a checkpoint on test.',
        limits=dict(download_bytes=64*1024**3,minimum_free_disk_bytes=100*1024**3,worker_timeout_seconds=1800),
        stages=['verify TRAIN files','collect real-physics retargeted trajectories','three-seed BC/RL development comparison','expand only after effectiveness gate; implement and validate task adapters separately'],
        status='authorized')
    write(planpath,value)
    return value


def prepare(out):
    import h5py
    import numpy as np
    plan=initialize(out)
    write(out/'data-status.json',dict(status='resolving',updated=time.time()))
    manifest_path=out/'data-manifest.json'
    if manifest_path.exists():manifest=json.loads(manifest_path.read_text())
    else:
        files=[]
        for task,count in plan['initial_download_counts'].items():
            url=f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data/RoboDojo/{task}/arx_x5/data?limit=100'
            with urllib.request.urlopen(url,timeout=30) as response:listing=json.load(response)
            candidates=sorted((x for x in listing if x['path'].endswith('.hdf5')),key=lambda x:x['path'])[:count]
            if len(candidates)!=count:raise RuntimeError(f'Insufficient public TRAIN sources: {task}')
            for x in candidates:
                files.append(dict(task=task,repository_path=x['path'],bytes=x['size'],sha256=x['lfs']['oid'],
                    local_path=str(out/'data'/task/'train'/Path(x['path']).name),status='pending'))
        if sum(x['bytes'] for x in files)>plan['limits']['download_bytes']:raise RuntimeError('Download budget exceeded')
        manifest=dict(repository=REPO,revision=REVISION,frozen_at=time.time(),files=files)
        write(manifest_path,manifest)
    # Reuse the byte-identical existing source without copying 626 MB.
    pilot=json.loads((ROOT/'runs/cloth-grasp-bench-v1/demo.json').read_text())
    for row in manifest['files']:
        if time.time()>=plan['deadline']:raise TimeoutError('72-hour deadline reached')
        path=Path(row['local_path']);path.parent.mkdir(parents=True,exist_ok=True)
        if row['sha256']==pilot['sha256'] and not path.exists():path.symlink_to(pilot['source'])
        if not path.exists():
            if shutil.disk_usage(out).free<row['bytes']+plan['limits']['minimum_free_disk_bytes']:raise RuntimeError('Free disk guard')
            partial=path.with_suffix('.partial')
            url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row["repository_path"]}'
            write(out/'data-status.json',dict(status='downloading',task=row['task'],file=path.name,
                 completed=sum(x['status']=='verified' for x in manifest['files']),total=len(manifest['files']),updated=time.time()))
            for attempt in range(5):
                try:
                    offset=partial.stat().st_size if partial.exists() else 0
                    req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
                    with urllib.request.urlopen(req,timeout=45) as response:
                        resume=offset and response.status==206
                        if resume and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Invalid range response')
                        with partial.open('ab' if resume else 'wb') as stream:
                            while True:
                                if time.time()>=plan['deadline']:raise TimeoutError('72-hour deadline reached')
                                chunk=response.read(4*1024*1024)
                                if not chunk:break
                                stream.write(chunk)
                    if partial.stat().st_size!=row['bytes'] or sha(partial)!=row['sha256']:raise ValueError('Downloaded content hash mismatch; preserved partial')
                    partial.replace(path);break
                except Exception:
                    if attempt==4:raise
                    time.sleep(min(2**attempt,16))
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError(f'Existing source hash mismatch: {path}')
        with h5py.File(path,'r') as f:
            frames=len(f['state/left_ee_poses']);events=[];closure_counts=[]
            arrays={f'action_{k}':f[f'action/{k}'][:] for k in f['action']}
            for side in ['left','right']:
                g=f[f'action/{side}_ee_joint_states'][:,0]
                closes=np.where((g[1:]<.3)&(g[:-1]>=.3))[0]+1
                closure_counts.append(len(closes))
                for ordinal,t in enumerate(closes):
                    if ordinal>1:continue
                    pose=f[f'state/{side}_ee_poses'][t]
                    events.append(dict(target=f'{side}_{"sleeve" if ordinal==0 else "hem"}',side=side,
                        close_frame=int(t),quaternion_wxyz=pose[3:].tolist(),window_valid=bool(t>=20 and t+5<frames)))
            frequency=int(f['additional_info/frequency'][()])
        extract=out/'extracted'/row['task']/path.stem;extract.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(extract/'demonstration.npz',**arrays)
        metadata=dict(source=str(path),sha256=row['sha256'],frames=frames,frequency_hz=frequency,events=events,
            semantic_assignment='For fold_clothes only: first closure per arm = sleeve, second = hem. Requires exactly two closures per arm and in-bounds windows; no semantic claim for other tasks.')
        write(extract/'demo.json',metadata)
        row.update(status='verified',frames=frames,extract=str(extract),
                   cloth_windows_valid=row['task']=='fold_clothes' and frequency==25 and closure_counts==[2,2] and len(events)==4 and all(e['window_valid'] for e in events))
        write(manifest_path,manifest)
    write(out/'data-status.json',dict(status='complete',completed=len(manifest['files']),total=len(manifest['files']),
         bytes=sum(x['bytes'] for x in manifest['files']),updated=time.time()))
    print(json.dumps(dict(files=len(manifest['files']),cloth_eligible=sum(x['cloth_windows_valid'] for x in manifest['files'])),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    args.run=args.run.resolve();args.run.mkdir(parents=True,exist_ok=True)
    try:prepare(args.run)
    except Exception as exc:
        write(args.run/'data-status.json',dict(status='failed',error=str(exc),updated=time.time()))
        raise
