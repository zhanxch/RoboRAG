#!/usr/bin/env python3
"""Inventory local RoboDojo tasks, eval fixtures and verified demonstration sources.

An inventory is not a successful simulator test. This does not download data,
train a model, or mark unsupported task adapters as evaluated.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=ROOT/'runs/cloth-grasp-bench-v1')
    p.add_argument('--data-root',type=Path,action='append',default=[])
    args=p.parse_args()
    upstream=ROOT/'vendor/RoboICL'
    native=upstream/'third_party/RoboDojo'
    source=json.loads((upstream/'configs/references/one_shot_j12_b12.json').read_text())
    references={r['task']:r['source'] for r in source['one_shot_references']}
    layouts=defaultdict(Counter)
    for path in (ROOT/'data/Assets/Eval_Layout/RoboDojo/arx_x5').glob('*/*.json'):
        task=path.stem.rsplit('_',1)[0]
        layouts[task][path.parent.name]+=1
    roots=args.data_root or [
        Path('/home/zxc/文档/ChatGPT/infiniDojo/reproduction/policy_compare/data/roboicl/runtime-data'),
        Path('/home/zxc/文档/ChatGPT/infiniDojo/reproduction/fold_clothes_compare/data/roboicl/runtime-data')]
    demos=defaultdict(list);seen=set()
    for root in roots:
        for path in root.glob('*/train/*.hdf5'):
            real=path.resolve()
            if real in seen:continue
            seen.add(real)
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
            sha=digest.hexdigest()
            demos[path.parent.parent.name].append(dict(path=str(real),sha256=sha,bytes=path.stat().st_size,
                                                      split_provenance='existing local TRAIN directory; source asset identity not present in HDF5'))
    rows=[]
    for path in sorted((native/'task/RoboDojo/config').glob('*.yml')):
        task=path.stem
        if task.startswith('_'):continue
        base=task.removesuffix('_random')
        types=sorted(set(re.findall(r'^(Rigid|Articulation|Garment|Fluid|Dynamic|Clutter):',path.read_text(),re.M)))
        family=('cloth' if 'Garment' in types else 'fluid_transfer' if 'Fluid' in types else
                'contact_or_pushing' if base in ['push_T','swap_T','press_by_number','play_Xylophone'] else
                'assembly' if base in ['insert_key','insert_tubes','plug_in_charger','fasten_screws'] else
                'dynamic_pickup' if 'conveyor' in base else
                'articulated_manipulation' if 'Articulation' in types else 'rigid_pick_place')
        docpath=upstream/'configs/tasks'/(base+'.json')
        doc=json.loads(docpath.read_text()).get('official_task_documentation',{}) if docpath.exists() else {}
        local=demos.get(base,[])
        rows.append(dict(task=task,base_task=base,variant=task.endswith('_random'),family=family,
                         object_systems=types,eval_layout_counts=dict(layouts[task]),
                         eval_layouts_total=sum(layouts[task].values()),train_hdf5=local,
                         train_hdf5_count=len(local),locked_reference=references.get(base),
                         description=doc.get('description'),documentation=doc.get('source'),
                         native_task_source_exists=(native/'task/RoboDojo/tasks'/(task+'.py')).exists(),
                         small_policy_adapter='local cloth subskill only' if task=='fold_clothes' else 'not implemented or validated',
                         full_task_evaluated=False))
    summary=dict(base_tasks=sum(not r['variant'] for r in rows),randomized_variants=sum(r['variant'] for r in rows),
                 total_configs=len(rows),eval_layout_files=sum(r['eval_layouts_total'] for r in rows),
                 local_unique_train_episodes=len(seen),tasks_with_local_train_data=len(demos),
                 families=dict(Counter(r['family'] for r in rows)),
                 full_tasks_evaluated_by_this_policy=0)
    value=dict(schema='robodojo.grasp.coverage.v1',generated_at=time.time(),summary=summary,tasks=rows,
               source_repository=source['source_repository'],
               interpretation='Filesystem availability only. Configs, layouts and demonstration files do not imply runtime coverage or policy success.',
               observed_dataset_website='https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo',
               development_fixture_exclusion=['fold_clothes / layout 0 / seed 0 was used to derive local development fixtures; do not claim it as an untouched official test'],
               next_stage=dict(status='prepared_not_started',condition='Complete pilot audit and establish reproducible grasp effectiveness before broad rollout.',
                   pilot_tasks=['fold_clothes','general_pickup','stack_blocks','classify_objects','fill_pen_holder','hang_mugs'],
                   data_curve_demonstrations_per_task=[1,10,50],training_seeds=[31001,31002,31003],
                   authorized_wall_budget_hours=72,compute='Local NVIDIA RTX 4090',
                   budget_authorization='User explicitly permits continuous local execution for up to 72 hours',
                   splits='Freeze episode identities and object/layout partitions before fitting. Never train on evaluation rollouts. All checkpoints selected on development data only.',
                   evaluation='Paired BC / BC+RL, macro-average by task, per-task and per-layout scores, uncertainty intervals across training seeds, all failures and timeouts retained.',
                   native_scores='Native full-task score is separate from local grasp success; use the same high-level controller for both low-level policies.',
                   unresolved=['Rigid and contact-task observation/action adapters are not implemented for this cloth policy','Additional TRAIN sources must be verified and downloaded by pinned revision and content hash','Separate train simulation layouts and object partitions must be constructed without using test fixtures','Measure per-task runtime before committing full coverage within a wall budget']))
    args.run.mkdir(parents=True,exist_ok=True)
    (args.run/'coverage.json').write_text(json.dumps(value,ensure_ascii=False,indent=2))
    (ROOT/'experiments/robodojo-grasp-scale-v1.json').write_text(json.dumps(value,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
