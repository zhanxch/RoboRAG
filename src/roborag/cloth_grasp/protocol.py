"""Pre-registered local grasp benchmark. Pure stdlib; no simulator import."""
import hashlib
import json
from pathlib import Path
import time

TARGETS = ['left_sleeve', 'right_sleeve', 'left_hem', 'right_hem']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def freeze(path, demo_sha256):
    path = Path(path)
    if path.exists():
        return read(path)
    offsets = [('center', [0., 0., .065]), ('diagonal', [.02, -.02, .065]),
               ('high_opposite', [-.02, .02, .09])]
    cases = []
    for garment in [9, 4]:
        for target in TARGETS:
            for label, offset in offsets:
                i = len(cases) + 1
                cases.append(dict(id=f'c{i:03d}', task='fold_clothes', subtask=target,
                                  garment=garment, distribution='seen_garment' if garment == 9 else 'unseen_garment',
                                  position=label, tcp_offset_m=offset, seed=800000+i,
                                  garment_position=[0., -.12, .8441], garment_yaw_deg=0.))
    value = dict(schema='cloth-grasp.benchmark.v1', frozen_at=time.time(),
                 demo_sha256=demo_sha256, cases=cases, policies=['bc', 'rl'],
                 training=dict(seed=3042, garment=9, episodes=20, warmup=12,
                               demo_retarget_episodes=8, geometry_teacher_episodes=4,
                               world_holdout_episode_numbers=[5, 6], bc_steps=1600,
                               world_window=8, imagination_horizon=16, policy_updates_per_round=5,
                               bc_weight=20., binary_gripper=True),
                 criterion=dict(lift_m=.08, max_target_tcp_gap_m=.055, closed_command_max=.35,
                                final_consecutive_steps=10, horizon=80, control_hz=25),
                 selection='Fixed final training update; no selection or tuning on benchmark outcomes.',
                 comparisons='Paired BC initialization versus final RL on the same 24 cases; both frozen.',
                 invalid_case_policy='Unreachable pregrasp is a recorded failure, never silently removed; infrastructure errors halt the run.',
                 scope='One parent task, four local subtasks; not four independent RoboDojo tasks. Privileged particle geometry, fixed orientation, no full-fold or real-robot claim.',
                 provenance='Development scene fixture cloned with explicitly fixed garment pose. This is a custom local benchmark, not the official RoboDojo score.')
    value['sha256'] = digest(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    return value


def read(path):
    value = json.loads(Path(path).read_text())
    expected = value.pop('sha256')
    if digest(value) != expected:
        raise ValueError('Frozen benchmark digest mismatch')
    value['sha256'] = expected
    return value


def wilson(successes, total):
    if not total:
        return [0., 1.]
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total)) / denom
    radius = z*((p*(1-p)/total + z*z/(4*total*total))**.5)/denom
    return [center-radius, center+radius]


def summarize(protocol, rows):
    test = [row for row in rows if row['role'] == 'test']
    expected = {(p, c['id']) for p in protocol['policies'] for c in protocol['cases']}
    observed = [(r['policy'], r['case_id']) for r in test]
    if len(set(observed)) != len(observed):
        raise ValueError('Duplicate benchmark outcome')
    if set(observed) != expected:
        raise ValueError(f'Incomplete benchmark: {len(observed)} / {len(expected)} outcomes')
    models = {}
    for policy in protocol['policies']:
        values = [r for r in test if r['policy'] == policy]
        n = sum(r['success'] for r in values)
        groups = {}
        for key in ['target', 'distribution', 'position']:
            groups[key] = {v: dict(successes=sum(r['success'] for r in values if r[key] == v),
                                   total=sum(r[key] == v for r in values))
                           for v in sorted({r[key] for r in values})}
        models[policy] = dict(successes=n, total=len(values), rate=n/len(values),
                              wilson95=wilson(n, len(values)), groups=groups)
    lookup = {(r['policy'], r['case_id']): r for r in test}
    pairs = [dict(case_id=c['id'], bc=lookup['bc', c['id']]['success'],
                  rl=lookup['rl', c['id']]['success']) for c in protocol['cases']]
    return dict(protocol_sha256=protocol['sha256'], models=models, pairs=pairs,
                rl_only=sum(p['rl'] and not p['bc'] for p in pairs),
                bc_only=sum(p['bc'] and not p['rl'] for p in pairs),
                note='24 deliberately structured cases, one training seed. Wilson intervals are descriptive; cases are not an IID sample of all clothing.')
