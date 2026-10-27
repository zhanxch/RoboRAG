"""Read-only historical/candidate launch and initial observation comparison."""
import hashlib
import json
import re
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = ROOT.parent / 'fold-v3-frozen-baseline'
BASE = BASE_ROOT / 'runs/v3-final-seed1-50'
RUN = ROOT / 'runs/v3-clean-fold-20261027'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launch(directory, repository):
    invocation = read(directory / 'agent-invocation.json')
    workspace = invocation['workspace']
    command = [re.sub(r'http://127\.0\.0\.1:\d+/mcp', 'http://127.0.0.1:PORT/mcp',
                      token.replace(workspace, 'WORKSPACE').replace(str(repository), 'REPO'))
               for token in invocation['command']]
    metadata = {k: v for k, v in invocation.items() if k not in ('command', 'workspace')}
    warnings = []
    with (directory / 'agent-events.jsonl').open() as stream:
        for index, line in enumerate(stream):
            if index >= 8:
                break
            item = read_event(line).get('item') or {}
            message = item.get('message', '')
            if item.get('type') == 'error' and message.startswith((
                    'Under-development features enabled:', 'Exceeded skills context budget.')):
                warnings.append(message)
    return command, metadata, warnings


def read_event(line):
    return json.loads(line)


cases = read(ROOT / 'experiments/fold_clothes/v3-clean-cases.json')
rows = []
for case in cases:
    name = f"seed{case['seed']}-{case['variant']}-l{case['layout']}"
    before, after = BASE / name, RUN / name
    a, b = launch(before, BASE_ROOT), launch(after, ROOT)
    observations = []
    for directory in (before, after):
        with (directory / 'commands.jsonl').open() as stream:
            first = read_event(next(stream))
        assert first['step_before'] == first['step_after'] == 0, name
        observations.append(Path(first['observation']))
    x, y = observations
    sx, sy = read(x / 'state.json'), read(y / 'state.json')
    images = []
    for camera in ('head', 'wrist_l', 'wrist_r'):
        px, py = x / f'{camera}.png', y / f'{camera}.png'
        ix, iy = [np.asarray(Image.open(p).convert('RGB')).astype(np.int16) for p in (px, py)]
        assert ix.shape == iy.shape, (name, camera)
        diff = np.abs(ix - iy)
        images.append(dict(camera=camera, baseline=str(px), candidate=str(py),
                           baseline_png_sha256=sha(px), candidate_png_sha256=sha(py),
                           pixel_equal=bool(np.array_equal(ix, iy)),
                           mean_absolute_channel_difference_0_255=float(diff.mean()),
                           max_channel_difference=int(diff.max()),
                           changed_pixel_fraction=float(np.any(diff > 0, axis=2).mean()),
                           over_10_pixel_fraction=float(np.any(diff > 10, axis=2).mean())))
    rows.append(dict(case=name, normalized_command_equal=a[0] == b[0],
                     invocation_metadata_equal=a[1] == b[1], startup_notices_equal=a[2] == b[2],
                     baseline_command_length=len(a[0]), candidate_command_length=len(b[0]),
                     command_differences=[dict(index=i, baseline=u, candidate=v)
                                          for i, (u, v) in enumerate(zip(a[0], b[0])) if u != v],
                     initial_state_equal=sx == sy,
                     differing_state_fields=[k for k in sorted(set(sx) | set(sy)) if sx.get(k) != sy.get(k)],
                     baseline_state_sha256=sha(x / 'state.json'), candidate_state_sha256=sha(y / 'state.json'),
                     images=images))
frames = [image for row in rows for image in row['images']]
summary = dict(cases=len(rows), normalized_commands_equal=sum(r['normalized_command_equal'] for r in rows),
               invocation_metadata_equal=sum(r['invocation_metadata_equal'] for r in rows),
               startup_notices_equal=sum(r['startup_notices_equal'] for r in rows),
               initial_states_equal=sum(r['initial_state_equal'] for r in rows),
               frames=len(frames), pixel_identical_frames=sum(f['pixel_equal'] for f in frames),
               mean_frame_mae=statistics.mean(f['mean_absolute_channel_difference_0_255'] for f in frames),
               max_frame_mae=max(f['mean_absolute_channel_difference_0_255'] for f in frames),
               max_over_10_pixel_fraction=max(f['over_10_pixel_fraction'] for f in frames))
report = dict(schema='roborag.v3-clean-input-audit.v1', baseline=str(BASE), candidate=str(RUN),
              summary=summary, cases=rows,
              scope=['Launch normalization replaces only workspace/repository locations and localhost MCP ports.',
                     'Only the two known startup notice types in the first eight CLI events are compared.',
                     'Initial state and RGB are compared at the first saved command with zero control steps.',
                     'Pixel differences are observations, not evidence that they caused outcome changes.',
                     'Historical provider snapshot and executable bytes were not recorded; equal command strings do not prove identical backend behavior.'])
output = RUN.with_suffix('.input-audit.json')
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(output=str(output), **summary), ensure_ascii=False))
