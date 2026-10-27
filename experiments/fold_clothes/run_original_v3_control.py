"""Schedule untouched v3 workers using the already validated seed-aware batch driver."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from roborag import episode_batch as scheduler

    protocol_path = ROOT / 'experiments/fold_clothes/v3-original-control-20261028.json'
    protocol = json.loads(protocol_path.read_text())
    original = Path(protocol['workspace'])
    output = Path(protocol['output'])
    freeze = json.loads(output.with_suffix('.freeze.json').read_text())
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == freeze['protocol_sha256']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=original,
                                   text=True).strip() == protocol['original_commit']
    for path, expected in freeze['driver_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    cases_file = ROOT / protocol['cases_file']
    assert hashlib.sha256(cases_file.read_bytes()).hexdigest() == protocol['cases_sha256']

    # This selects the worker script and source-freeze root, not the model policy.
    # Every worker starts original/scripts/run_rag_mcp.py in its own process.
    # That script inserts original/src first; MCP and simulator PYTHONPATH are
    # explicitly rebuilt by the untouched original runner.
    scheduler.ROOT = original
    conditions = protocol['conditions']
    assert conditions['parallel'] == 2 and conditions['controller'] == 'delta'
    assert conditions['task'] == 'fold_clothes' and not conditions['no_rag']
    assert protocol['seed'] == 1 and len(json.loads(cases_file.read_text())) == 50
    assert scheduler.protocol_sources(Path(conditions['bank'])) == freeze['source_sha256']
    sys.argv = [str(__file__), '--task', 'fold_clothes', '--controller', 'delta',
                '--max-wall-seconds', str(conditions['max_wall_seconds']),
                '--bank', conditions['bank'], '--output', str(output),
                '--cases-file', str(cases_file), '--seed', '1', '--parallel', '2',
                '--model', conditions['model'], '--effort', conditions['effort']]
    scheduler.main()


if __name__ == '__main__':
    main()
