"""Compare complete archived evidence responses, including decoded image bytes."""
import asyncio
import base64
import hashlib
import json
import tempfile
from pathlib import Path

from mcp import Client
from roborag.rag_mcp import EpisodeMemory, create_server

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'fold-v3-frozen-baseline/runs/v3-final-seed1-50'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(block):
    if hasattr(block, 'model_dump'):
        block = block.model_dump(by_alias=True, exclude_none=True)
    if block['type'] == 'text':
        return dict(type='text', text=block['text'])
    assert block['type'] == 'image', block['type']
    raw = base64.b64decode(block['data'], validate=True)
    return dict(type='image', mimeType=block['mimeType'], sha256=digest(raw), bytes=len(raw))


async def main():
    rows = []
    for directory in sorted(BASE.glob('seed1-*-l*')):
        if not directory.is_dir():
            continue
        config = json.loads((directory / 'run.json').read_text())
        calls = []
        with (directory / 'agent-events.jsonl').open() as stream:
            for line in stream:
                event = json.loads(line)
                item = event.get('item') or {}
                if event.get('type') == 'item.completed' and item.get('tool') in (
                        'retrieve_evidence', 'read_action_reference'):
                    assert not item.get('error'), (directory.name, item.get('error'))
                    calls.append(item)
        count = dict(retrieve_evidence=0, read_action_reference=0, failed_responses=0,
                     images=0, image_bytes=0)
        with tempfile.TemporaryDirectory(prefix='v3-evidence-audit-') as temporary:
            memory = EpisodeMemory(config['args']['bank'], ROOT / 'data', temporary,
                                   temporary, 'fold_clothes', controller='delta')
            async with Client(create_server(memory)) as client:
                for item in calls:
                    args = item['arguments']
                    if isinstance(args, str):
                        args = json.loads(args)
                    response = await client.call_tool(item['tool'], args)
                    expected_error = item['status'] == 'failed'
                    assert bool(response.is_error) == expected_error, (directory.name, item['tool'])
                    before = [canonical(block) for block in item['result']['content']]
                    after = [canonical(block) for block in response.content]
                    assert before == after, (directory.name, item['tool'], args)
                    if expected_error:
                        count['failed_responses'] += 1
                    else:
                        count[item['tool']] += 1
                    count['images'] += sum(block['type'] == 'image' for block in before)
                    count['image_bytes'] += sum(block.get('bytes', 0) for block in before)
        rows.append(dict(case=directory.name, responses_equal=True,
                         events_sha256=digest((directory / 'agent-events.jsonl').read_bytes()), **count))
    assert len(rows) == 50
    totals = {key: sum(row[key] for row in rows) for key in (
        'retrieve_evidence', 'read_action_reference', 'failed_responses', 'images', 'image_bytes')}
    assert totals['retrieve_evidence'] == 53 and totals['read_action_reference'] == 241, totals
    report = dict(schema='roborag.v3-visual-history-replay.v1', passed=True,
                  baseline=str(BASE), totals=totals, cases=rows,
                  scope='All archived evidence response text, image order, MIME types and decoded image bytes compared against current extracted MCP tools. No model calls or robot movement; no claim of native outcome equivalence.')
    target = ROOT / 'results/v3-clean-visual-history-20261027.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(target), passed=True, cases=len(rows), **totals)))


if __name__ == '__main__':
    asyncio.run(main())
