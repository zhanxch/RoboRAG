import json
from pathlib import Path
import tempfile
import unittest

from roborag.cloth_grasp.protocol import freeze, read, summarize


class ClothProtocolTests(unittest.TestCase):
    def test_frozen_protocol_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'benchmark.json'
            p=freeze(path,'train-sha')
            self.assertEqual(len(p['cases']),24)
            self.assertEqual(len({c['id'] for c in p['cases']}),24)
            self.assertEqual(p['sha256'],freeze(path,'train-sha')['sha256'])
            p['cases'][0]['tcp_offset_m'][0]=1
            path.write_text(json.dumps(p))
            with self.assertRaises(ValueError):read(path)

    def test_summary_requires_exactly_one_outcome_per_policy_case(self):
        with tempfile.TemporaryDirectory() as folder:
            p=freeze(Path(folder)/'benchmark.json','train-sha')
            rows=[dict(role='test',policy=name,case_id=c['id'],target=c['subtask'],
                       distribution=c['distribution'],position=c['position'],success=name=='bc')
                  for c in p['cases'] for name in p['policies']]
            summary=summarize(p,rows)
            self.assertEqual(summary['models']['bc']['successes'],24)
            self.assertEqual(summary['bc_only'],24)
            with self.assertRaises(ValueError):summarize(p,rows[:-1])
            with self.assertRaises(ValueError):summarize(p,rows+[rows[0]])


if __name__=='__main__':unittest.main()
