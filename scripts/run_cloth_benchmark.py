#!/usr/bin/env python3
"""Run a frozen local cloth experiment, one owned simulator process at a time."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from roborag.cloth_grasp.protocol import freeze, read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--cases-per-worker',type=int,choices=[1,2,3,4,6],default=4)
    args=parser.parse_args()
    out=args.run.resolve()
    demo=json.loads((out/'demo.json').read_text())
    protocol=freeze(out/'benchmark.json',demo['sha256'])
    print('Frozen benchmark:',protocol['sha256'],len(protocol['cases']),'cases',flush=True)
    if args.freeze_only:return
    with open('/tmp/roborag-cloth-grasp-sim.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.write(str(os.getpid()));lock.flush()
        jobs=[]
        if not (out/'checkpoint-freeze.json').exists() or not (out/'final.pt').exists():
            jobs.append(('training','train',[]))
        batch=args.cases_per_worker
        jobs += [(f'test-{i//batch+1}','evaluate',['--case-start',str(i),'--case-count',str(batch)]) for i in range(0,24,batch)]
        for name,mode,extra in jobs:
            read(out/'benchmark.json')
            cmd=[sys.executable,str(ROOT/'scripts/cloth_grasp.py'),mode,'--run',str(out),
                 '--protocol',str(out/'benchmark.json'),'--seed',str(protocol['training']['seed']),*extra]
            print('Starting',name,flush=True)
            with (out/(name+'.log')).open('a') as log:
                log_offset=log.tell()
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                began=time.time()
                completed_at=None
                try:
                    while child.poll() is None:
                        try:status=json.loads((out/'status.json').read_text())
                        except (FileNotFoundError,json.JSONDecodeError):status={}
                        fresh=status.get('updated',0)>=began
                        log_path=out/(name+'.log')
                        with log_path.open('rb') as reader:
                            reader.seek(max(log_offset,log_path.stat().st_size-8192))
                            recent_log=reader.read().decode(errors='replace')
                        if fresh and status.get('status')!='complete' and 'ERROR_OUT_OF_DEVICE_MEMORY' in recent_log:
                            value=dict(status='failed',error='Renderer GPU memory exhausted; saved checkpoints and completed cases are preserved',updated=time.time())
                            temp=out/'status.json.tmp';temp.write_text(json.dumps(value));temp.replace(out/'status.json')
                            raise RuntimeError(value['error'])
                        if fresh and status.get('status')=='failed':
                            raise RuntimeError(status.get('error','Simulator failed'))
                        if fresh and status.get('status')=='complete':
                            completed_at=completed_at or time.time()
                            # Isaac shutdown can leave renderer threads alive. Completion
                            # is written only after all artifacts are saved. Reap this
                            # owned process group before the next simulator is launched.
                            if time.time()-completed_at>15:
                                os.killpg(child.pid,signal.SIGKILL)
                                child.wait()
                                break
                        time.sleep(2)
                    if not completed_at:
                        status=json.loads((out/'status.json').read_text())
                        if status.get('status')!='complete' or status.get('updated',0)<began:
                            raise RuntimeError(f'{name} did not finish: exit={child.returncode}, status={status}')
                finally:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait()
            print('Completed',name,flush=True)
        if not (out/'benchmark-results.json').exists():raise RuntimeError('Missing paired benchmark results')
        print('Experiment complete:',out,flush=True)


if __name__=='__main__':main()
