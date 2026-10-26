#!/usr/bin/env python3
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from roborag.supermimic_cloth.audit import audit
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--package',action='store_true');a=p.parse_args()
r=audit(a.run,a.package);print(json.dumps({k:v for k,v in r.items() if k!='episodes'}))
