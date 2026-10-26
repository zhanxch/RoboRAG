#!/usr/bin/env python3
import runpy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
runpy.run_module('roborag.supermimic_cloth.learn',run_name='__main__')
