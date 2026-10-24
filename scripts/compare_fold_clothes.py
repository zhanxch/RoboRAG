#!/usr/bin/env python3
"""Compare recorded fold episodes against the local HexaShell baseline."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from roborag.fold_comparison import main
if __name__ == "__main__":
    main()
