"""CPU-only contract runner; it never loads real VGGT weights or performs a model forward."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-contracts',action='store_true')
    args=parser.parse_args()
    if args.cpu_contracts:
        suite=unittest.defaultTestLoader.discover(str(REPO/'tests'),pattern='test_object_locus_frozen_vggt_posefree_contracts.py',top_level_dir=str(REPO/'tests'))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
