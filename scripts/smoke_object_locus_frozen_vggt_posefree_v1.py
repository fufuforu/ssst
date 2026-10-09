"""CPU contracts and explicitly deferred real-weight GPU smoke entry points."""
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
    parser.add_argument('--single-card-real',action='store_true',help='one allocated RTX3090; isolated one-rank real VGGT smoke')
    parser.add_argument('--eight-card-real',action='store_true',help='torchrun eight RTX3090 real VGGT smoke')
    parser.add_argument('--window-4253-calibration',action='store_true',help='record old signed Sim(3) and verify real shared-context depth Sim(3) v2')
    parser.add_argument('--manifest',type=Path,default=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json'))
    parser.add_argument('--checkpoint',type=Path,default=Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt'))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--vggt-revision')
    parser.add_argument('--artifact-manifest',type=Path,default=REPO/'vggt_artifact_manifest.json')
    args=parser.parse_args()
    if args.cpu_contracts:
        suite=unittest.defaultTestLoader.discover(str(REPO/'tests'),pattern='test_object_locus_frozen_vggt_posefree_contracts.py',top_level_dir=str(REPO/'tests'))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    modes=[args.single_card_real,args.eight_card_real,args.window_4253_calibration]
    if sum(modes)>1:parser.error('select at most one GPU mode')
    if args.window_4253_calibration:
        if not args.vggt_revision or len(args.vggt_revision)!=40:parser.error('window calibration requires verified --vggt-revision')
        if args.output_dir is None:parser.error('window calibration requires --output-dir')
        from scripts.object_locus_frozen_vggt_posefree_runtime import run_window4253_calibration
        run_window4253_calibration(manifest_path=args.manifest,checkpoint=args.checkpoint,
            output_dir=args.output_dir,hf_revision=args.vggt_revision,artifact_manifest=args.artifact_manifest)
    elif any(modes):
        if not args.vggt_revision or len(args.vggt_revision)!=40:parser.error('real smoke requires verified --vggt-revision')
        if args.output_dir is None:parser.error('real smoke requires a separate --output-dir')
        from scripts.object_locus_frozen_vggt_posefree_runtime import run_real_smoke
        mode='eight' if args.eight_card_real else 'single'
        run_real_smoke(mode=mode,manifest_path=args.manifest,checkpoint=args.checkpoint,
            output_dir=args.output_dir,hf_revision=args.vggt_revision,artifact_manifest=args.artifact_manifest)
