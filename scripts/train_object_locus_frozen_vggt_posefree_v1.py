"""Review-only entry point. It prepares/prints the locked plan and never submits jobs."""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))

from scripts.object_locus_frozen_vggt_posefree_runtime import (
    CHECKPOINT, EXPECTED_CHECKPOINT_SHA, EXPECTED_VGGT_COMMIT,
    load_manifest, plan_record, sha256,
)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true',help='validate local manifest and print the review plan')
    parser.add_argument('--manifest',type=Path,default=None)
    parser.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    parser.add_argument('--run-dir',type=Path,default=Path('/space/mawb/ssst/workspace_group_plus/object_locus_frozen_vggt_posefree_v1'))
    parser.add_argument('--vggt-revision',default=os.environ.get('VGGT_HF_REVISION'),
                        help='pinned facebook/VGGT-1B Hugging Face commit SHA')
    parser.add_argument('--run-training',action='store_true',help='reserved for a separately reviewed next phase')
    args=parser.parse_args(argv)
    if args.run_training:
        if args.vggt_revision is None or re.fullmatch(r'[0-9a-fA-F]{40}',args.vggt_revision) is None:
            raise SystemExit('--run-training requires --vggt-revision with a reviewed full 40-character HF SHA')
        from scripts.object_locus_frozen_vggt_posefree_runtime import run_training
        result=run_training(manifest_path=args.manifest or Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json'),
                            checkpoint=args.checkpoint,run_dir=args.run_dir,hf_revision=args.vggt_revision)
        print(json.dumps(result,indent=2,sort_keys=True))
        return result
    record=plan_record()
    manifest_path=args.manifest or Path(record['manifest'])
    manifest,scenes,windows=load_manifest(manifest_path)
    record['manifest']=str(manifest_path)
    record['manifest_sha256']=sha256(manifest_path)
    record['actual_train_scenes']=len(scenes)
    record['windows']=len(windows)
    record['checkpoint']={'path':str(args.checkpoint),'expected_sha256':EXPECTED_CHECKPOINT_SHA,
                          'actual_sha256':sha256(args.checkpoint) if args.checkpoint.exists() else None}
    record['vggt']={'repository':'https://github.com/facebookresearch/vggt',
                    'source_commit':EXPECTED_VGGT_COMMIT,'model_id':'facebook/VGGT-1B',
                    'hf_revision':args.vggt_revision,'weights_sha256':'NOT_VERIFIED'}
    record['status']='PREPARED_FOR_REVIEW_ONLY'
    print(json.dumps(record,indent=2,sort_keys=True))
    return record


if __name__=='__main__': main()
