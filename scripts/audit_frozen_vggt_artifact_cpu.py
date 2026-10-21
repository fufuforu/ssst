"""CPU-only strict checkpoint key/shape audit for the pinned official VGGT artifact.

This command loads parameters and inspects dtypes/freeze state. It never invokes
the aggregator, camera head, depth head, or any model forward.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-manifest',type=Path,default=REPO/'vggt_artifact_manifest.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    import torch
    torch.set_num_threads(8)
    from tokengs.models.frozen_vggt_posefree import FrozenVGGT,verify_official_source_revision
    manifest=json.loads(args.artifact_manifest.read_text())
    if manifest.get('status')!='VERIFIED':raise RuntimeError('artifact manifest is not marked VERIFIED')
    source_commit=verify_official_source_revision()
    if source_commit!=manifest.get('official_source_commit'):
        raise RuntimeError('official VGGT source checkout differs from artifact manifest')
    model=FrozenVGGT.from_pretrained(local_files_only=True,revision=manifest['hf_revision'],
                                     artifact_manifest=args.artifact_manifest)
    identity=model.source_identity
    aggregator_dtypes=sorted({str(p.dtype).removeprefix('torch.') for p in model.model.aggregator.parameters()})
    camera_dtypes=sorted({str(p.dtype).removeprefix('torch.') for p in model.model.camera_head.parameters()})
    depth_dtypes=sorted({str(p.dtype).removeprefix('torch.') for p in model.model.depth_head.parameters()})
    if any(p.requires_grad for p in model.parameters()) or model.training or model.model.training:
        raise AssertionError('VGGT frozen/eval contract failed')
    if aggregator_dtypes!=['bfloat16'] or camera_dtypes!=['float32'] or depth_dtypes!=['float32']:
        raise AssertionError('VGGT BF16 aggregator / FP32 head precision boundary failed')
    report={'status':'PASS_CPU_STRICT_KEYS_NO_FORWARD','source_commit':source_commit,
        'hf_revision':identity['revision'],'files':identity['files'],
        'loaded_subtrees':identity['loaded_subtrees'],
        'loaded_source_key_count':identity['loaded_source_key_count'],
        'target_model_key_count':identity['target_model_key_count'],
        'loaded_key_sha256':identity['loaded_key_sha256'],
        'explicitly_excluded_source_key_count':identity['explicitly_excluded_source_key_count'],
        'explicitly_excluded_source_keys':identity['explicitly_unused_source_keys'],
        'missing_key_count':identity['missing_key_count'],
        'unexpected_key_count':identity['unexpected_key_count'],
        'shape_mismatch_key_count':identity['shape_mismatch_key_count'],
        'aggregator_dtypes':aggregator_dtypes,'camera_head_dtypes':camera_dtypes,
        'depth_head_dtypes':depth_dtypes,'all_parameters_frozen':True,'eval':True,
        'forward_executed':False}
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='explicitly_excluded_source_keys'}))
    del model
    return 0


if __name__=='__main__':raise SystemExit(main())
