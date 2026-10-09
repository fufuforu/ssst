"""Official pose-free evaluation: generate once, calibrate, then re-render views."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
EXPECTED_FULL_VALIDATION_SHA256='59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b'
DISCLOSURE="场景生成只使用两张context；监督/目标相机使用独立图像标定。指定新视角渲染仍需要目标相机。"


def generate_and_render(model, context_rgb, context_plus_target_rgb):
    """One context-only generation, followed by camera-only target calibration."""
    generated=model.generate(context_rgb)
    calibrated=model.calibrate_targets(context_plus_target_rgb,generated)
    cam_view=torch.linalg.inv(calibrated['c2w']).transpose(-1,-2)
    render=model.render_generated_at(generated,cam_view,calibrated['intrinsics'])
    return {'generated':generated,'camera_calibration':calibrated,'render':render}


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True,
        help='pose-free checkpoint_latest.pt or a retained epoch endpoint')
    parser.add_argument('--manifest',type=Path,required=True,
        help='locked full1201 manifest.json')
    parser.add_argument('--cohort',choices=('full_validation','manifest_windows','expanded_train_probe32'),required=True,
        help='select a fixed cohort named by the supplied manifest; full_validation is the registered held-out pair file')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--vggt-revision',required=True,
        help='verified full Hugging Face revision SHA for facebook/VGGT-1B')
    parser.add_argument('--artifact-manifest',type=Path,default=REPO/'vggt_artifact_manifest.json')
    parser.add_argument('--device',default='cuda')
    return parser


def _restore_checkpoint(model, checkpoint_path, revision, manifest_sha256):
    from scripts.object_locus_frozen_vggt_posefree_runtime import restore_model_state_strict
    blob=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    from scripts.object_locus_frozen_vggt_posefree_runtime import EXPECTED_CHECKPOINT_SHA
    current_sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    identity_keys=('repository','model_id','revision','files','loaded_subtrees',
                   'loaded_source_key_count','explicitly_excluded_source_key_count','loaded_key_sha256')
    artifact_identity={k:model.frozen_vggt.source_identity.get(k) for k in identity_keys}
    for key,value in {'vggt_revision':revision,'manifest_sha256':manifest_sha256,
        'source_checkpoint_sha256':EXPECTED_CHECKPOINT_SHA,'git_sha':current_sha,
        'vggt_artifact_identity':artifact_identity}.items():
        if blob.get(key)!=value:raise RuntimeError(f'checkpoint provenance mismatch for {key}')
    restore_model_state_strict(model,blob['model'])
    exposure=int(blob['completed_exposures'])
    if exposure!=int(blob['completed_updates'])*8:
        raise RuntimeError('checkpoint new-training exposure/update clock mismatch')
    model.understanding_step=exposure
    return blob


def run_evaluation(args):
    import hashlib
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f'evaluation output root must be fresh; refusing to overwrite existing files: {args.output_root}')
    from scripts.object_locus_frozen_vggt_posefree_runtime import (
        MANIFEST, build_model, load_manifest, sha256,
    )
    from scripts.object_locus_v3_set_runtime import build_batch
    from scripts.export_object_locus_v3_set_official import write_official_pair
    from scripts.eval_object_locus_v1 import _official_run

    manifest,_,manifest_windows=load_manifest(args.manifest)
    if sha256(args.manifest)!=sha256(MANIFEST):
        raise RuntimeError('evaluation manifest differs from the registered locked full1201 manifest')
    cohort_source=None
    if args.cohort=='manifest_windows':windows=manifest_windows
    elif args.cohort=='expanded_train_probe32':windows=manifest['monitor_splits']['expanded_train_probe32']
    else:
        cohort_source=Path(manifest['full_validation_source'])
        if not cohort_source.is_file():raise FileNotFoundError(f'locked validation cohort missing: {cohort_source}')
        if sha256(cohort_source)!=EXPECTED_FULL_VALIDATION_SHA256:
            raise RuntimeError('registered full-validation cohort SHA256 mismatch')
        source_rows=json.loads(cohort_source.read_text())
        if len(source_rows)!=1860:
            raise RuntimeError(f'registered full-validation cohort count mismatch: {len(source_rows)}')
        windows=[]
        for item in source_rows:
            context=[int(x) for x in item['context_ids']]
            targets=[int(x) for x in item['target_ids']]
            novel=[x for x in targets if x not in set(context)]
            if len(context)!=2 or not novel:raise ValueError('malformed locked full-validation pair')
            windows.append({'scene':item['scan'],'context':context,'novel':novel,
                            'pair_iou':item.get('iou')})
    os.environ['VGGT_HF_REVISION']=args.vggt_revision
    model,opt,_,_=build_model(artifact_manifest=args.artifact_manifest)
    model=model.to(device=args.device,dtype=torch.float32)
    blob=_restore_checkpoint(model,args.checkpoint,args.vggt_revision,sha256(args.manifest))
    model.eval();model.frozen_vggt.eval()
    args.output_root.mkdir(parents=True,exist_ok=True)
    records=[]
    with torch.no_grad():
      for index,window in enumerate(windows):
        batch=build_batch(opt,window,args.device)
        generated=model.generate(batch['images_input'])
        calibrated=model.calibrate_targets(batch['images_all'],generated)
        cam_view=torch.linalg.inv(calibrated['c2w']).transpose(-1,-2)
        output=model.render_generated_at(generated,cam_view,calibrated['intrinsics'])
        # Existing exporter keeps its candidate thresholds, class maps, void
        # handling, packed format and score definition unchanged.
        row=write_official_pair(output,batch,window,args.output_root,target_frames='novel')
        records.append(row)
        del output,generated,calibrated,batch
        if (index+1)%100==0:print(f'prepared {index+1}/{len(windows)} fixed windows',flush=True)
    official=_official_run(args.output_root,args.output_root/'official_metrics.json')
    report={'status':'EVALUATED','cohort':args.cohort,'windows':len(windows),
        'manifest_sha256':sha256(args.manifest),'cohort_source':str(cohort_source) if cohort_source else str(args.manifest),
        'cohort_source_sha256':sha256(cohort_source) if cohort_source else sha256(args.manifest),
        'checkpoint':str(args.checkpoint),
        'new_completed_updates':int(blob['completed_updates']),
        'new_completed_exposures':int(blob['completed_exposures']),
        'source_epoch6_exposures':50064,'vggt_revision':args.vggt_revision,
        'disclosure':DISCLOSURE,'prediction_records':records,
        'official_result':official.get('result')}
    (args.output_root/'evaluation_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA evaluation requires a GPU; use --help for interface inspection only')
    if args.vggt_revision is None or len(args.vggt_revision)!=40:
        raise SystemExit('--vggt-revision must be a full reviewed 40-character SHA')
    result=run_evaluation(args)
    print(json.dumps({k:v for k,v in result.items() if k!='prediction_records'},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
