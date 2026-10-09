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


def _official_export_view(rendered):
    """Adapt pose-free render fields to the existing official export contract."""
    result=dict(rendered)
    result['render']={key:rendered[key] for key in ('images_pred','depths_pred','alphas_pred')
                      if key in rendered}
    if 'images_pred' not in result['render']:
        raise KeyError('pose-free renderer omitted RGB needed by the official exporter')
    return result


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
    from scripts.object_locus_frozen_vggt_posefree_runtime import (
        EXPECTED_CHECKPOINT_SHA, WORLD_SIZE, TOTAL_UPDATES, training_configuration,
    )
    current_sha=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    identity_keys=('repository','model_id','revision','files','loaded_subtrees',
                   'loaded_source_key_count','explicitly_excluded_source_key_count','loaded_key_sha256')
    artifact_identity={k:model.frozen_vggt.source_identity.get(k) for k in identity_keys}
    for key,value in {'vggt_revision':revision,'manifest_sha256':manifest_sha256,
        'source_checkpoint_sha256':EXPECTED_CHECKPOINT_SHA,
        'vggt_artifact_identity':artifact_identity}.items():
        if blob.get(key)!=value:raise RuntimeError(f'checkpoint provenance mismatch for {key}')
    if blob.get('world_size')!=WORLD_SIZE or blob.get('total_updates')!=TOTAL_UPDATES:
        raise RuntimeError('evaluation checkpoint is not from the locked eight-rank schedule')
    if blob.get('config')!=training_configuration() or len(blob.get('rank_rng',[]))!=WORLD_SIZE:
        raise RuntimeError('evaluation checkpoint training config/rank RNG structure mismatch')
    restore_model_state_strict(model,blob['model'])
    exposure=int(blob['completed_exposures'])
    if exposure!=int(blob['completed_updates'])*8:
        raise RuntimeError('checkpoint new-training exposure/update clock mismatch')
    model.understanding_step=exposure
    blob['evaluation_code_sha']=current_sha
    return blob


def _save_reconstruction_cache(root,window,batch,render):
    import numpy as np
    required=('images_pred','depths_pred')
    missing=[key for key in required if key not in render]
    if missing:raise KeyError(f'pose-free evaluator renderer is missing reconstruction fields: {missing}')
    frame_ids=batch['frame_ids'][0].detach().cpu().numpy().astype(np.int64)
    context_ids=np.asarray(window['context'],dtype=np.int64)
    novel_ids=np.asarray([int(x) for x in window['novel'] if int(x) not in set(context_ids.tolist())],dtype=np.int64)
    target=Path(root)/(str(window['scene'])+'_context'+'_'.join(map(str,context_ids.tolist())))/'reconstruction_cache.npz'
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,
        frame_ids=frame_ids,context_ids=context_ids,novel_ids=novel_ids,
        pred_rgb=render['images_pred'][0].detach().float().cpu().numpy(),
        gt_rgb=batch['images_all'][0].detach().float().cpu().numpy(),
        pred_depth=render['depths_pred'][0].detach().float().cpu().numpy(),
        gt_depth_m=batch['depth_gt_m_all'][0].detach().float().cpu().numpy(),
        depth_valid=batch['depth_gt_valid_all'][0].detach().bool().cpu().numpy())
    return target


def summarize_reconstruction_caches(root):
    """Use pinned SIU3R image metrics and its per-image scale/shift depth fit."""
    import math
    import numpy as np
    import sys
    sys.path.insert(0,'/space/mawb/SIU3R')
    from src.config import EvaluatorCfg
    from src.evaluator import Evaluator
    from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME,STUFF_CLASSES,THING_CLASSES
    root=Path(root)
    cfg=EvaluatorCfg(dataset_name='scannet',eval_context_miou=False,eval_context_pq=False,
        eval_context_map=False,eval_target_miou=False,eval_target_pq=False,eval_target_map=False,
        eval_image_quality=True,eval_depth_quality=False,id2label=PANOPTIC_SEMANTIC2NAME,
        stuffs=STUFF_CLASSES,things=THING_CLASSES,device='cpu',eval_path=str(root))
    evaluator=Evaluator(cfg);evaluator.setup()
    scopes=('context','target-all','true-novel')
    totals={scope:{key:[] for key in ('psnr','ssim','lpips','absrel','rmse')}|{'undefined_depth':0} for scope in scopes}
    per_image=[]
    for cache_path in sorted(root.glob('*/reconstruction_cache.npz')):
        with np.load(cache_path) as z:data={key:z[key] for key in z.files}
        frame_ids=data['frame_ids'].tolist();context=set(data['context_ids'].tolist());novel=set(data['novel_ids'].tolist())-context
        selected={'context':[i for i,f in enumerate(frame_ids) if f in context],
                  'target-all':list(range(len(frame_ids))),
                  'true-novel':[i for i,f in enumerate(frame_ids) if f in novel]}
        for scope,indices in selected.items():
            for i in indices:
                pred=torch.from_numpy(data['pred_rgb'][i]).float()[None]
                truth=torch.from_numpy(data['gt_rgb'][i]).float()[None]
                rgb={'psnr':float(evaluator.psnr(pred,truth).item()),
                     'ssim':float(evaluator.ssim(pred,truth).item()),
                     'lpips':float(evaluator.lpips(pred,truth).item())}
                evaluator.psnr.reset();evaluator.ssim.reset();evaluator.lpips.reset()
                if not all(math.isfinite(value) for value in rgb.values()):raise FloatingPointError(f'nonfinite SIU3R image metrics: {cache_path}')
                pred_depth=torch.from_numpy(data['pred_depth'][i]).float()/0.15
                gt_depth=torch.from_numpy(data['gt_depth_m'][i]).float()
                valid=torch.from_numpy(data['depth_valid'][i]).bool()&(gt_depth>0)
                depth={'absrel':None,'rmse':None,'scale':None,'shift':None}
                if valid.any():
                    scale,shift=evaluator.fit_scale_and_shift(pred_depth,torch.where(valid,gt_depth,torch.zeros_like(gt_depth)))
                    error=(pred_depth*scale+shift)[valid]-gt_depth[valid]
                    depth={'absrel':float((error.abs()/gt_depth[valid]).mean()),
                           'rmse':float(error.square().mean().sqrt()),'scale':float(scale),'shift':float(shift)}
                    if not all(math.isfinite(v) for v in depth.values()):raise FloatingPointError(f'nonfinite SIU3R aligned depth: {cache_path}')
                else:totals[scope]['undefined_depth']+=1
                for key,value in rgb.items():totals[scope][key].append(value)
                for key in ('absrel','rmse'):
                    if depth[key] is not None:totals[scope][key].append(depth[key])
                per_image.append({'cache':cache_path.name,'frame_id':frame_ids[i],'scope':scope,**rgb,**depth,'depth_defined':depth['absrel'] is not None})
    summary={scope:{key:(float(np.mean(values)) if isinstance(values,list) and values else
                         (None if isinstance(values,list) else values))
                      for key,values in metrics.items()} for scope,metrics in totals.items()}
    payload={'protocol':'Pinned SIU3R Evaluator PSNR/SSIM/LPIPS; per-image SIU3R fit_scale_and_shift on valid positive GT depth; render depth / 0.15 then affine-aligned; RGB/depth renders use context-only generation and independent predicted camera calibration.',
             'scopes':summary,'images':per_image,'cache_count':len(list(root.glob('*/reconstruction_cache.npz')))}
    (root/'reconstruction_metrics.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    return payload


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
        export_view=_official_export_view(output)
        cache_path=_save_reconstruction_cache(args.output_root,window,batch,export_view['render'])
        # Existing exporter keeps its candidate thresholds, class maps, void
        # handling, packed format and score definition unchanged.
        row=write_official_pair(export_view,batch,window,args.output_root,target_frames='novel')
        records.append(row)
        del output,generated,calibrated,batch
        if (index+1)%100==0:print(f'prepared {index+1}/{len(windows)} fixed windows',flush=True)
    official=_official_run(args.output_root,args.output_root/'official_metrics.json')
    reconstruction_metrics=summarize_reconstruction_caches(args.output_root)
    report={'status':'EVALUATED','cohort':args.cohort,'windows':len(windows),
        'unique_scenes':len({str(window['scene']) for window in windows}),
        'manifest_sha256':sha256(args.manifest),'cohort_source':str(cohort_source) if cohort_source else str(args.manifest),
        'cohort_source_sha256':sha256(cohort_source) if cohort_source else sha256(args.manifest),
        'checkpoint':str(args.checkpoint),
        'new_completed_updates':int(blob['completed_updates']),
        'new_completed_exposures':int(blob['completed_exposures']),
        'source_epoch6_exposures':50064,'vggt_revision':args.vggt_revision,
        'training_code_sha':blob.get('git_sha'),'evaluation_code_sha':blob.get('evaluation_code_sha'),
        'reconstruction_metrics_path':str(args.output_root/'reconstruction_metrics.json'),
        'reconstruction_metric_scopes':reconstruction_metrics['scopes'],
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
