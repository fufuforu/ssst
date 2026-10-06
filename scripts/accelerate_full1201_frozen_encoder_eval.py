"""Resume Full1201 evaluation from immutable exports; never loads a model."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPORT = Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu')
BASE = Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
OLD_EVAL = REPORT / 'evaluation'
OUT = Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu/evaluation_accelerated')
SIU3R = Path('/space/mawb/SIU3R')
SIU3R_PIN = '8ea80166be76854f938e90521f1a5b688b755c87'
sys.path.insert(0, str(SIU3R))
from src.config import EvaluatorCfg
from src.evaluator import Evaluator
from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME, STUFF_CLASSES, THING_CLASSES
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity

SCOPES = ('context', 'target_all', 'true_novel')


def assert_siu3r_pin():
    import subprocess
    got = subprocess.check_output(['git', '-C', str(SIU3R), 'rev-parse', 'HEAD'], text=True).strip()
    if got != SIU3R_PIN:
        raise RuntimeError(f'SIU3R commit mismatch: {got}')


def evaluator_cfg(scope, *, device='cpu', image=False, depth=False, seg=True, map_cpu=False):
    ctx = scope == 'context'
    cfg = EvaluatorCfg(
        dataset_name='scannet',
        eval_context_miou=seg and ctx, eval_context_pq=seg and ctx, eval_context_map=seg and ctx,
        eval_target_miou=seg and not ctx, eval_target_pq=seg and not ctx, eval_target_map=seg and not ctx,
        eval_image_quality=image, eval_depth_quality=depth,
        id2label=PANOPTIC_SEMANTIC2NAME, stuffs=STUFF_CLASSES, things=THING_CLASSES,
        device=device, eval_path='.',
    )
    ev = Evaluator(cfg)
    ev.setup()
    if map_cpu and seg:
        metric = MeanAveragePrecision(iou_type='segm', class_metrics=True, sync_on_compute=False).to('cpu')
        if ctx:
            ev.context_mAP = metric
            ev.context_map_preds, ev.context_map_gts = CPUList(), CPUList()
        else:
            ev.target_mAP = metric
            ev.target_map_preds, ev.target_map_gts = CPUList(), CPUList()
    return ev


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu')
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    return value


class CPUList(list):
    def append(self, value):
        super().append(to_cpu(value))


def metric_run(root, scope, *, device='cpu', image=False, depth=False, seg=True, map_cpu=False):
    ev = evaluator_cfg(scope, device=device, image=image, depth=depth, seg=seg, map_cpu=map_cpu)
    ev._depth_alignments = []
    fit = ev.fit_scale_and_shift

    def fit_recorded(pred, gt):
        scale, shift = fit(pred, gt)
        ev._depth_alignments.append({'scale': float(scale), 'shift': float(shift), 'valid_pixels': int((gt > 0).sum())})
        return scale, shift

    ev.fit_scale_and_shift = fit_recorded
    result = ev.evaluate(root)
    return ev, result


def copy_window(src, dst):
    def link_or_copy(s, d):
        real = os.path.realpath(s)
        # SIU3R writes render_scores.json and depth_scores.json in the scene
        # directory. Those mutable files must never share an inode with inputs.
        if Path(real).suffix.lower() == '.json':
            return shutil.copy2(real, d)
        try:
            os.link(real, d)
            return d
        except OSError:
            return shutil.copy2(real, d)
    shutil.copytree(src, dst, symlinks=False, copy_function=link_or_copy)


def flatten(obj, prefix=''):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f'{prefix}.{k}' if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f'{prefix}[{i}]'))
    elif isinstance(obj, (int, float, np.number)):
        out[prefix] = float(obj)
    return out


def compare_numeric(ref, got, *, atol=2e-5, rtol=2e-5):
    a, b = flatten(ref), flatten(got)
    if set(a) != set(b):
        raise RuntimeError(f'metric key mismatch: missing={sorted(set(a)-set(b))[:8]} extra={sorted(set(b)-set(a))[:8]}')
    diffs = {}
    for k in a:
        x, y = a[k], b[k]
        if math.isnan(x) and math.isnan(y):
            continue
        if not math.isclose(x, y, abs_tol=atol, rel_tol=rtol):
            raise RuntimeError(f'metric mismatch {k}: reference={x} accelerated={y}')
        diffs[k] = abs(x-y)
    return {'metrics_compared': len(diffs), 'max_abs_delta': max(diffs.values(), default=0.0), 'max_abs_delta_key': max(diffs, key=diffs.get) if diffs else None}


def smoke():
    assert_siu3r_pin()
    src = OLD_EVAL / 'aggregate/full_best/epoch_08/context/scene0011_00_context1727_1744'
    if not src.is_dir():
        raise FileNotFoundError(src)
    root = OUT / 'smoke'; root.mkdir(parents=True, exist_ok=True)
    cpu_root, gpu_root, depth_root = root/'cpu_reference', root/'gpu_accelerated', root/'cpu_depth'
    for dst in (cpu_root, gpu_root, depth_root):
        if dst.exists(): shutil.rmtree(dst)
    copy_window(src, cpu_root/src.name); copy_window(src, gpu_root/src.name); copy_window(src, depth_root/src.name)
    _, reference = metric_run(cpu_root, 'context', image=True, depth=True, seg=True)
    _, accelerated = metric_run(gpu_root, 'context', device='cuda:0', image=True, depth=False, seg=True, map_cpu=True)
    _, depth_only = metric_run(depth_root, 'context', image=False, depth=True, seg=False)
    image_cmp = compare_numeric({k: reference[k] for k in ('psnr','ssim','lpips')}, {k: accelerated[k] for k in ('psnr','ssim','lpips')})
    seg_keys=('context_ious_per_class','context_miou','context_pqs_per_class','context_pq','context_map')
    seg_cmp = compare_numeric({k: reference[k] for k in seg_keys}, {k: accelerated[k] for k in seg_keys})
    # torch.linalg.lstsq is the unchanged official CPU routine; different CPU
    # call context can vary by a few float32 ulps while per-image scores agree.
    depth_cmp = compare_numeric({k: reference[k] for k in ('absrel','rmse')}, {k: depth_only[k] for k in ('absrel','rmse')},atol=2e-5,rtol=0)
    output={'status':'PASS','source_window':src.name,'model_inference':False,'optimizer_updates':0,
      'siu3r_commit':SIU3R_PIN,'device_for_image_and_semantic_metrics':'cuda:0',
      'global_instance_mask_storage':'CPU; single-window temporary segmentation tensors on GPU',
      'depth_alignment':'original SIU3R CPU function; separate official depth-only call',
      'image_metric_comparison':image_cmp,'semantic_panoptic_instance_comparison':seg_cmp,'depth_comparison':depth_cmp,
      'cpu_reference':reference,'gpu_accelerated':accelerated,'cpu_depth_only':depth_only}
    (root/'acceleration_smoke.json').write_text(json.dumps(output,indent=2,allow_nan=True)+'\n')
    print(json.dumps({'status':'PASS','image':image_cmp,'segmentation':seg_cmp,'depth':depth_cmp},indent=2))


def dev8_scenes():
    d=json.loads((REPORT/'manifest.json').read_text())
    return {str(w['scene']) for w in d['monitor_splits']['dev8']}


def symlink(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    resolved=Path(src).resolve()
    if dst.is_symlink() and dst.resolve()==resolved:
        return
    if dst.exists() or dst.is_symlink(): dst.unlink()
    dst.symlink_to(resolved)


def prepare_filtered_seg_scope(scope, excluded):
    source=OLD_EVAL/'aggregate/full_best/epoch_08'/scope
    dest=OUT/'aggregate/frozen_excluding_dev8_best/epoch_08'/scope
    dest.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads((source/'scope_manifest.json').read_text())
    windows={f"{r['scene']}_context{'_'.join(map(str,r['context']))}" for r in source_manifest['frames'] if r['scene'] not in excluded}
    seg='context_seg' if scope=='context' else 'target_seg'
    for wid in sorted(windows):
        src=source/wid
        if not src.is_dir(): raise RuntimeError(f'missing exported window {src}')
        dst=dest/wid;dst.mkdir(parents=True,exist_ok=True)
        for sub in (f'{seg}_pred',f'{seg}_gt'): symlink(src/sub,dst/sub)
    (dest/'scope_manifest.json').write_text(json.dumps({'scope':scope,'windows':sorted(windows),'scene_count':len({x.split('_context')[0] for x in windows})},indent=2)+'\n')
    return dest,len(windows)


def evaluate_segmentation():
    assert_siu3r_pin()
    excluded=dev8_scenes();out={}
    # The completed context official result is immutable and reused as-is.
    ctx=json.loads((OLD_EVAL/'aggregate/full_excluding_dev8_best/epoch_08/context/official_result.json').read_text())
    dst=OUT/'aggregate/frozen_excluding_dev8_best/epoch_08/context';dst.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OLD_EVAL/'aggregate/full_excluding_dev8_best/epoch_08/context/official_result.json',dst/'official_result.json')
    ctx_manifest=json.loads((OLD_EVAL/'aggregate/full_excluding_dev8_best/epoch_08/context/scope_manifest.json').read_text())
    out['context']={'reused_completed_official_result':True,'window_count':len({(r['scene'],tuple(r['context'])) for r in ctx_manifest['frames']}),
                    'scene_count':len({r['scene'] for r in ctx_manifest['frames']}),'result':ctx}
    for scope in ('target_all','true_novel'):
        root=OUT/'aggregate/frozen_excluding_dev8_best/epoch_08'/scope
        saved=root/'official_result.json'; saved_manifest=root/'scope_manifest.json'
        if saved.is_file() and saved_manifest.is_file():
            manifest=json.loads(saved_manifest.read_text())
            saved_windows=set(manifest.get('windows',[]))
            actual_windows={p.name for p in root.iterdir() if p.is_dir()}
            if len(saved_windows)!=1812 or saved_windows!=actual_windows:
                raise RuntimeError(f'incomplete saved segmentation scope {scope}: manifest={len(saved_windows)}, dirs={len(actual_windows)}, expected=1812')
            res=json.loads(saved.read_text())
            out[scope]={'official_result':res,'windows':1812,
                'scenes':len({p.name.split('_context')[0] for p in root.iterdir() if p.is_dir()}),
                'metrics_device':'reused completed global official result','depth/image_metrics_enabled':False,
                'reused_completed_scope':True}
            print('OFFICIAL_SCOPE_REUSED',scope,1812,flush=True)
            continue
        root,count=prepare_filtered_seg_scope(scope,excluded)
        ev,res=metric_run(root,scope,device='cuda:0',image=False,depth=False,seg=True,map_cpu=True)
        result={'official_result':res,'windows':count,'scenes':len({p.name.split('_context')[0] for p in root.iterdir() if p.is_dir()}),
                'metrics_device':'cuda:0 for mIoU/PQ; CPU for packed AP mask aggregation','depth/image_metrics_enabled':False}
        (root/'official_result.json').write_text(json.dumps(res,indent=2,allow_nan=True)+'\n')
        out[scope]=result
        print('OFFICIAL_SCOPE_DONE',scope,count,flush=True)
    (OUT/'frozen_excluding_dev8_official_segmentation.json').write_text(json.dumps(out,indent=2,allow_nan=True)+'\n')


def read_baseline_workers():
    base=OLD_EVAL/'unfrozen_depth/epoch_06/full'; found={}
    for rank in sorted(base.glob('rank[0-9][0-9]')):
        done=json.loads((rank/'worker_done.json').read_text())
        if done.get('status')!='PASS':raise RuntimeError(f'failed depth-only export worker: {rank}')
        for row in json.loads((rank/'window_records.json').read_text()):
            key=(row['scene'],tuple(row['context']))
            if key in found:raise RuntimeError(f'duplicate baseline depth export window {key}')
            found[key]=(rank,row)
    if len(found)!=1860:raise RuntimeError(f'baseline depth exports cover {len(found)} windows, expected 1860')
    return found


def evaluate_baseline_depth():
    assert_siu3r_pin()
    found=read_baseline_workers();result={'epoch':6,'window_count':len(found),'scopes':{}}
    for scope in SCOPES:
        root=OUT/'aggregate/unfrozen_full_epoch06'/scope
        root.mkdir(parents=True,exist_ok=True)
        frames=[]
        for (scene,ctx),(rank,row) in found.items():
            wid=f"{scene}_context{'_'.join(map(str,ctx))}";src=rank/'all'/wid
            ids=row['context'] if scope=='context' else row['target'] if scope=='target_all' else row['novel']
            dst=root/wid;dst.mkdir(exist_ok=True)
            for sub in ('depth','depth_gt'):
                (dst/sub).mkdir(exist_ok=True)
                for frame in ids:
                    name=f'{scene}_{int(frame)}.png';sp=src/sub/name
                    if not sp.is_file():
                        raise RuntimeError(f'missing exported baseline {sub}: window={wid} frame={frame} path={sp}')
                    symlink(sp,dst/sub/name)
            # One frame belongs to both paired directories, but contributes a
            # single per-image metric row.
            frames.extend((scene,ctx,int(frame),wid) for frame in ids)
        # The failed 58638 pass completed official context evaluation and wrote
        # every window's depth_scores.json/results.json. Reuse those exact
        # official scores; calculate only the missing alignment metadata with
        # the pinned evaluator's original CPU least-squares function.
        reuse_context=(scope=='context' and (root/'results.json').is_file() and
            len(list(root.glob('*/depth_scores.json')))==1860)
        if reuse_context:
            ev=evaluator_cfg(scope,device='cpu',image=False,depth=False,seg=False)
            res=json.loads((root/'results.json').read_text())
            if 'absrel' not in res or 'rmse' not in res:
                raise RuntimeError('saved completed baseline context result lacks official depth metrics')
            alignments=[]
            with torch.no_grad():
                for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                    for item in sorted((scene_dir/'depth').glob('*.png')):
                        pred=ev.load_image_to_tensor(item,normalize=False)/1000.0
                        gt=ev.load_image_to_tensor(scene_dir/'depth_gt'/item.name,normalize=False)/1000.0
                        scale,shift=ev.fit_scale_and_shift(pred,gt)
                        alignments.append({'scale':float(scale),'shift':float(shift),'valid_pixels':int((gt>0).sum())})
        else:
            ev,res=metric_run(root,scope,image=False,depth=True,seg=False)
            alignments=ev._depth_alignments
        scores=[]
        aligns=iter(alignments)
        for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for score in json.loads((scene_dir/'depth_scores.json').read_text()):
                item=score['item'];a=next(aligns)
                scene,frame=item.rsplit('_',1);frame=int(frame.replace('.png',''))
                row=dict(model='unfrozen_epoch06',epoch=6,scope=scope,window=scene_dir.name,frame=item.removesuffix('.png'),
                    absrel=float(score['absrel']),rmse=float(score['rmse']),scale=a['scale'],shift=a['shift'],valid_pixels=a['valid_pixels'],
                    pred_png=str(scene_dir/'depth'/item),gt_png=str(scene_dir/'depth_gt'/item))
                scores.append(row)
        try:next(aligns);raise RuntimeError(f'extra alignment rows in baseline depth {scope}')
        except StopIteration:pass
        if len(scores)!=len(frames) or len({(r['window'],r['frame']) for r in scores})!=len(scores):
            raise RuntimeError(f'baseline per-image depth coverage mismatch for {scope}: rows={len(scores)} expected={len(frames)}')
        if any(int(r['valid_pixels'])<1 or not all(math.isfinite(float(r[k])) for k in ('absrel','rmse','scale','shift')) for r in scores):
            raise RuntimeError(f'nonfinite or invalid-GT baseline depth row in {scope}')
        for metric,key in (('absrel','absrel'),('rmse','rmse')):
            mean=sum(r[metric] for r in scores)/len(scores)
            if not math.isclose(mean,float(res[key]),rel_tol=0,abs_tol=1e-12):raise RuntimeError(f'official per-image mean mismatch {scope}/{metric}')
        result['scopes'][scope]={'AbsRel':float(res['absrel']),'RMSE_m':float(res['rmse']),'frame_count':len(scores),'depth_complete':True}
        with (OUT/f'depth_per_image_unfrozen_epoch06_{scope}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
        (root/'official_result.json').write_text(json.dumps(res,indent=2,allow_nan=True)+'\n')
        print('BASELINE_DEPTH_SCOPE_DONE',scope,len(scores),flush=True)
    (OUT/'unfrozen_epoch06_depth_results.json').write_text(json.dumps(result,indent=2,allow_nan=True)+'\n')
    # Excluding dev8 is a strict row filter and per-image arithmetic mean, not another evaluator pass.
    excluded=dev8_scenes();all_rows=[]
    for scope in SCOPES:
        with (OUT/f'depth_per_image_unfrozen_epoch06_{scope}.csv').open(newline='') as f: rows=list(csv.DictReader(f))
        keep=[r for r in rows if r['window'].rsplit('_context',1)[0] not in excluded]
        if not keep:raise RuntimeError(f'empty baseline depth after dev8 exclusion for {scope}')
        all_rows.extend(keep)
        result['scopes'][scope]['excluding_dev8']={'AbsRel':sum(float(r['absrel']) for r in keep)/len(keep),
          'RMSE_m':sum(float(r['rmse']) for r in keep)/len(keep),'frame_count':len(keep)}
    with (OUT/'depth_per_image_unfrozen_epoch06_excluding_dev8_scenes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    (OUT/'unfrozen_epoch06_depth_results.json').write_text(json.dumps(result,indent=2,allow_nan=True)+'\n')


def remaining():
    smoke_path=OUT/'smoke/acceleration_smoke.json'
    if not smoke_path.is_file() or json.loads(smoke_path.read_text()).get('status')!='PASS':
        raise RuntimeError('GPU single-window equivalence smoke is absent or failed')
    evaluate_segmentation()
    evaluate_baseline_depth()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('smoke','remaining'),required=True);a=p.parse_args()
    if a.mode=='smoke':smoke()
    else:remaining()
