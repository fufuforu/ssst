"""Official global aggregation for read-only Full1201 eval exports."""
from __future__ import annotations
import argparse,csv,hashlib,json,subprocess,sys,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu')
BASE_REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
SIU3R=Path('/space/mawb/SIU3R'); PIN='8ea80166be76854f938e90521f1a5b688b755c87'
EVAL=REPORT/'evaluation';sys.path.insert(0,str(SIU3R))
from src.config import EvaluatorCfg
from src.evaluator import Evaluator
from src.utils.scannet_constant import PANOPTIC_SEMANTIC2NAME,STUFF_CLASSES,THING_CLASSES
COVERAGE=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def windows(split):
 m=json.loads((REPORT/'manifest.json').read_text())
 if split in ('dev8','val32'):
  return [dict(scene=w['scene'],context=list(map(int,w['context'])),target=list(map(int,w['target'])),novel=list(map(int,w['novel']))) for w in m['monitor_splits'][split]]
 p=BASE_REPORT/'delivery_epoch6/full_validation_manifest.json'
 if not p.exists():p=BASE_REPORT/'full_validation_manifest.json'
 return [dict(scene=w['scan'],context=list(map(int,w['context_ids'])),target=list(map(int,w['target_ids'])),novel=[int(x) for x in w['target_ids'] if x not in w['context_ids']]) for w in json.loads(p.read_text())]
def key(w):return str(w['scene']),tuple(map(int,w['context']))
def pair(w):return f"{w['scene']}_context{'_'.join(map(str,w['context']))}"
def source_records(base,epoch,split):
 found={}
 roots=sorted((Path(base)/f'epoch_{epoch:02}'/split).glob('rank[0-9][0-9]'))
 for root in roots:
  if not (root/'worker_done.json').exists():continue
  d=json.loads((root/'worker_done.json').read_text())
  if d['status']!='PASS':raise RuntimeError(f'failed worker {root}')
  for row in json.loads((root/'window_records.json').read_text()):
   k=(row['scene'],tuple(row['context']))
   if k in found:raise RuntimeError(f'duplicate shard output {k}')
   found[k]=(root,row)
 return found
def collect(base,epoch,split,expected):
 found=source_records(base,epoch,split);want={key(w) for w in expected}
 if set(found)!=want:raise RuntimeError(f'shard coverage mismatch missing={len(want-set(found))} extra={len(set(found)-want)}')
 rank_rows=[]
 for root in sorted((Path(base)/f'epoch_{epoch:02}'/split).glob('rank[0-9][0-9]')):
  d=json.loads((root/'worker_done.json').read_text())
  rows=json.loads((root/'window_records.json').read_text())
  rank_rows.append(dict(rank=d['rank'],job_id=d.get('job_id'),windows=d['windows'],
    completed_exposures=d['completed_exposures'],encoder_digest=d.get('frozen_encoder_digest'),
    window_ids=[f"{x['scene']}_context{'_'.join(map(str,x['context']))}" for x in rows]))
 COVERAGE.append(dict(worker_root=str(Path(base).resolve()),epoch=epoch,split=split,
   expected_windows=len(want),observed_windows=len(found),unique_window_ids=len({k for k in found}),
   ranks=rank_rows,complete=True))
 return found
def save_coverage():
 p=EVAL/'shard_coverage.json';old=json.loads(p.read_text()) if p.exists() else []
 keys={(x['worker_root'],x['epoch'],x['split']) for x in old}
 old.extend(x for x in COVERAGE if (x['worker_root'],x['epoch'],x['split']) not in keys)
 p.write_text(json.dumps(old,indent=2)+'\n')
def symlink(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.is_symlink() or dst.exists():dst.unlink()
 dst.symlink_to(src)
def link_files(src,dst,names):
 dst.mkdir(parents=True,exist_ok=True)
 for n in names:
  p=src/n
  if p.is_file():symlink(p,dst/n)
def make_root(out,items,records,scope,seg=True):
 out=Path(out)
 if out.exists():shutil.rmtree(out)
 out.mkdir(parents=True,exist_ok=True);fr=[];seen=set()
 for w in items:
  k=key(w);rankroot,row=records[k];name=pair(w);d=out/name;d.mkdir(parents=True,exist_ok=True)
  raw=rankroot/'all'/name;nov=rankroot/'novel'/name
  if scope=='context':
   if seg:
    for sub in ('context_seg_pred','context_seg_gt'):symlink(raw/sub,d/sub)
   ids=w['context']
  elif scope=='target_all':
   if seg:
    for sub in ('target_seg_pred','target_seg_gt'):symlink(raw/sub,d/sub)
   ids=w['target']
  else:
   if seg:
    for sub in ('target_seg_pred','target_seg_gt'):symlink(nov/sub,d/sub)
   ids=w['novel']
  files=[f'{w["scene"]}_{i}.png' for i in ids]
  for sub in ('rgb','rgb_gt','depth','depth_gt'):link_files(raw/sub,d/sub,files)
  for i in ids:
   ident=(k,int(i))
   if ident in seen:raise RuntimeError(f'duplicate frame {ident}')
   seen.add(ident)
   fr.append(dict(scene=w['scene'],context=list(w['context']),frame=int(i),scope=scope,
    rgb=(d/'rgb'/f'{w["scene"]}_{i}.png').is_file(),depth=(d/'depth'/f'{w["scene"]}_{i}.png').is_file(),gt=(d/'depth_gt'/f'{w["scene"]}_{i}.png').is_file()))
 (out/'scope_manifest.json').write_text(json.dumps(dict(scope=scope,frames=fr),indent=2)+'\n')
 return fr
def official(root,scope,seg=True,image_quality=True):
 if subprocess.check_output(['git','-C',str(SIU3R),'rev-parse','HEAD'],text=True).strip()!=PIN:raise RuntimeError('SIU3R SHA mismatch')
 ctx=scope=='context'
 cfg=EvaluatorCfg(dataset_name='scannet',eval_context_miou=seg and ctx,eval_context_pq=seg and ctx,eval_context_map=seg and ctx,
  eval_target_miou=seg and not ctx,eval_target_pq=seg and not ctx,eval_target_map=seg and not ctx,
  eval_image_quality=image_quality,eval_depth_quality=True,id2label=PANOPTIC_SEMANTIC2NAME,stuffs=STUFF_CLASSES,things=THING_CLASSES,device='cpu',eval_path=str(root))
 ev=Evaluator(cfg);ev.setup();ev._depth_alignments=[]
 original=ev.fit_scale_and_shift
 def recorded_fit(pred,gt):
  scale,shift=original(pred,gt)
  ev._depth_alignments.append(dict(scale=float(scale),shift=float(shift),valid_pixels=int((gt>0).sum())))
  return scale,shift
 ev.fit_scale_and_shift=recorded_fit
 return ev,ev.evaluate(root)
def depth_rows(ev,root,scope,model_name,epoch):
 out=[];files=[]
 for scene in sorted(p for p in Path(root).iterdir() if p.is_dir()):
  files.extend(sorted((scene/'depth').glob('*.png')))
 if len(files)!=len(ev._depth_alignments):raise RuntimeError('SIU3R did not align every exported prediction depth')
 for pred,alignment in zip(files,ev._depth_alignments):
  gt=pred.parent.parent/'depth_gt'/pred.name
  a=np.array(Image.open(pred));b=np.array(Image.open(gt))
  if a.shape!=b.shape:raise RuntimeError(f'depth shape mismatch {pred}')
  g=ev.load_image_to_tensor(gt,normalize=False)/1000.;n=int((g>0).sum())
  if n<1:raise RuntimeError(f'no valid gt in exported depth {gt}')
  if n!=alignment['valid_pixels']:raise RuntimeError(f'official validity pixel count mismatch {gt}')
  saved=json.loads((pred.parent.parent/'depth_scores.json').read_text())
  official_row=next((x for x in saved if x['item']==pred.name),None)
  if official_row is None:raise RuntimeError(f'official per-image depth metric missing: {pred}')
  out.append(dict(model=model_name,epoch=epoch,scope=scope,window=pred.parent.parent.name,frame=pred.stem,
   absrel=float(official_row['absrel']),rmse=float(official_row['rmse']),scale=alignment['scale'],shift=alignment['shift'],
   valid_pixels=n,pred_png=str(pred),gt_png=str(gt)))
 for field,global_name in (('absrel','absrel'),('rmse','rmse')):
  avg=float(np.mean([r[field] for r in out])) if out else float('nan')
  if not np.isclose(avg,float(ev._depth_global_result.get(global_name,float('nan'))),rtol=0,atol=1e-12):
   raise RuntimeError(f'official per-image {field} mean differs from SIU3R aggregate')
 return out
def diagnostics(records,items,scope):
 local_scope='novel' if scope=='true_novel' else scope
 rows=[records[key(w)][1]['diagnostics'][local_scope] for w in items]
 def counts(k):
  z={f:sum(int(r[k][f]) for r in rows) for f in ('tp','fp','fn')}
  z['precision']=z['tp']/max(1,z['tp']+z['fp']);z['recall']=z['tp']/max(1,z['tp']+z['fn']);return z
 ca=counts('candidate_ca');cw=counts('candidate_cw')
 raw=[v for r in rows for v in r.get('raw_best_ious',[])]
 n=sum(int(r.get('matched_gt_count',0)) for r in rows)
 def macro(field):
  x=[float(r['candidate_ap'][field]) for r in rows if isinstance(r.get('candidate_ap',{}).get(field),(int,float)) and np.isfinite(r['candidate_ap'][field])]
  return float(np.mean(x)) if x else None
 return dict(candidate_mAP_window_macro=macro('map'),candidate_AP50_window_macro=macro('map_50'),
  candidate_ap_aggregation='window-macro torchmetrics AP; diagnostic only',class_aware_precision=cw['precision'],class_aware_recall=cw['recall'],
  raw_best_mask_iou_ge_0_5_gt_fraction=sum(v>=.5 for v in raw)/max(1,len(raw)),
  matched_class_accuracy=sum(float(r.get('matched_19_class_accuracy',0))*int(r.get('matched_gt_count',0)) for r in rows)/max(1,n),
  candidate_counts=ca,class_aware_counts=cw,gt_count=len(raw),matched_gt_count=n)
def evaluate(name,epoch,items,records,seg=True,image_quality=True,excluded=()):
 chosen=[w for w in items if w['scene'] not in set(excluded)]
 result=dict(name=name,epoch=epoch,window_count=len(chosen),scene_count=len({w['scene'] for w in chosen}),scopes={},failed_depth_frames=[])
 per_image=[]
 for scope in ('context','target_all','true_novel'):
  root=EVAL/'aggregate'/name/f'epoch_{epoch:02}'/scope
  frames=make_root(root,chosen,records,scope,seg);ev,m=official(root,scope,seg,image_quality)
  ev._depth_global_result=m
  detail=depth_rows(ev,root,scope,name,epoch);per_image.extend(detail)
  missing=[x for x in frames if not (x['depth'] and x['gt'])]
  result['failed_depth_frames'].extend(missing)
  branch='context' if scope=='context' else 'target'
  amap=m.get(branch+'_map') or {}
  sc=dict(AbsRel=m.get('absrel'),RMSE_m=m.get('rmse'),PSNR=m.get('psnr'),SSIM=m.get('ssim'),LPIPS=m.get('lpips'),
    official_mIoU_s=m.get(branch+'_miou'),official_PQ=m.get(branch+'_pq'),official_mAP=amap.get('map'),official_AP50=amap.get('map_50'),
    frame_count=len(frames),depth_frame_count=len(detail),depth_complete=len(detail)==len(frames) and not missing,
    local_instance_diagnostics=diagnostics(records,chosen,scope) if seg else None)
  result['scopes'][scope]=sc
  (root/'official_result.json').write_text(json.dumps(m,indent=2,default=str)+'\n')
 with (EVAL/f'depth_per_image_{name}.csv').open('w',newline='') as f:
  fields=list(per_image[0]) if per_image else ['model','epoch','scope','window','frame','absrel','rmse','scale','shift','valid_pixels']
  wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(per_image)
 result['depth_complete']=not result['failed_depth_frames'] and all(x['depth_complete'] for x in result['scopes'].values())
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','dev8','best','baseline_depth'],required=True)
 ap.add_argument('--epoch',type=int);ap.add_argument('--worker-root',type=Path,required=True);ap.add_argument('--dev8-root',type=Path);a=ap.parse_args()
 EVAL.mkdir(parents=True,exist_ok=True)
 if a.mode=='smoke':
  ws=windows('dev8')[:1];rec=collect(a.worker_root,a.epoch or 1,'dev8',ws)
  out=evaluate('single_window_smoke',a.epoch or 1,ws,rec)
  (EVAL/'single_window_smoke.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
  save_coverage()
  print(json.dumps({'windows':out['window_count'],'depth_complete':out['depth_complete'],'scopes':list(out['scopes'])},indent=2));return
 if a.mode=='dev8':
  curve=[]
  for e in (1,2,4,6,8):
   ws=windows('dev8');rec=collect(a.worker_root,e,'dev8',ws);r=evaluate(f'dev8_epoch_{e:02}',e,ws,rec)
   ck=Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu')/f'checkpoint_epoch_{e:02}.pt'
   curve.append(dict(epoch=e,updates=e*1043,exposures=e*8344,true_novel_official_AP50=r['scopes']['true_novel']['official_AP50'],checkpoint_sha256=sha(ck),scopes=r['scopes']))
  best=max(curve,key=lambda r:(-float('inf') if r['true_novel_official_AP50'] is None else r['true_novel_official_AP50'],-r['epoch']))
  epoch0=a.worker_root/'epoch_00_verification.json'
  if not epoch0.exists():raise RuntimeError('epoch0 strict-load/digest verification missing')
  out=dict(epoch0_start=json.loads(epoch0.read_text()),selection='highest dev8 true-novel official packed AP50; exact ties choose earlier epoch',selected_epoch=best['epoch'],selected_checkpoint_sha256=best['checkpoint_sha256'],curve=curve)
  (EVAL/'dev8_selection.json').write_text(json.dumps(out,indent=2)+'\n')
  save_coverage()
  with (EVAL/'dev8_curve.csv').open('w',newline='') as f:
   fields=['epoch','updates','exposures','scope','AbsRel','RMSE_m','PSNR','SSIM','LPIPS','official_mIoU_s','official_PQ','official_mAP','official_AP50','checkpoint_sha256']
   wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
   for r in curve:
    for scope,metrics in r['scopes'].items():
     wr.writerow(dict(epoch=r['epoch'],updates=r['updates'],exposures=r['exposures'],scope=scope,
       **{k:metrics.get(k) for k in fields if k not in ('epoch','updates','exposures','scope','checkpoint_sha256')},
       checkpoint_sha256=r['checkpoint_sha256']))
  print(json.dumps({'selected_epoch':best['epoch'],'AP50':best['true_novel_official_AP50']},indent=2));return
 if a.mode=='best':
  e=a.epoch
  if e in (None,0):e=int(json.loads((EVAL/'dev8_selection.json').read_text())['selected_epoch'])
  dev=windows('dev8');val=windows('val32');full=windows('full');devkeys={key(w) for w in dev}
  rd=collect(a.dev8_root or a.worker_root,e,'dev8',dev)
  rv=collect(a.worker_root,e,'val32',[w for w in val if key(w) not in devkeys])
  prior=devkeys|{key(w) for w in val}
  rf=collect(a.worker_root,e,'full',[w for w in full if key(w) not in prior])
  allrec={**rd,**rv,**rf};vc={key(w):allrec[key(w)] for w in val};fc={key(w):allrec[key(w)] for w in full}
  scenes={w['scene'] for w in dev}
  out=dict(best_epoch=e,best_checkpoint_sha256=sha(Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu')/f'checkpoint_epoch_{e:02}.pt'),
    val32=evaluate('val32_best',e,val,vc),full=evaluate('full_best',e,full,fc),
    full_excluding_dev8_scenes=evaluate('full_excluding_dev8_best',e,full,fc,excluded=scenes))
  qsrc=(a.dev8_root or a.worker_root)/'qualitative'/f'epoch_{e:02}'/'val32'
  qdst=EVAL/'qualitative/val32_first2';qdst.mkdir(parents=True,exist_ok=True)
  qfiles=sorted(qsrc.glob('*.png'))
  if len(qfiles)!=2:raise RuntimeError(f'expected two fixed val32 qualitative panels, found {len(qfiles)}')
  for src in qfiles:shutil.copy2(src,qdst/src.name)
  (EVAL/'frozen_best_results.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
  save_coverage()
  print(json.dumps({k:out[k]['window_count'] for k in ('val32','full','full_excluding_dev8_scenes')},indent=2));return
 if a.mode=='baseline_depth':
  ws=windows('full');rec=collect(a.worker_root,6,'full',ws);out=evaluate('full_unfrozen_epoch06_depth',6,ws,rec,seg=False,image_quality=False)
  (EVAL/'unfrozen_epoch06_depth_results.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
  save_coverage()
  print(json.dumps({'windows':out['window_count'],'depth_complete':out['depth_complete']}))
if __name__=='__main__':main()
