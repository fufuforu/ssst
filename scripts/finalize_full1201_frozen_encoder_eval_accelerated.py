"""Produce complete, scope-aligned report and small delivery archives."""
from __future__ import annotations
import csv,hashlib,json,math,shutil,zipfile,subprocess
from pathlib import Path

REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu')
OLD=REPORT/'evaluation'
BASE=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
OUT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu/evaluation_accelerated')
SCOPES=('context','target_all','true_novel')
METRICS=('AbsRel','RMSE_m','PSNR','SSIM','LPIPS','official_mIoU_s','official_PQ','official_mAP','official_AP50')
BASE_COHORT={'full':'all','excluding_dev8_scenes':'excluding_dev8_scenes'}
PAPER={
 'reconstruction':{'AbsRel':.07421,'RMSE_m':.2081,'PSNR':25.96,'SSIM':.8220,'LPIPS':.1841},
 'context':{'official_mIoU_s':.5922,'official_mAP':.2817,'official_PQ':.6612,'official_AP50':None,'mIoU_t':.5273},
 'true_novel':{'official_mIoU_s':.5920,'official_mAP':.2714,'official_PQ':.6495,'official_AP50':None,'mIoU_t':.5270},
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def mean(xs):
 if not xs:raise RuntimeError('cannot average empty metric list')
 if not all(math.isfinite(float(x)) for x in xs):raise RuntimeError('nonfinite per-image metric')
 return sum(map(float,xs))/len(xs)
def write_csv(path,rows,fields):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def read_json(p):return json.loads(Path(p).read_text())
def read_old_metrics():
 p=BASE/'full_validation_metrics.csv'
 with p.open(newline='') as f:rows=list(csv.DictReader(f))
 return {(r['cohort'],r['scope']):r for r in rows if r['epoch']=='6' and r['split']=='full_validation' and r['status']=='COMPLETE'}
def official_values(result,scope):
 branch='context' if scope=='context' else 'target'
 mp=result[f'{branch}_map']
 return {'official_mIoU_s':result[f'{branch}_miou'],'official_PQ':result[f'{branch}_pq'],
         'official_mAP':mp['map'],'official_AP50':mp['map_50']}
def old_values(rows,cohort,scope):
 oldscope={'context':'context','target_all':'target_all','true_novel':'novel'}[scope]
 r=rows[(BASE_COHORT[cohort],oldscope)]
 return {'PSNR':float(r['psnr']),'SSIM':float(r['ssim']),'LPIPS':float(r['lpips']),
  'official_mIoU_s':float(r['official_miou']),'official_PQ':float(r['official_pq']),
  'official_mAP':float(r['official_map']),'official_AP50':float(r['official_ap50'])}
def mean_render_scores(scope,excluded=()):
 root=OLD/'aggregate/full_best/epoch_08'/scope;excluded=set(excluded);vals={k:[] for k in ('PSNR','SSIM','LPIPS')};seen=set()
 for d in sorted(p for p in root.iterdir() if p.is_dir()):
  scene=d.name.rsplit('_context',1)[0]
  if scene in excluded:continue
  rows=read_json(d/'render_scores.json')
  for r in rows:
   key=(d.name,r['item'])
   if key in seen:raise RuntimeError(f'duplicate render score row {key}')
   seen.add(key)
   for metric,src in (('PSNR','psnr'),('SSIM','ssim'),('LPIPS','lpips')):vals[metric].append(float(r[src]))
 return {k:mean(v) for k,v in vals.items()},len(seen)
def depth_rows(path,scope,excluded=()):
 excluded=set(excluded)
 with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
 selected=[r for r in rows if r['scope']==scope and r['window'].rsplit('_context',1)[0] not in excluded]
 if len({(r['window'],r['frame']) for r in selected})!=len(selected):raise RuntimeError(f'duplicate depth rows {path}/{scope}')
 return selected
def depth_values(rows):
 return {'AbsRel':mean([r['absrel'] for r in rows]),'RMSE_m':mean([r['rmse'] for r in rows])}
def diag_summary(scope,excluded=()):
 source=OLD/'best/epoch_08/full';records={}
 for rank in sorted(source.glob('rank[0-9][0-9]')):
  for r in read_json(rank/'window_records.json'):
   key=(r['scene'],tuple(r['context']))
   if key in records:raise RuntimeError(f'duplicate diagnostics window {key}')
   records[key]=r
 selected=[v for (scene,_),v in records.items() if scene not in set(excluded)]
 name='novel' if scope=='true_novel' else scope;rows=[r['diagnostics'][name] for r in selected]
 def counts(field):
  tp=sum(int(r[field]['tp']) for r in rows);fp=sum(int(r[field]['fp']) for r in rows);fn=sum(int(r[field]['fn']) for r in rows)
  return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn)}
 aps=[r['candidate_ap'] for r in rows if r.get('candidate_ap')]
 raw=[v for r in rows for v in r.get('raw_best_ious',[])]
 den=sum(int(r.get('matched_gt_count',0)) for r in rows)
 return {'windows':len(selected),'candidate_mAP_window_macro':mean([x['map'] for x in aps]),
  'candidate_AP50_window_macro':mean([x['map_50'] for x in aps]),
  'class_aware_precision_recall':counts('candidate_cw'),
  'raw_best_mask_IoU_ge_0_5_GT_fraction':sum(v>=.5 for v in raw)/max(1,len(raw)),
  'matched_class_accuracy':sum(float(r.get('matched_19_class_accuracy',0))*int(r.get('matched_gt_count',0)) for r in rows)/max(1,den),
  'matched_GT_count':den}
def run():
 smoke=read_json(OUT/'smoke/acceleration_smoke.json')
 if smoke.get('status')!='PASS':raise RuntimeError('accelerated single-window validation is not PASS')
 selected=read_json(OLD/'dev8_selection.json');epoch=int(selected['selected_epoch'])
 if epoch!=8:raise RuntimeError(f'registered dev8 selection changed unexpectedly: {epoch}')
 baseline_csv=read_old_metrics();dev=json.loads((REPORT/'manifest.json').read_text())
 excluded={str(w['scene']) for w in dev['monitor_splits']['dev8']}
 train=read_json(REPORT/'run_manifest.json');complete=read_json(REPORT/'training_complete.json')
 baseline_depth=read_json(OUT/'unfrozen_epoch06_depth_results.json')
 frozen_full={};frozen_excl={};baseline_full={};baseline_excl={};diag={};frozen_excluded_depth_rows=[]
 frozen_depth_csv=OLD/'depth_per_image_full_best.csv'
 baseline_depth_csv={s:OUT/f'depth_per_image_unfrozen_epoch06_{s}.csv' for s in SCOPES}
 for scope in SCOPES:
  frozen_result=read_json(OLD/f'aggregate/full_best/epoch_08/{scope}/official_result.json')
  frozen_img,full_img_count=mean_render_scores(scope)
  frozen_depth=depth_values(depth_rows(frozen_depth_csv,scope))
  frozen_full[scope]={**frozen_depth,**frozen_img,**official_values(frozen_result,scope)}
  old=old_values(baseline_csv,'full',scope)
  baseline_full[scope]={**old,**baseline_depth['scopes'][scope]}
  img_excl,img_excl_count=mean_render_scores(scope,excluded)
  scope_excluded_depth_rows=depth_rows(frozen_depth_csv,scope,excluded)
  frozen_excluded_depth_rows.extend(scope_excluded_depth_rows)
  d_excl=depth_values(scope_excluded_depth_rows)
  frozen_seg=read_json(OUT/f'aggregate/frozen_excluding_dev8_best/epoch_08/{scope}/official_result.json')
  frozen_excl[scope]={**d_excl,**img_excl,**official_values(frozen_seg,scope)}
  old_ex=old_values(baseline_csv,'excluding_dev8_scenes',scope)
  dbr=depth_rows(baseline_depth_csv[scope],scope,excluded)
  baseline_excl[scope]={**old_ex,**depth_values(dbr)}
  diag[scope]={'full':diag_summary(scope),'excluding_dev8_scenes':diag_summary(scope,excluded)}
  # Existing full image score/depth data must reconcile to the official full result.
  for k,raw in (('PSNR',frozen_result['psnr']),('SSIM',frozen_result['ssim']),('LPIPS',frozen_result['lpips']),
                ('AbsRel',frozen_result['absrel']),('RMSE_m',frozen_result['rmse'])):
   if not math.isclose(float(frozen_full[scope][k]),float(raw),rel_tol=0,abs_tol=1e-5):raise RuntimeError(f'full per-image reconciliation failed {scope}/{k}')
  print(f'AGGREGATED {scope}: full_frames={full_img_count}, excluding_dev8_frames={img_excl_count}',flush=True)
 comparison=[]
 for cohort,fr,un in (('full',frozen_full,baseline_full),('excluding_dev8_scenes',frozen_excl,baseline_excl)):
  for scope in SCOPES:
   row={'cohort':cohort,'scope':scope}
   for m in METRICS:
    a=float(un[scope][m]);b=float(fr[scope][m]);row[f'unfrozen_epoch6_{m}']=a;row[f'frozen_best_{m}']=b;row[f'delta_{m}']=b-a
   comparison.append(row)
 write_csv(OUT/'frozen_vs_unfrozen_full_and_excluded.csv',comparison,['cohort','scope']+[f'{p}_{m}' for m in METRICS for p in ('unfrozen_epoch6','frozen_best','delta')])
 full_json={'full':{'frozen':frozen_full,'unfrozen_epoch6':baseline_full},'excluding_dev8_scenes':{'frozen':frozen_excl,'unfrozen_epoch6':baseline_excl},'diagnostics':diag}
 (OUT/'frozen_vs_unfrozen_full_and_excluded.json').write_text(json.dumps(full_json,indent=2,allow_nan=False)+'\n')
 val32={}
 for scope in SCOPES:
  result=read_json(OLD/f'aggregate/val32_best/epoch_08/{scope}/official_result.json')
  image,n=mean_render_scores(scope)
  depths=depth_values(depth_rows(OLD/'depth_per_image_val32_best.csv',scope))
  val32[scope]={**depths,**image,**official_values(result,scope),'frame_count':n}
 (OUT/'val32_reused_results.json').write_text(json.dumps(val32,indent=2,allow_nan=False)+'\n')
 shutil.copy2(OLD/'dev8_curve.csv',OUT/'dev8_curve.csv');shutil.copy2(OLD/'dev8_selection.json',OUT/'dev8_selection.json')
 shutil.copy2(OLD/'depth_per_image_full_best.csv',OUT/'depth_per_image_full_best.csv')
 shutil.copy2(OLD/'depth_per_image_val32_best.csv',OUT/'depth_per_image_val32_best.csv')
 write_csv(OUT/'depth_per_image_frozen_best_excluding_dev8_scenes.csv',frozen_excluded_depth_rows,list(frozen_excluded_depth_rows[0]))
 qual_src=OLD/'qualitative/val32_first2';qual_dst=OUT/'qualitative/val32_first2';qual_dst.mkdir(parents=True,exist_ok=True)
 if len(list(qual_src.glob('*.png')))!=2:raise RuntimeError('expected the two registered val32 qualitative panels')
 for p in qual_src.glob('*.png'):shutil.copy2(p,qual_dst/p.name)
 jobs=read_json(OUT/'accelerated_jobs.json')
 repo=Path(__file__).resolve().parents[1]
 code_sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 prov={'training_sha':train['git_sha'],'training_job_id':train['job_id'],'completed_updates':complete['updates'],'completed_exposures':complete['exposures'],
   'selected_epoch':epoch,'selected_checkpoint_sha256':selected['selected_checkpoint_sha256'],'training_manifest_sha256':train['hashes']['manifest.json'],
   'training_plan_sha256':train['hashes']['training_plan.json'],'siu3r_commit':'8ea80166be76854f938e90521f1a5b688b755c87',
   'optimizer_updates':0,'old_cancelled_job':'58537','recovery_manifest_sha256':sha(OUT/'recovery_manifest.json'),
   'accelerator_smoke_sha256':sha(OUT/'smoke/acceleration_smoke.json'),'evaluation_code_git_sha':code_sha,
   'accelerator_script_sha256':sha(repo/'scripts/accelerate_full1201_frozen_encoder_eval.py'),
   'finalizer_script_sha256':sha(Path(__file__)),'evaluation_jobs':jobs.get('jobs',{})}
 (OUT/'eval_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
 paper=[]
 paper.append({'method':'SIU3R published Ours','scope':'reconstruction aggregate',**PAPER['reconstruction'],'mIoU_t':None,'source':'SIU3R arXiv:2507.02705 Table 1'})
 for scope in SCOPES:
  paper_scope='context' if scope=='context' else 'true_novel' if scope=='true_novel' else None
  pvals=PAPER.get(paper_scope,{})
  paper.append({'method':'SIU3R published Ours','scope':scope,**{m:pvals.get(m) for m in METRICS},'mIoU_t':pvals.get('mIoU_t'),'source':'SIU3R arXiv:2507.02705 Table 1'})
  paper.append({'method':'Unfrozen Full1201 epoch6','scope':scope,**baseline_full[scope],'mIoU_t':'NOT_TRAINED','source':'completed control full_validation_metrics.csv + this depth-only completion'})
  paper.append({'method':f'Frozen Full1201 epoch{epoch}','scope':scope,**frozen_full[scope],'mIoU_t':'NOT_TRAINED','source':'this evaluation'})
 write_csv(OUT/'paper_task_table.csv',paper,['method','scope']+list(METRICS)+['mIoU_t','source'])
 depth_completeness={'frozen_full':{},'frozen_excluding_dev8_scenes':{},'unfrozen_full':{},'unfrozen_excluding_dev8_scenes':{}}
 for scope in SCOPES:
  depth_completeness['frozen_full'][scope]={'complete':True,'images':len(depth_rows(frozen_depth_csv,scope))}
  depth_completeness['frozen_excluding_dev8_scenes'][scope]={'complete':True,'images':len(depth_rows(frozen_depth_csv,scope,excluded))}
  depth_completeness['unfrozen_full'][scope]={'complete':bool(baseline_depth['scopes'][scope]['depth_complete']),'images':baseline_depth['scopes'][scope]['frame_count']}
  depth_completeness['unfrozen_excluding_dev8_scenes'][scope]={'complete':True,'images':len(depth_rows(baseline_depth_csv[scope],scope,excluded))}
 (OUT/'depth_completeness.json').write_text(json.dumps(depth_completeness,indent=2)+'\n')
 lines=['# Full1201 frozen MASt3R encoder evaluation (accelerated resume)','',
  '## Provenance and selection','',f"- Training job {train['job_id']}; training SHA {train['git_sha']}; updates {complete['updates']}; exposures {complete['exposures']}.",
  f"- Registered dev8 selection: epoch {epoch}, true-novel packed official AP50 {selected['curve'][-1]['true_novel_official_AP50'] if selected['curve'][-1]['epoch']==epoch else next(x['true_novel_official_AP50'] for x in selected['curve'] if x['epoch']==epoch)}; checkpoint SHA256 {selected['selected_checkpoint_sha256']}.",
  '- The 58537 aggregate job was stopped after completed results and exports were independently indexed and checked. The unfinished 465-item target-all pass was discarded; no inference was restarted.',
  '- No checkpoint was loaded and no model inference or optimizer update occurred during accelerated evaluation.',
  '', '## Frozen versus unfrozen Full1201 epoch 6','',
  'AbsRel/RMSE use SIU3R official per-image scale-and-shift alignment; RMSE unit: m. Official semantic/panoptic/instance metrics use packed SIU3R evaluator outputs. Image metrics for frozen full/excluded cohorts are arithmetic means of existing per-image render scores. Baseline image metrics reuse the already completed control CSV.',
  '', '| Cohort | Scope | Metric | Unfrozen epoch 6 | Frozen epoch 8 | Frozen − unfrozen |','|---|---|---|---:|---:|---:|']
 for row in comparison:
  for m in METRICS:lines.append(f"| {row['cohort']} | {row['scope']} | {m} | {row[f'unfrozen_epoch6_{m}']:.6f} | {row[f'frozen_best_{m}']:.6f} | {row[f'delta_{m}']:.6f} |")
 lines+=['','## Local instance diagnostics','',
  'Candidate mAP/AP50, class-aware precision/recall, raw best-mask IoU≥0.5 GT fraction, and matched classification accuracy are reported in frozen_vs_unfrozen_full_and_excluded.json as diagnostics only.',
  '', '## Reference and protocol notes','',
  'Published SIU3R values are transcribed from arXiv:2507.02705 Table 1; fields not reported by that source remain missing. SIU3R uses unposed input images; our experiment uses GT camera poses.',
  'Depth predictions are the original gsplat RGB+ED output at scene scale 0.15; depth metrics use the SIU3R official per-image scale-and-shift protocol and CPU function. Image metrics use the original SIU3R per-image definitions. Packed scope map: context→all/context, target-all→all/target, true novel→novel/target.',
  'Text task is NOT_TRAINED; mIoUₜ is not measured. No follow-on checkpoint is automatically selected.',
  '', '## Evaluation acceleration validation','',
  f"Single-window CPU/GPU equivalence passed: image max absolute delta {smoke['image_metric_comparison']['max_abs_delta']:.8g}; semantic/panoptic/instance max absolute delta {smoke['semantic_panoptic_instance_comparison']['max_abs_delta']:.8g}; official CPU depth delta {smoke['depth_comparison']['max_abs_delta']:.8g}.",
  'GPU was used for image metrics and the semantic/PQ states. Packed AP masks were retained and accumulated on CPU; SIU3R depth alignment remained on CPU. Existing full exports were reused; no model inference was run.',
  '', '## Artifacts','',
  '- frozen_vs_unfrozen_full_and_excluded.csv/json contains both cohorts, all scopes and all required metrics; paper_task_table.csv contains the sourced paper comparison.',
  '- dev8_curve.csv and dev8_selection.json record checkpoint selection; val32_reused_results.json records the reused fixed val32 results.',
  '- depth_per_image_full_best.csv, frozen and unfrozen per-image exclusion CSVs, per-scope unfrozen depth CSVs, and depth_per_image_unfrozen_epoch06_excluding_dev8_scenes.csv contain per-image scores, scale, shift and valid pixel counts.',
  '- recovery_manifest.json identifies verified completed exports and SHA256 records; qualitative/val32_first2 contains both registered panels.',
  '']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 package()

def package():
 delivery=OUT/'delivery';delivery.mkdir(parents=True,exist_ok=True)
 core=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.csv')]+[OUT/'report.md']
 core=[p for p in core if p.is_file() and not p.name.startswith('depth_per_image_')]
 depth=[p for p in OUT.glob('depth_per_image_*.csv') if p.is_file()]
 qual=sorted((OUT/'qualitative/val32_first2').glob('*.png'))
 groups=[('full1201_frozen_encoder_summary.zip',core),('full1201_frozen_encoder_depth.zip',depth),('full1201_frozen_encoder_qualitative.zip',qual)]
 records=[]
 for name,files in groups:
  if not files:continue
  path=delivery/name
  with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   for p in files:z.write(p,p.relative_to(OUT))
  if path.stat().st_size>=28*1024*1024:raise RuntimeError(f'archive exceeds 28 MiB: {path}')
  records.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'files':[p.name for p in files]})
 (OUT/'delivery_manifest.json').write_text(json.dumps({'archives':records,'max_bytes_each':28*1024*1024,'checkpoints_datasets_and_bulk_exports_excluded':True},indent=2)+'\n')
if __name__=='__main__':run()
