"""Render comparison tables, provenance report, and size-limited delivery ZIP."""
from __future__ import annotations
import csv,hashlib,json,zipfile
from pathlib import Path

R=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu')
E=R/'evaluation'
B=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
METRICS=['AbsRel','RMSE_m','PSNR','SSIM','LPIPS','official_mIoU_s','official_PQ','official_mAP','official_AP50']
SCOPES=['context','target_all','true_novel']
PAPER={'reconstruction':dict(AbsRel=.07421,RMSE_m=.2081,PSNR=25.96,SSIM=.8220,LPIPS=.1841),
 'context':dict(official_mIoU_s=.5922,official_mAP=.2817,official_PQ=.6612,official_AP50=None,mIoU_t=.5273),
 'true_novel':dict(official_mIoU_s=.5920,official_mAP=.2714,official_PQ=.6495,official_AP50=None,mIoU_t=.5270)}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write_csv(path,rows,fields):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def load_baseline():
 rows=list(csv.DictReader((B/'full_validation_metrics.csv').open()))
 return {(r['cohort'],r['scope']):r for r in rows if r['epoch']=='6' and r['split']=='full_validation' and r['status']=='COMPLETE'}
def normalized_unfrozen(depth,base_row):
 return dict(AbsRel=depth.get('AbsRel'),RMSE_m=depth.get('RMSE_m'),
  PSNR=float(base_row['psnr']),SSIM=float(base_row['ssim']),LPIPS=float(base_row['lpips']),
  official_mIoU_s=float(base_row['official_miou']),official_PQ=float(base_row['official_pq']),
  official_mAP=float(base_row['official_map']),official_AP50=float(base_row['official_ap50']))
def baseline_scope(rows,depth_obj,cohort,scope):
 oldscope={'context':'context','target_all':'target_all','true_novel':'novel'}[scope]
 return normalized_unfrozen(depth_obj['scopes'][scope],rows[(cohort,oldscope)])
def fmt(x):
 return '—' if x is None else f'{x:.6f}' if isinstance(x,(float,int)) else str(x)
def run():
 train=json.loads((R/'run_manifest.json').read_text())
 complete=json.loads((R/'training_complete.json').read_text())
 chosen=json.loads((E/'dev8_selection.json').read_text())
 frozen=json.loads((E/'frozen_best_results.json').read_text())
 baseline_depth=json.loads((E/'unfrozen_epoch06_depth_results.json').read_text())
 baseline=load_baseline();epoch=int(chosen['selected_epoch']);summaries={};table_rows=[]
 for cohort,entry in (('full','full'),('excluding_dev8_scenes','full_excluding_dev8_scenes')):
  fr=frozen[entry];scopes=[]
  for scope in SCOPES:
   un=baseline_scope(baseline,baseline_depth,cohort,scope);frozenvals={k:fr['scopes'][scope].get(k) for k in METRICS}
   row={'scope':scope}
   for metric in METRICS:
    u=un.get(metric);v=frozenvals.get(metric)
    delta=(v-u) if isinstance(u,(int,float)) and isinstance(v,(int,float)) else None
    row.update({f'unfrozen_epoch6_{metric}':u,f'frozen_best_{metric}':v,f'delta_{metric}':delta})
   scopes.append(row);table_rows.append(dict(cohort=cohort,**row))
  summaries[cohort]=dict(windows=fr['window_count'],scenes=fr['scene_count'],depth_complete=fr['depth_complete'],scopes=scopes)
 fields=['cohort','scope']+[f'{which}_{metric}' for metric in METRICS for which in ('unfrozen_epoch6','frozen_best','delta')]
 write_csv(E/'full_comparison.csv',table_rows,fields)
 (E/'full_comparison.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n')
 paper_rows=[dict(method='SIU3R published Ours',scope='reconstruction aggregate',
  **{k:PAPER['reconstruction'].get(k) for k in METRICS},mIoU_t=None,source='SIU3R arXiv:2507.02705 Table 1')]
 for scope in SCOPES:
  p=PAPER.get('context' if scope=='context' else 'true_novel' if scope=='true_novel' else None,{})
  paper_rows.append(dict(method='SIU3R published Ours',scope=scope,**{k:p.get(k) for k in METRICS},mIoU_t=p.get('mIoU_t'),
    source='SIU3R arXiv:2507.02705 Table 1'))
  paper_rows.append(dict(method='Unfrozen Full1201 epoch6',scope=scope,**baseline_scope(baseline,baseline_depth,'all',scope),
    mIoU_t='NOT_TRAINED',source='this evaluation'))
  paper_rows.append(dict(method=f'Frozen Full1201 epoch{epoch}',scope=scope,
    **{k:frozen['full']['scopes'][scope].get(k) for k in METRICS},mIoU_t='NOT_TRAINED',source='this evaluation'))
 write_csv(E/'paper_task_table.csv',paper_rows,['method','scope']+METRICS+['mIoU_t','source'])

 lines=['# Full1201 frozen MASt3R encoder evaluation','',
  '## Run and selection','',
  f"- Formal training job: {train['job_id']} on {train['node']}; training SHA {train['git_sha']}.",
  f"- Training completed: {complete['updates']} optimizer updates and {complete['exposures']} window exposures.",
  f"- Selected epoch: {epoch}, chosen only by dev8 true-novel official packed AP50 ({chosen['selected_checkpoint_sha256']}).",
  '- Exact AP50 ties select the earlier epoch. Epoch 0 is retained as a start record and excluded from selection.',
  '- The dev8 metrics for epochs 1/2/4/6/8 are in dev8_curve.csv and dev8_selection.json.',
  f"- Frozen full validation: {frozen['full']['window_count']} windows / {frozen['full']['scene_count']} scenes.",
  f"- Excluding all dev8 scenes: {frozen['full_excluding_dev8_scenes']['window_count']} windows / {frozen['full_excluding_dev8_scenes']['scene_count']} scenes.",
  '- Checkpoints were loaded strictly; completed_exposures was restored and passed to the original forward. No optimizer was constructed; optimizer updates were 0.',
  '- Frozen encoder parameter and buffer digest matched the formal initialization digest at every evaluated checkpoint.',
  f"- Best-checkpoint val32 covers {frozen['val32']['window_count']} windows / {frozen['val32']['scene_count']} scenes; its per-scope metrics are in frozen_best_results.json.",
  '',
  '## Complete validation: frozen versus trainable epoch 6','',
  'Depth: SIU3R official per-image scale-and-shift aligned; RMSE unit: m. Packed metrics use the unmodified SIU3R evaluator and official export protocol.',
  '',
  '| Cohort | Scope | Metric | Trainable epoch 6 | Frozen best | Frozen − trainable |','|---|---|---|---:|---:|---:|']
 for cohort in ('full','excluding_dev8_scenes'):
  for row in summaries[cohort]['scopes']:
   for metric in METRICS:
    lines.append(f"| {cohort} | {row['scope']} | {metric} | {fmt(row[f'unfrozen_epoch6_{metric}'])} | {fmt(row[f'frozen_best_{metric}'])} | {fmt(row[f'delta_{metric}'])} |")
 lines+=['','## Instance diagnostics','',
  'Candidate mAP/AP50 is the arithmetic mean of per-window TorchMetrics AP values (diagnostic only); it does not replace official packed mAP/AP50. Class-aware precision/recall, raw best-mask IoU≥0.5 GT fraction, and matched classification accuracy are included in each scope JSON.',
  '',
  '## Published SIU3R reference','',
  'Published values below are transcribed from SIU3R, arXiv:2507.02705, Table 1. AP50 is not reported there. The paper uses unposed images, while this experiment uses GT camera poses.',
  '',
  '| Method | Scope | AbsRel | RMSE (m) | PSNR | SSIM | LPIPS | official mIoUₛ | PQ | mAP | AP50 | mIoUₜ |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for row in paper_rows:
  lines.append('| '+' | '.join([row['method'],row['scope']]+[fmt(row.get(k)) for k in METRICS]+[fmt(row.get('mIoU_t'))])+' |')
 lines+=['','## Protocol notes','',
  '- Scope mapping: context → all/context; target-all → all/target; true novel → novel/target. Local semantic mIoU is not substituted for official mIoUₛ.',
  '- Prediction depth uses the existing renderer depths_pred from RGB+ED at scene scale 0.15 and is converted to meters once. Provider GT depth is in meters; its scene-scale companion is checked against 0.15×GT. SIU3R Visualizer writes the integer-millimeter PNGs.',
  '- SIU3R depth evaluation uses GT PNG depth > 0 only, per-image least-squares scale and shift, and arithmetic mean over images. Per-image CSVs include scale, shift, valid pixels, AbsRel and RMSE. Missing/nonfinite frames and completeness are explicit.',
  '- Our depth render is the original gsplat RGB+ED path; metrics use SIU3R official scale-and-shift. We use GT camera poses, unlike the unposed SIU3R setting.',
  '- Text head is NOT_TRAINED; mIoUₜ is not measured. No text smoke or text training was run.',
  '',
  '## Artifacts','',
  '- frozen_best_results.json, unfrozen_epoch06_depth_results.json, full_comparison.csv/json, and paper_task_table.csv.',
  '- shard_coverage.json records fixed manifest-index shard coverage and duplicate/missing checks.',
  '- qualitative/val32_first2 contains the first two fixed val32 RGB, depth and instance panels.',
  '- Bulk official exports remain under aggregate/ and are excluded from the delivery ZIP.',
  '']
 scopes={x['scope']:x for x in summaries['full']['scopes']}
 understanding=[scopes['true_novel'][f'delta_{m}'] for m in ('official_mIoU_s','official_PQ','official_mAP')]
 improved=all(x is not None and x>0 for x in understanding)
 recon=scopes['true_novel'];affected=any(recon[f'delta_{m}'] is not None and abs(recon[f'delta_{m}'])>.01 for m in ('AbsRel','RMSE_m','PSNR','SSIM','LPIPS'))
 conclusion=(('All three' if improved else 'Not all three')+' true-novel official understanding deltas are positive '
  f'(mIoUₛ {fmt(understanding[0])}, PQ {fmt(understanding[1])}, mAP {fmt(understanding[2])}). '
  f'Reconstruction deltas are AbsRel {fmt(recon["delta_AbsRel"])}, RMSE {fmt(recon["delta_RMSE_m"])} m, '
  f'PSNR {fmt(recon["delta_PSNR"])} dB, SSIM {fmt(recon["delta_SSIM"])}, LPIPS {fmt(recon["delta_LPIPS"])}. '
  f'Use frozen epoch {epoch}, selected by the registered dev8 rule, as the visual checkpoint candidate for a later text-training request; mIoUₜ remains unmeasured.')
 lines.append(conclusion)
 (E/'report.md').write_text('\n'.join(lines)+'\n')
 (E/'eval_provenance.json').write_text(json.dumps(dict(training_sha=train['git_sha'],training_job_id=train['job_id'],
  selected_epoch=epoch,selected_checkpoint_sha256=chosen['selected_checkpoint_sha256'],
  training_manifest_sha256=train['hashes']['manifest.json'],training_plan_sha256=train['hashes']['training_plan.json'],
  validation_manifest_sha256=sha(B/'delivery_epoch6/full_validation_manifest.json'),
  siu3r_commit='8ea80166be76854f938e90521f1a5b688b755c87',optimizer_updates=0),indent=2)+'\n')
 package()
def package():
 out=E/'delivery';out.mkdir(parents=True,exist_ok=True)
 names=['report.md','dev8_curve.csv','dev8_selection.json','frozen_best_results.json','unfrozen_epoch06_depth_results.json',
  'full_comparison.csv','full_comparison.json','paper_task_table.csv','eval_provenance.json','preflight_contracts.json',
  'shard_coverage.json','failures_and_missing.json','single_window_smoke.json']
 files=[E/n for n in names if (E/n).is_file()]+sorted(E.glob('depth_per_image_*.csv'))
 files+=sorted((E/'qualitative/val32_first2').glob('*.png'))
 def make_zip(path,selected):
  with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   for p in selected:z.write(p,p.relative_to(E))
  if path.stat().st_size>=28*1024*1024:raise RuntimeError(f'archive exceeds 28 MiB: {path}')
 try:
  z=out/'full1201_frozen_encoder_results.zip';make_zip(z,files);archives=[z]
 except RuntimeError:
  core=[p for p in files if not p.name.startswith('depth_per_image_') and p.suffix!='.png']
  depth=[p for p in files if p.name.startswith('depth_per_image_')]
  imgs=[p for p in files if p.suffix=='.png']
  archives=[]
  for name,group in [('full1201_frozen_encoder_summary.zip',core),('full1201_frozen_encoder_depth_csv.zip',depth)]:
   z=out/name;make_zip(z,group);archives.append(z)
  if imgs:
   z=out/'full1201_frozen_encoder_qualitative.zip';make_zip(z,imgs);archives.append(z)
 (E/'delivery_manifest.json').write_text(json.dumps({'archives':[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in archives],
  'excluded':['checkpoints','datasets','bulk prediction exports'],'max_bytes_each':28*1024*1024},indent=2)+'\n')
if __name__=='__main__':run()
