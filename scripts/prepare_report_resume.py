#!/usr/bin/env python3
"""Build a report-only read-only file view from a completed cached evaluation."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def copy_file(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--attempt',type=Path,required=True)
 ap.add_argument('--execution-sha',required=True);a=ap.parse_args();src=a.source.resolve();dst=a.attempt.resolve()
 if not (dst/'preflight/report_preflight.json').is_file():raise RuntimeError('report preflight result is required before preparing attempt root')
 pre=json.loads((dst/'preflight/report_preflight.json').read_text())
 if pre.get('status')!='PASS':raise RuntimeError('report preflight is not PASS')
 allowed_dirs={'slurm','predictions','official','labels','heads','cache','r3d','reconstruction_cache','features','reference_gpu','provenance','preflight'}
 unexpected=[p.name for p in dst.iterdir() if p.name not in allowed_dirs]
 if unexpected:raise RuntimeError(f'refusing to overlay unexpected attempt-root outputs: {unexpected}')
 reuse=json.loads((src/'evaluation_reuse_manifest.json').read_text())
 if reuse.get('status')!='PASS' or reuse.get('missing_consumed_paths'):raise RuntimeError('source evaluation reuse manifest is incomplete')
 manifest_paths={str(row['relative_path']) for row in reuse['files']}
 for p in dst.iterdir():
  if p.is_dir() and p.name not in allowed_dirs:raise RuntimeError(f'unexpected partial output directory: {p}')
  if p.is_file() and p.name!='preflight/report_preflight.json' and p.name not in manifest_paths:raise RuntimeError(f'unexpected partial output file: {p}')
 dest_dirs=('slurm','predictions','official','labels','heads','cache','r3d','reconstruction_cache','features','reference_gpu','provenance','preflight')
 for d in dest_dirs:(dst/d).mkdir(parents=True,exist_ok=True)
 entries={};manifest_source_mismatches=[]
 for row in reuse['files']:
  rel=Path(row.get('attempt06_relative_path') or row['relative_path'])
  from_attempt=src/rel;origin=Path(row['source_path'])
  if not from_attempt.is_file() or not origin.is_file():raise FileNotFoundError(f'reuse input missing: attempt06={from_attempt}, origin={origin}')
  expected=row['sha256'];size=int(row['size'])
  if origin.stat().st_size!=size or sha(origin)!=expected:raise RuntimeError(f'original manifest source hash/size mismatch: {rel}')
  current_size=from_attempt.stat().st_size;current_sha=sha(from_attempt)
  if current_size!=size or current_sha!=expected:
   if str(rel)!='previous_attempt.json':raise RuntimeError(f'reuse source hash/size mismatch: {rel}')
   manifest_source_mismatches.append({'relative_path':str(rel),'old_reuse_manifest_sha256':expected,'old_reuse_manifest_size':size,
    'attempt06_current_sha256':current_sha,'attempt06_current_size':current_size,'reason':'attempt06 administrative provenance file was updated after its reused-input manifest was written'})
   expected=current_sha;size=current_size
  target=dst/row['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
  # Checkpoint and recorded provenance inputs remain byte copies as in attempt06.
  copy_method=row.get('method','')
  immutable_cache=rel.parts[0] in ('cache','r3d','features','reconstruction_cache','reference_gpu')
  if target.exists() or target.is_symlink():
   if not target.is_file() or target.stat().st_size!=size or sha(target)!=expected:raise RuntimeError(f'partial attempt input changed: {target}')
   method='existing_partial_verified'
  elif (size<=8*1024*1024 and not immutable_cache) or copy_method.startswith(('byte_copy','byte_copy_checkpoint','byte_copy_original_provenance','byte_copy_source_input','byte_copy_actual_entrypoint_preflight')):
   copy_file(from_attempt,target);method='byte_copy'
  else:
   target.symlink_to(from_attempt);method='file_symlink'
  if target.stat().st_size!=size or sha(target)!=expected:raise RuntimeError(f'new view hash/size mismatch: {target}')
  entries[str(target.relative_to(dst))]={'source_path':str(from_attempt),'original_manifest_source_path':str(origin),'size':size,'sha256':expected,'method':method}
 # Report source envelopes and packed predictions are outputs of the completed eval,
 # not part of the extraction cache reuse manifest. Copy/link each file individually.
 for relroot in ('official','predictions/dev','predictions/test'):
  base=src/relroot
  if not base.is_dir():raise FileNotFoundError(base)
  for p in sorted(x for x in base.rglob('*') if x.is_file()):
   rel=p.relative_to(src);target=dst/rel
   if str(rel) in entries:continue
   target.parent.mkdir(parents=True,exist_ok=True);size=p.stat().st_size;digest=sha(p)
   if size<=8*1024*1024:copy_file(p,target);method='byte_copy'
   else:target.symlink_to(p);method='file_symlink'
   if target.stat().st_size!=size or sha(target)!=digest:raise RuntimeError(f'official/prediction reuse changed: {target}')
   entries[str(rel)]={'source_path':str(p),'original_manifest_source_path':str(p),'size':size,'sha256':digest,'method':method}
 # Copy the small reports/metrics/contract inputs actually read by the report code.
 small=('official_metrics.csv','feature_probe_metrics.csv','funnel_metrics.csv','per_gt.csv','per_query.csv','per_window.csv','oracle_metrics.csv',
  'eval_complete.json','endpoint_load_check.json','extraction_complete.json','freeze_check.json','heads_complete.json','head_H1_complete.json','head_H2_complete.json','head_H3_complete.json',
  'r3d_endpoint_manifest.json','r3d_frozen_inference_receipt.json','r3d_cache_manifest.json','r3d_h0_cache_pair_identity.json','h0_cache_replay_parity.json',
  'cache_manifest.json','cohort_identity.json','cohort_manifest.json','data_contract.json','endpoint_source_hashes.json','environment_preflight.json','gpu_runtime.json',
  'previous_attempt.json','protocol.json','source_manifest.json','public_cohort_identity_receipt.json','parallel_execution_registration.json','r3d_registration_normalized.json',
  'r3d_registration_decision_contract_checks.json','cached_eval_provenance.json','cache_manifest.json')
 small=small+('h0_against_previous_precheck.json','entrypoint_preflight.json','contract_preflight.json','startup_confirmation.json','smoke.json','smoke_cpu.json',
  'cached_eval_cache_sha_verification.json','execution_code_identity.json')
 for name in small:
  p=src/name
  if not p.is_file():raise FileNotFoundError(f'report small input missing: {p}')
  target=dst/name;copy_file(p,target);entries[name]={'source_path':str(p),'original_manifest_source_path':str(p),'size':p.stat().st_size,'sha256':sha(p),'method':'byte_copy'}
 report_preflight=dst/'preflight/report_preflight.json'
 copy_file(report_preflight,dst/'report_preflight.json')
 entries['report_preflight.json']={'source_path':str(report_preflight),'original_manifest_source_path':str(report_preflight),'size':report_preflight.stat().st_size,'sha256':sha(report_preflight),'method':'byte_copy'}
 for p in sorted((src/'preflight').glob('*')):
  if not p.is_file() or p.name=='report_preflight.json':continue
  target=dst/'preflight'/p.name;copy_file(p,target)
  entries[str(target.relative_to(dst))]={'source_path':str(p),'original_manifest_source_path':str(p),'size':p.stat().st_size,'sha256':sha(p),'method':'byte_copy'}
 # Labels are mutable in later stage flows; copy files individually into a real directory.
 for p in sorted((src/'labels').glob('*.csv')):
  target=dst/'labels'/p.name;copy_file(p,target);entries[str(target.relative_to(dst))]={'source_path':str(p),'original_manifest_source_path':str(p),'size':p.stat().st_size,'sha256':sha(p),'method':'byte_copy'}
 # Preserve source receipts/manifests/logs exactly, nested under provenance.
 source_prov=dst/'provenance/source_attempt06';source_prov.mkdir(parents=True,exist_ok=True)
 for name in ('git_provenance.json','execution_files_manifest.json','source_stage_provenance.json','evaluation_reuse_manifest.json','cache_reuse_manifest.json','jobs.json','report_stage_failure.json'):
  source=src/name;target=source_prov/name;copy_file(source,target)
  entries[str(target.relative_to(dst))]={'source_path':str(source),'original_manifest_source_path':str(source),'size':source.stat().st_size,'sha256':sha(source),'method':'byte_copy_provenance'}
 for name in ('slurm/failure_report_59179.json','slurm/frozen-unified-eval-59179.err','slurm/frozen-unified-eval-59179.out'):
  if (src/name).is_file():
   source=src/name;target=source_prov/Path(name).name;copy_file(source,target)
   entries[str(target.relative_to(dst))]={'source_path':str(source),'original_manifest_source_path':str(source),'size':source.stat().st_size,'sha256':sha(source),'method':'byte_copy_provenance'}
 for attempt in ('attempt02','attempt05'):
  p=src/'provenance'/f'source_{attempt}'
  if p.is_dir():
   target_root=dst/'provenance'/f'source_{attempt}';shutil.copytree(p,target_root,dirs_exist_ok=True,symlinks=True)
   for source in sorted(x for x in p.rglob('*') if x.is_file()):
    target=target_root/source.relative_to(p);entries[str(target.relative_to(dst))]={'source_path':str(source),'original_manifest_source_path':str(source),'size':source.stat().st_size,'sha256':sha(source),'method':'byte_copy_provenance'}
 # The original 20-case manifest/source are explicit immutable report inputs.
 from scripts.report_object_locus_frozen_probe import FOCUS_SOURCE,FOCUS_MANIFEST
 for label,p in [('focus20_cases_source.jsonl',FOCUS_SOURCE),('focus20_selection_manifest.json',FOCUS_MANIFEST)]:
  if not p.is_file():raise FileNotFoundError(p)
  copy_file(p,dst/label);entries[label]={'source_path':str(p),'original_manifest_source_path':str(p),'size':p.stat().st_size,'sha256':sha(p),'method':'byte_copy'}
 for name in ('predictions','official','labels','heads','cache','r3d','reconstruction_cache','features','reference_gpu','slurm','provenance','preflight'):
  if (dst/name).is_symlink():raise RuntimeError(f'mutable/output directory cannot be linked: {dst/name}')
 reuse_doc={'status':'PASS','source_root':str(src),'new_root':str(dst),'source_execution_sha':json.loads((src/'git_provenance.json').read_text())['commit'],
  'new_report_execution_sha':a.execution_sha,'source_reuse_manifest_sha256':sha(src/'evaluation_reuse_manifest.json'),'files':[{'relative_path':k,**v} for k,v in sorted(entries.items())],
  'manifest_source_mismatches':manifest_source_mismatches,'file_count':len(entries),'directory_symlinks_used':False,'source_inputs_read_only':True}
 (dst/'evaluation_reuse_manifest.json').write_text(json.dumps(reuse_doc,indent=2)+'\n')
 provenance={'status':'PASS','source_attempt06_root':str(src),'new_attempt_root':str(dst),'source_eval_job':'59179',
  'source_eval_execution_sha':json.loads((src/'git_provenance.json').read_text())['commit'],'report_execution_sha':a.execution_sha,
  'A_B':json.loads((src/'source_stage_provenance.json').read_text())['source_stages']['A'],
  'B':json.loads((src/'source_stage_provenance.json').read_text())['source_stages']['B'],
  'C':json.loads((src/'source_stage_provenance.json').read_text())['source_stages']['C'],
  'eval':{'job_id':'59179','status':'eval_complete PASS; report failed before statistics','receipt_sha256':sha(src/'eval_complete.json')},
  'source_A_B_C_receipts_preserved_under':'provenance/source_attempt06','evaluation_outputs_reused_read_only':True}
 (dst/'source_stage_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 effective={'status':'LOCKED_REPORT_ONLY_RECOVERY','attempt_root':str(dst),'source_eval_root':str(src),
  'inherited_protocol_documents':['object_locus_frozen_probe_parallel_r3d_eval_codex_prompt.md','object_locus_frozen_probe_r3d_retry02_codex_prompt.md','object_locus_frozen_probe_resume_cd_codex_prompt.md','object_locus_frozen_probe_attempt06_classification_flatten_fix.md'],
  'report_only_scope':['paired_scene_bootstrap_from_saved_payloads','saved_reconstruction_cache_reduction','report_assembly','zip_packaging'],
  'prohibited_this_attempt':['eval main','new prediction export','model load or forward','head training','feature extraction','GPU request'],
  'source_unified_eval':{'execution_sha':json.loads((src/'git_provenance.json').read_text())['commit'],'job_id':'59179','receipt':'eval_complete.json PASS'},
  'execution_sha':a.execution_sha,'source_point_estimates_unchanged':True,'synthetic_preflight_values_excluded_from_formal_results':True}
 (dst/'effective_execution_protocol.json').write_text(json.dumps(effective,indent=2,ensure_ascii=False)+'\n')
 (dst/'execution_code_identity.json').write_text(json.dumps({'report_execution_sha':a.execution_sha,'source_eval_execution_sha':json.loads((src/'git_provenance.json').read_text())['commit'],
  'report_execution_branch':'object-locus-frozen-representation-diagnostic-v1','evaluation_rerun':False},indent=2)+'\n')
 print(json.dumps({'status':'PASS','files':len(entries),'attempt_root':str(dst)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
