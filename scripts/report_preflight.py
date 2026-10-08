#!/usr/bin/env python3
"""Read-only report assembly preflight over an already completed attempt."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,tempfile
from pathlib import Path

from scripts import report_object_locus_frozen_probe as report
from scripts.object_locus_r3d_registration import normalize_r3d_registration
from scripts.object_locus_probe_metrics import verify_gc_cache_manifest,verify_r3d_cache_manifest

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def add(items,name,status,**fields):items.append({'name':name,'status':status,**fields})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 root=a.source.resolve();out=a.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
 os.environ['CUDA_VISIBLE_DEVICES']=''
 items=[];required=['eval_complete.json','endpoint_load_check.json','extraction_complete.json','freeze_check.json','heads_complete.json',
  'r3d_endpoint_manifest.json','r3d_frozen_inference_receipt.json','r3d_cache_manifest.json','r3d_h0_cache_pair_identity.json',
  'h0_cache_replay_parity.json','h0_against_previous_precheck.json','cache_manifest.json','evaluation_reuse_manifest.json','official_metrics.csv','official','feature_probe_metrics.csv',
  'funnel_metrics.csv','per_gt.csv','per_query.csv','per_window.csv','oracle_metrics.csv','labels/train_labels.csv','labels/dev_test_scope_labels.csv',
  'labels/r3d_dev_test_scope_labels.csv','features/dev_test_q_z.npz','heads','reconstruction_cache','predictions/test','cohort_manifest.json']
 missing=[str(root/x) for x in required if not (root/x).exists()]
 for external in (report.FOCUS_SOURCE,report.FOCUS_MANIFEST,Path('/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1/four_arm_evaluation_retry01/official_results.json')):
  if not external.is_file():missing.append(str(external))
 add(items,'required_report_input_paths','PASS' if not missing else 'FAIL',missing=missing)
 if missing:raise RuntimeError(f'report inputs missing: {missing}')
 for name,fn in [('GC001_source_cache_manifest',verify_gc_cache_manifest),('R3D_source_cache_manifest',verify_r3d_cache_manifest)]:
  value=fn(root);add(items,name,value['status'],result=value)
 for name,expected in [('eval_complete.json',{'status':'PASS','readouts':11,'official_results':22}),
  ('extraction_complete.json',{'status':'PASS','complete':True,'cached_windows':1040}),
  ('r3d_frozen_inference_receipt.json',{'status':'PASS','windows':32}),
  ('heads_complete.json',{'status':'PASS','best_checkpoints':9,'final_checkpoints':9})]:
  blob=load(root/name);bad={k:{'actual':blob.get(k),'expected':v} for k,v in expected.items() if blob.get(k)!=v}
  add(items,'receipt:'+name,'PASS' if not bad else 'FAIL',sha256=sha(root/name),mismatches=bad)
  if bad:raise RuntimeError(f'receipt mismatch {name}: {bad}')
 fixed_pre=load(root/'h0_against_previous_precheck.json')
 if fixed_pre.get('counts_exact_match') is not True:raise RuntimeError('cached H0 fixed-count receipt is not PASS')
 add(items,'cached_H0_reference_receipt','PASS',sha256=sha(root/'h0_against_previous_precheck.json'),receipt=fixed_pre)
 head_identity=load(root/'preflight/head_identity_preflight.json');head_files=[]
 if head_identity.get('status')!='PASS' or len(head_identity.get('heads',[]))!=9:raise RuntimeError('C head identity receipt incomplete')
 for h in head_identity['heads']:
  for kind in ('best','final'):
   rec=h[kind];p=root/'heads'/h['head']/f"seed_{h['seed']}"/f'{kind}.pt'
   if not p.is_file() or p.stat().st_size!=rec['size'] or sha(p)!=rec['sha256'] or rec.get('identity')!='PASS':
    raise RuntimeError(f'head identity mismatch: {p}')
   if kind=='best' and h['manifest'].get('best_epoch')!=rec['epoch']:raise RuntimeError(f'best epoch mismatch: {p}')
   head_files.append({'path':str(p),'kind':kind,'sha256':rec['sha256'],'size':rec['size'],'epoch':rec['epoch']})
 add(items,'nine_best_final_head_identities','PASS',source_job=head_identity['source_job'],source_execution_sha=head_identity['source_execution_sha'],files=head_files)
 files={'official_metrics.csv':root/'official_metrics.csv','report module':Path(report.__file__),'metrics helper':Path('/space/mawb/ssst_object_locus_frozen_representation_diagnostic_v1/scripts/object_locus_probe_metrics.py')}
 normalized=report.normalize_official_rows(report.read_csv(root/'official_metrics.csv'),root/'official')
 source_rows=report.read_csv(root/'official_metrics.csv')
 identities=set();checks=[]
 for row in source_rows:
  envelope_path=root/'official'/row['cohort']/(row['readout']+'.json')
  envelope=json.loads(envelope_path.read_text());parsed=report.normalize_official_result(envelope['result'])
  for scope in ('context','true-novel'):
   identity=(row['readout'],row['cohort'],scope)
   if identity in identities:raise RuntimeError(f'duplicate official identity: {identity}')
   identities.add(identity)
   canonical=next(x for x in normalized if (x['head'],x['seed'],x['cohort'],x['scope'])==(row['head'],row['seed'],row['cohort'],scope))
   expected=parsed[scope]
   for k in ('mIoU','PQ','mAP','AP50'):
    if k not in expected or expected[k] is None:raise RuntimeError(f'missing required official metric {identity}/{k}')
    csv_scope=report.normalize_official_result(json.loads(row['result']))[scope]
    if float(canonical[k])!=float(expected[k]) or float(canonical[k])!=float(csv_scope[k]):raise RuntimeError(f'normalized official value mismatch {identity}/{k}')
   checks.append({'identity':identity,'source_official_json':str(envelope_path),
    'source_json_sha256':sha(envelope_path),'metrics':{k:canonical[k] for k in ('mIoU','PQ','mAP','AP50')}})
 if len(normalized)!=44 or len(identities)!=44:raise RuntimeError('official identity coverage is not 44 unique scopes')
 add(items,'official_envelope_normalization','PASS',function='report.normalize_official_rows',source_csv_sha256=sha(root/'official_metrics.csv'),
  source_official_json_sha256={str(root/'official'/x['cohort']/(x['readout']+'.json')):sha(root/'official'/x['cohort']/(x['readout']+'.json')) for x in source_rows},
  envelope_count=len(source_rows),scope_rows=len(normalized),unique_identities=len(identities),source_value_checks=checks)
 # Compare only registered 8 counters; A1 stays a separately reported diagnostic.
 funnel=report.read_csv(root/'funnel_metrics.csv');oracle=report.read_csv(root/'oracle_metrics.csv')
 test_novel=[r for r in funnel if r['cohort']=='test' and r['scope']=='true-novel' and r['head']=='H0']
 oracle_novel=[r for r in oracle if r['cohort']=='test' and r['scope']=='true-novel']
 actual={'test_windows':len(test_novel),'gt_count':sum(int(r['gt_count']) for r in test_novel),
  'raw_iou_ge_05':sum(int(r['a0_ge_05']) for r in oracle_novel),'raw_iou_ge_075':sum(int(r['a0_ge_075']) for r in oracle_novel),
  'a1_max_cardinality_ge_05':sum(int(r['a1_ge_05']) for r in oracle_novel),'a1_max_cardinality_ge_075':sum(int(r['a1_ge_075']) for r in oracle_novel),
  'candidate_ca_tp':sum(int(json.loads(r['candidate_ca']).get('tp',0)) for r in test_novel),
  'candidate_cw_tp':sum(int(json.loads(r['candidate_cw']).get('tp',0)) for r in test_novel),
  'packed_ca_tp':sum(int(json.loads(r['panoptic_ca']).get('tp',0)) for r in test_novel),
  'packed_cw_tp':sum(int(json.loads(r['panoptic_cw']).get('tp',0)) for r in test_novel)}
 expected={'test_windows':24,'gt_count':104,'raw_iou_ge_05':88,'raw_iou_ge_075':63,'candidate_ca_tp':68,'candidate_cw_tp':61,'packed_ca_tp':62,'packed_cw_tp':56}
 count=report.compare_fixed_counts(actual,expected)
 add(items,'fixed_GC001_reference_counts','PASS' if count['passed'] else 'FAIL',function='report.compare_fixed_counts',actual=actual,expected=expected,result=count)
 if not count['passed']:raise RuntimeError(f'fixed reference count mismatch: {count}')
 funnel_results={};source_sums={}
 for h,s in [('H0',None)]+[(h,seed) for h in ('H1','H2','H3') for seed in (20261,20262,20263)]+[('R3D',None)]:
  subset=[r for r in funnel if r['cohort']=='test' and r['scope']=='true-novel' and r['head']==h and (r['seed'] in ('','None') if s is None else int(r['seed'])==s)]
  name=report.readout_name(h,s);agg=report.aggregate_funnel_rows(subset);funnel_results[name]=agg
  source_sums[name]={group:{k:sum(int(json.loads(row[field]).get(k,0)) for row in subset) for k in ('tp','fp','fn')}
   for group,field in [('candidate_ca','candidate_ca'),('candidate_cw','candidate_cw'),('packed_ca','panoptic_ca'),('packed_cw','panoptic_cw')]}
  if agg!=source_sums[name]:raise RuntimeError(f'funnel row sum mismatch for {name}')
 add(items,'funnel_aggregation','PASS',function='report.aggregate_funnel_rows',readouts=funnel_results,independent_csv_sums=source_sums)
 support=report.class_support_rows(root)
 scratch=out.parent/'class_support_roundtrip.csv'
 with scratch.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(support[0]));w.writeheader();w.writerows(support)
 reread=report.read_csv(scratch)
 if len(reread)!=len(support) or {r['label_source'] for r in reread}!={'GC001-native','R3D-native'}:raise RuntimeError('class support CSV roundtrip or label_source failed')
 add(items,'class_support_csv_roundtrip','PASS',function='report.class_support_rows + DictWriter',rows=len(reread),columns=list(reread[0]),
  label_sources={x:sum(r['label_source']==x for r in reread) for x in ('GC001-native','R3D-native')},csv_sha256=sha(scratch))
 # Use actual normalized registration and the exact pure summary used by report.
 endpoint=load(root/'r3d_endpoint_manifest.json');registration=normalize_r3d_registration(checkpoint_metadata=endpoint['checkpoint_metadata'],checkpoint_sha256=endpoint['checkpoint_sha256'])
 if registration['registration_sha256']!=endpoint['normalized_registration_sha256']:raise RuntimeError('normalized R3D registration differs from locked endpoint receipt')
 boot={'map':{'ci95':[.01,.03]},'ap50':{'ci95':[-.01,.02]}}
 h0={'mAP':.20,'AP50':.30,'PQ':.25};r3={'mAP':.22,'AP50':.31,'PQ':.25}
 hrec={'context':{'psnr':20.,'absrel':.1},'true-novel':{'psnr':19.,'absrel':.12}}
 rrec={'context':{'psnr':19.9,'absrel':.1},'true-novel':{'psnr':18.9,'absrel':.121}}
 case_results={}
 cases={'SUCCESS':(boot,r3,True,hrec,rrec),'FAILURE':({'map':{'ci95':[-.03,-.01]},'ap50':{'ci95':[-.03,-.02]}},r3,True,hrec,rrec),
  'INCONCLUSIVE':({'map':{'ci95':[-.01,.03]},'ap50':{'ci95':[-.01,.02]}},{'mAP':.20,'AP50':.30,'PQ':.25},True,hrec,rrec),
  'INVALID':(boot,r3,False,hrec,rrec),'INCOMPLETE':(boot,r3,True,{'context':{'psnr':None},'true-novel':{'psnr':19.,'absrel':.12}},rrec)}
 for wanted,(b,rr,complete,hh,rec) in cases.items():
  result=report.summarize_r3d_result(registration,b,h0,rr,hh,rec,protocol_complete=complete)
  if result['status']!=wanted:raise RuntimeError(f'R3D report contract {wanted} returned {result}')
  md=report.r3d_report_line(result)
  doc={'synthetic_case':wanted,'result':result,'markdown_line':md}
  path=out.parent/f'synthetic_r3d_{wanted.lower()}.json';path.write_text(json.dumps(doc,indent=2)+'\n')
  if json.loads(path.read_text())['result']['status']!=wanted or f'`{wanted}`' not in md:raise RuntimeError(f'R3D synthetic serialization/render failed: {wanted}')
  case_results[wanted]={'result':result,'json_path':str(path),'json_sha256':sha(path),'markdown_line':md}
 add(items,'R3D_registered_outcome_schema','PASS',function='report.summarize_r3d_result + report.r3d_report_line',cases=case_results,
  note='synthetic inputs are isolated preflight fixtures; they are not formal metrics or algorithm conclusions')
 regchecks={'normalized_registration_sha256':registration['registration_sha256'],'source_path':registration['registration_path'],
  'source_sha256':registration['registration_source_sha256'] if 'registration_source_sha256' in registration else None,
  'registration_status':registration['status'],'protocol_matches_locked_prompt':registration['protocol_matches_locked_prompt']}
 add(items,'R3D_real_registration','PASS',**regchecks)
 # Pin the source-point files before any report job can run.
 important=[]
 for p in sorted((root/'official').glob('*/*.json'))+[root/'official_metrics.csv']:
  important.append({'path':str(p),'size':p.stat().st_size,'sha256':sha(p)})
 input_paths=['official_metrics.csv','feature_probe_metrics.csv','funnel_metrics.csv','per_gt.csv','per_query.csv','per_window.csv','oracle_metrics.csv',
  'labels/train_labels.csv','labels/dev_test_scope_labels.csv','labels/r3d_dev_test_scope_labels.csv','cohort_manifest.json','r3d_registration_normalized.json']
 source_hashes={rel:{'path':str(root/rel),'size':(root/rel).stat().st_size,'sha256':sha(root/rel)} for rel in input_paths}
 add(items,'report_input_source_hashes','PASS',files=source_hashes,official_source_files=important,
  external_fixed_case_source={'path':str(report.FOCUS_SOURCE),'sha256':sha(report.FOCUS_SOURCE)},
  external_fixed_case_manifest={'path':str(report.FOCUS_MANIFEST),'sha256':sha(report.FOCUS_MANIFEST)},
  official_and_prediction_tree=report.snapshot_official_inputs(root))
 env=report.assemble_environment(root,report_runtime={'preflight':True,'CUDA_VISIBLE_DEVICES':'','sys_executable':sys.executable})
 if env['probe_training']['legacy_cpu_runtime']['status']!='NOT_RECORDED':raise RuntimeError('unexpected historical cpu_runtime artifact presence')
 if 'not generated by GPU-head training' not in env['legacy_cpu_training_interpretation']:raise RuntimeError('legacy CPU training is incorrectly inferred')
 add(items,'stage_environment_assembly','PASS',function='report.assemble_environment',result=env,
  function_sha256=sha(Path(report.__file__)))
 result={'status':'PASS','source_root':str(root),'source_execution_sha':load(root/'git_provenance.json')['commit'],
  'preflight_module_path':str(Path(__file__).resolve()),'preflight_module_sha256':sha(Path(__file__)),
  'preflight_runtime':{'python':sys.executable,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch_cuda_available':False},
  'checks':items,'report_statistics_executed':False,'official_eval_executed':False,'reconstruction_reduction_executed':False,
  'bootstrap_executed':False,'formal_report_or_complete_written':False}
 out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'status':'PASS','checks':len(items),'report_preflight':str(out)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
