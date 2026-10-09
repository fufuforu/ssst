#!/usr/bin/env python3
"""Real cached-payload preflight for the frozen-probe report bootstrap entry."""
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import report_object_locus_frozen_probe as report

ATTEMPTS=Path('/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts')
SOURCE=ATTEMPTS/'attempt06';FAILED=ATTEMPTS/'attempt07'
READOUTS=['H0']+[f'{h}_seed_{s}' for h in ('H1','H2','H3') for s in (20261,20262,20263)]+['R3D']

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def h0_inventory(root,windows):
 base=root/'predictions/test/H0/official'
 if not base.exists():return {'root':str(root),'official_root':str(base),'exists':False,'children':[],'missing_path':str(base)}
 expected={report.packed_pair_name(w) for w in windows}
 out=[]
 for p in sorted(base.iterdir()):
  q=p.resolve(strict=True)
  out.append({'name':p.name,'is_file':p.is_file(),'is_dir':p.is_dir(),'is_symlink':p.is_symlink(),
      'resolved_path':str(q),'has_target_seg_pred':(q/'target_seg_pred').is_dir() if q.is_dir() else False,
      'has_target_seg_gt':(q/'target_seg_gt').is_dir() if q.is_dir() else False,
      'expected_window_identity':p.name in expected})
 dirs={x['name'] for x in out if x['is_dir']}
 return {'root':str(root),'official_root':str(base),'exists':True,'children':out,'expected_window_count':len(expected),
      'actual_window_directories':len(dirs&expected),'extra_directories':sorted(dirs-expected),
      'missing_directories':sorted(expected-dirs),'extra_non_window_items':[x for x in out if x['name'] not in expected]}

def snapshot_dir(path):
 p=Path(path)
 if not p.exists():return {'exists':False,'path':str(p),'children':[]}
 rows=[]
 for x in sorted(p.iterdir()):
  r={'name':x.name,'is_file':x.is_file(),'is_dir':x.is_dir(),'is_symlink':x.is_symlink(),
     'resolved_path':str(x.resolve(strict=True)) if x.exists() else str(x)}
  if x.is_file():r.update({'size':x.stat().st_size,'sha256':digest(x)})
  rows.append(r)
 return {'exists':True,'path':str(p),'children':rows}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=Path,required=True);args=ap.parse_args();root=args.attempt
 root.mkdir(parents=True,exist_ok=True)
 expected=json.loads((SOURCE/'cohort_manifest.json').read_text())['test']
 diagnosis={'status':'RUNNING','expected_test_windows':[{'scene':w['scene'],'context':w['context'],'novel':w['novel'],
    'pair_name':report.packed_pair_name(w)} for w in expected],
    'roots':{name:h0_inventory(path,expected) for name,path in [('attempt06',SOURCE),('attempt07',FAILED),(root.name,root)]}}
 for name,rec in diagnosis['roots'].items():
  if not rec.get('exists') or rec['actual_window_directories']!=24 or rec['missing_directories'] or rec['extra_directories']:
   raise RuntimeError(f'{name} H0 directory is missing/extra manifest windows: {rec}')
 # Verify file-by-file reused view against the valid completed evaluation source.
 src_hashes=report.snapshot_official_inputs(SOURCE);view_hashes=report.snapshot_official_inputs(root)
 if src_hashes!=view_hashes:raise RuntimeError('attempt08 official/prediction files differ from attempt06 source')
 for rel,h in src_hashes.items():
  p=SOURCE/rel
  if not p.is_file() or p.stat().st_size!=h['size'] or digest(p)!=h['sha256']:
   raise RuntimeError(f'attempt06 source hash/size mismatch: {rel}')
 diagnosis['new_view_matches_attempt06']='PASS'
 diagnosis['source_input_file_count']=len(src_hashes)
 reuse_path=root/'evaluation_reuse_manifest.json';reuse=json.loads(reuse_path.read_text())
 reuse_checked=0
 prior=reuse.get('status')=='PASS' and reuse.get('view_validation',{}).get('status')=='PASS'
 for item in reuse['files']:
  source=Path(item['source_path']);target=root/item['relative_path']
  if not source.is_file() or not target.is_file():raise FileNotFoundError(f'reuse source/view input missing: {source} -> {target}')
  if source.stat().st_size!=item['size'] or target.stat().st_size!=item['size']:
   raise RuntimeError(f'reuse source/view size mismatch: {source} -> {target}')
  if not prior and (digest(source)!=item['sha256'] or digest(target)!=item['sha256']):
   raise RuntimeError(f'reuse source/view SHA mismatch: {source} -> {target}')
  if item['method']=='file_symlink' and target.resolve(strict=True)!=source.resolve(strict=True):
   raise RuntimeError(f'file-level input symlink target mismatch: {target}')
  if item['method']=='byte_copy' and target.is_symlink():raise RuntimeError(f'small source input should be copied, not linked: {target}')
  reuse_checked+=1
 if not prior:
  reuse['status']='PASS';reuse['view_validation']={'status':'PASS','verified_files':reuse_checked,
   'source_and_view_sizes_checked':True,'source_and_view_sha256_checked':True,'file_level_symlink_targets_checked':True}
 else:
  reuse['view_validation']['subsequent_read_check']={'status':'PASS','verified_files':reuse_checked,
   'sizes_and_link_targets_rechecked':True,'sha256_evidence':'reused from prior PASS verification; no input files modified'}
 report.dump(reuse_path,reuse)
 # Same formal index function validates both fixed cohorts and all 11 readouts.
 test_index=report.build_packed_window_index(root,'test',READOUTS)
 dev_index=report.build_packed_window_index(root,'dev',READOUTS)
 pred_frames={name:sum(len(e['files']['context']['pred_png'])+len(e['files']['target']['pred_png']) for e in test_index['entries'] if e['readout']==name) for name in READOUTS}
 if any(n!=144 for n in pred_frames.values()) or sum(pred_frames.values())!=1584:
  raise RuntimeError(f'packed prediction frame inventory mismatch: {pred_frames}')
 diagnosis['all_11_test_readout_frame_counts']=pred_frames
 diagnosis['all_11_test_windows']=24
 diagnosis['all_11_dev_windows']=8
 diagnosis['extra_non_window_items_by_readout']={k:v['excluded_auxiliary_items'] for k,v in test_index['directory_diagnostics'].items()}
 report.dump(root/'packed_window_diagnosis.json',diagnosis)
 report.dump(root/'packed_window_index.json',test_index)
 # Evaluator.setup receives isolated scratch path; parser payload input remains the read-only attempt08 view.
 import torch
 torch.set_num_threads(4)
 scratch=root/'preflight'/'siu3r_setup_scratch';scratch.mkdir(parents=True,exist_ok=True)
 roots_before={n:snapshot_dir(p/'predictions/test/H0/official') for n,p in [('attempt06',SOURCE),('attempt07',FAILED),(root.name,root)]}
 scratch_before=snapshot_dir(scratch)
 evaluator=report.make_official_evaluator(root,setup_path=scratch)
 roots_after={n:snapshot_dir(p/'predictions/test/H0/official') for n,p in [('attempt06',SOURCE),('attempt07',FAILED),(root.name,root)]}
 scratch_after=snapshot_dir(scratch)
 setup_mutations={n:roots_before[n]!=roots_after[n] for n in roots_before}
 if any(setup_mutations.values()) or scratch_before!=scratch_after:
  raise RuntimeError(f'SIU3R Evaluator.setup mutated a prediction root or scratch: roots={setup_mutations}, scratch={scratch_before!=scratch_after}')
 official=report.read_csv(root/'official_metrics.csv');perwindow=report.read_csv(root/'per_window.csv')
 points_path=root/'preflight'/'point_reproduction_preflight.json'
 comparisons,points,scenes,matrix=report.official_bootstrap(root,official,perwindow,preflight_only=True,setup_path=scratch)
 result=json.loads((root/'preflight/bootstrap_entry_preflight.json').read_text())
 result['source_files']={'cohort_manifest':{'path':str(root/'cohort_manifest.json'),'sha256':digest(root/'cohort_manifest.json')},
    'per_window':{'path':str(root/'per_window.csv'),'sha256':digest(root/'per_window.csv')},
    'official_metrics':{'path':str(root/'official_metrics.csv'),'sha256':digest(root/'official_metrics.csv')},
    'siu3r_evaluator_source':{'path':'/space/mawb/SIU3R/src/evaluator.py','sha256':digest('/space/mawb/SIU3R/src/evaluator.py')},
    'report_function_source':{'path':str(ROOT/'scripts/report_object_locus_frozen_probe.py'),'sha256':digest(ROOT/'scripts/report_object_locus_frozen_probe.py')},
    'preflight_code':{'path':str(Path(__file__).resolve()),'sha256':digest(__file__)}}
 result['checks']={'manifest_test_window_count':24,'dev_window_count':8,'readouts':11,'prediction_frames_per_test_readout':pred_frames,
     'prediction_frames_total':sum(pred_frames.values()),'attempt06_attempt07_new_root_h0_diagnosis':'PASS',
     'source_hashes_before_after_setup_unchanged':not any(setup_mutations.values()),'setup_scratch_unchanged':scratch_before==scratch_after,
     'point_values_match_cached_official_atol_1e6':all(v['mAP_abs_difference']<=1e-6 and v['AP50_abs_difference']<=1e-6 for v in result['point_reproduction'].values()),
     'first_replicate_all_finite':result['replicate0_all_finite'],'formal_2000_resamples_unchanged':result['formal_resamples_remain']==2000}
 if not all(result['checks'].values()):raise RuntimeError(f'report bootstrap preflight failed: {result["checks"]}')
 result['status']='PASS'
 diagnosis['status']='PASS'
 diagnosis['evaluator_setup']={'status':'PASS','prediction_roots_unchanged':not any(setup_mutations.values()),
   'prediction_roots_before':roots_before,'prediction_roots_after':roots_after,
   'isolated_scratch_unchanged':scratch_before==scratch_after,'scratch_before':scratch_before,
   'scratch_after':scratch_after,'setup_path':str(scratch)}
 diagnosis['reuse_manifest_validation']={'status':'PASS','verified_files':reuse_checked}
 report.dump(root/'packed_window_diagnosis.json',diagnosis)
 report.dump(points_path,{'status':'PREFLIGHT_ONLY_NOT_GENERALIZATION_RESULT','point_reproduction':result['point_reproduction'],
     'first_shared_selection_global_ap':result['replicate0_global_ap'],'comparison_deltas':{
     'H1_seed_20261-H0':{m:result['replicate0_global_ap']['H1_seed_20261'][m]-result['replicate0_global_ap']['H0'][m] for m in ('mAP','AP50')},
     'H2_seed_20261-H1_seed_20261':{m:result['replicate0_global_ap']['H2_seed_20261'][m]-result['replicate0_global_ap']['H1_seed_20261'][m] for m in ('mAP','AP50')},
     'H3_seed_20261-H1_seed_20261':{m:result['replicate0_global_ap']['H3_seed_20261'][m]-result['replicate0_global_ap']['H1_seed_20261'][m] for m in ('mAP','AP50')},
     'R3D-H0':{m:result['replicate0_global_ap']['R3D'][m]-result['replicate0_global_ap']['H0'][m] for m in ('mAP','AP50')}},
     'scene_order':scenes,'selection_scene_names':result['selection_scene_names'],'selection_draw_counts':result['selection_scene_draw_counts']})
 report.dump(root/'preflight/bootstrap_entry_preflight.json',result)
 print(json.dumps({'status':'PASS','points':result['point_reproduction'],'first_replicate':result['replicate0_global_ap'],
     'attempt06_attempt07_new_root_h0_diagnosis':{k:{'expected':v['expected_window_count'],'actual_dirs':v['actual_window_directories'],
       'extra_non_window_items':[x['name'] for x in v['extra_non_window_items']]} for k,v in diagnosis['roots'].items()}},indent=2),flush=True)

if __name__=='__main__':main()
