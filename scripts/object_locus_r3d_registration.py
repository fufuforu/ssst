"""CPU-only mapping of the frozen R3D registration and registered outcome rules."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

REGISTRATION_PATH=Path('/space/mawb/ssst/group_plus/object_locus_output_refine_gc001_v1/attempts/attempt02/evaluation_registration.json')
R3D_ATTEMPT=REGISTRATION_PATH.parent
EXPECTED={
 'source_checkpoint_sha256':'68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a',
 'plan_sha256':'0a7c8173b3dcc187d7ce3074e69d9c8649204a933262ded3a773b08898650ea8',
 'gc001_checkpoint_sha256':'72f7440b5b3cf2fd877dfc534ae5c4a409c76c9884729fe23247008aab8c0d58',
 'source_manifest_sha256':'a03e6842e9657d512a0de2b9dd20ed73aa9bcc61ec8f4fd8d33ed07e8cc11249',
 'checkpoint_sha256':'9f06d78c58bd5e00b84ed712840e767c3151db1fe170f3408c6079fca263db2c',
 'training_code_sha':'d4c096b80c4a28e93e095abac7dd777ed1af5f3a',
 'gc001_source_sha256':'68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a'}
SUCCESS_EXPECTED={'delta_map_min':.01,'delta_map_paired_scene_bootstrap_95ci_lower_gt':0,
 'delta_ap50_min':-.01,'delta_pq_min':-.01,'context_psnr_drop_max_db':.5,
 'true_novel_psnr_drop_max_db':.5,'true_novel_absrel_ratio_max':1.05,
 'integrity_and_budget_valid':True}
FAILURE_EXPECTED=['valid delta-mAP CI upper <= 0','valid delta-AP50 CI upper < -0.01','reconstruction protection condition fails']
FAILURE_MAP={
 FAILURE_EXPECTED[0]:{'condition':'map_ci_upper_lte_zero','source_field_path':'evaluation_registration.json:failure[0]'},
 FAILURE_EXPECTED[1]:{'condition':'ap50_ci_upper_lt_minus_0_01','source_field_path':'evaluation_registration.json:failure[1]'},
 FAILURE_EXPECTED[2]:{'condition':'registered_reconstruction_protection_failed','source_field_path':'evaluation_registration.json:failure[2]'}}
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def _need(obj,path):
 cur=obj
 for part in path.split('.'):
  if part.isdigit():cur=cur[int(part)]
  else:cur=cur[part]
 return cur
def normalize_r3d_registration(source_dir=R3D_ATTEMPT,*,checkpoint_metadata,checkpoint_sha256):
 root=Path(source_dir)
 names=('evaluation_registration.json','training_complete.json','run_manifest.json','progress.json','checkpoint_roundtrip_exact.json')
 files={n:root/n for n in names}
 for p in files.values():
  if not p.is_file():raise ValueError(f'missing registered R3D source: {p}')
 docs={n:json.loads(p.read_text()) for n,p in files.items()}
 reg,receipt,run,progress,roundtrip=(docs[n] for n in names)
 regpath='evaluation_registration.json:'
 runpath='run_manifest.json:'
 rcpath='training_complete.json:'
 pp='progress.json:'
 cp='checkpoint_epoch8.pt metadata:'
 if reg['source_checkpoint_sha256']!=EXPECTED['source_checkpoint_sha256']:raise ValueError('R3D source checkpoint SHA conflicts with fixed source')
 if reg['source_manifest_sha256']!=EXPECTED['source_manifest_sha256'] or reg['control_checkpoint_sha256']!=EXPECTED['gc001_checkpoint_sha256']:raise ValueError('R3D registered control checkpoint/source manifest identity conflict')
 if reg['fixed_training_plan_sha256']!=EXPECTED['plan_sha256']:raise ValueError('R3D plan SHA conflicts with fixed plan')
 if reg['alpha']!=.01:raise ValueError('R3D registered alpha does not match fixed endpoint')
 if reg['training']['logical_global_slots']!=8:raise ValueError('registered logical slots must be 8')
 if reg['execution_conditions']['R3D']!='4 physical GPUs x two batch-1 microbatches, accumulation=2':raise ValueError('registered R3D execution condition conflicts')
 if run['code_sha']!=EXPECTED['training_code_sha'] or run['source_checkpoint_sha256']!=EXPECTED['gc001_source_sha256'] or run['plan_sha256']!=EXPECTED['plan_sha256']:raise ValueError('run manifest source/code/plan identity conflict')
 if (run['world_size'],run['logical_global_slots'],run['gradient_accumulation_steps'])!=(4,8,2):raise ValueError('run manifest physical/logical/accumulation budget conflict')
 if (run['updates'],run['new_exposures'],run['model_exposure_endpoint'],run['source_manifest_sha256'])!=(1008,8064,58128,EXPECTED['source_manifest_sha256']):raise ValueError('run manifest update/exposure/source manifest conflict')
 if (reg['training']['epochs'],reg['training']['updates'],reg['training']['new_exposures'])!=(8,1008,8064):raise ValueError('registered training budget conflict')
 if reg['success']!=SUCCESS_EXPECTED:raise ValueError(f'registered success rules differ: {reg["success"]!r}')
 if reg['failure']!=FAILURE_EXPECTED:raise ValueError(f'registered failure rules differ: {reg["failure"]!r}')
 if reg['otherwise']!='INCONCLUSIVE unless protocol error makes result INVALID':raise ValueError('registered otherwise clause conflicts')
 if receipt['status']!='WAIT_USER_NOTIFICATION_FOR_EVALUATION' or receipt['completed_updates']!=1008 or receipt['new_exposures']!=8064 or receipt['model_exposure']!=58128:raise ValueError('training completion receipt does not match endpoint budget')
 counts=receipt['window_exposure_counts']
 if len(counts)!=1008 or any(x!=8 for x in counts):raise ValueError('training window exposure counts are not exactly eight')
 if (progress['status'],progress['completed_updates'],progress['new_exposures'],progress['model_exposure'])!=('COMPLETE',1008,8064,58128):raise ValueError('progress receipt does not match completed endpoint')
 if roundtrip['status']!='PASS' or roundtrip['all_state_keys_shapes_dtypes_values_exact'] is not True:raise ValueError('checkpoint roundtrip is not exact PASS')
 ckpt=Path(checkpoint_metadata.get('checkpoint_path',root/'checkpoint_epoch8.pt'))
 observed={k:checkpoint_metadata[k] for k in ('epoch','completed_updates','new_exposures','source_exposure','model_exposure','alpha','code_sha','source_sha256','plan_sha256')}
 expected={'epoch':8,'completed_updates':1008,'new_exposures':8064,'source_exposure':50064,'model_exposure':58128,'alpha':.01,
  'code_sha':EXPECTED['training_code_sha'],'source_sha256':EXPECTED['gc001_source_sha256'],'plan_sha256':EXPECTED['plan_sha256']}
 if observed!=expected:raise ValueError(f'epoch8 checkpoint metadata mismatch: {observed!r}')
 if checkpoint_sha256!=EXPECTED['checkpoint_sha256']:raise ValueError(f'R3D checkpoint SHA mismatch: {checkpoint_sha256}')
 mappings={
  'source_checkpoint_sha256':regpath+'source_checkpoint_sha256','plan_sha256':regpath+'fixed_training_plan_sha256',
  'logical_global_slots':regpath+'training.logical_global_slots','physical_world_size':runpath+'world_size',
  'gradient_accumulation_steps':runpath+'gradient_accumulation_steps','execution_description':regpath+'execution_conditions.R3D',
  'delta_map_min':regpath+'success.delta_map_min','map_ci_lower_gt':regpath+'success.delta_map_paired_scene_bootstrap_95ci_lower_gt',
  'delta_ap50_min':regpath+'success.delta_ap50_min','delta_pq_min':regpath+'success.delta_pq_min',
  'context_psnr_drop_max_db':regpath+'success.context_psnr_drop_max_db','true_novel_psnr_drop_max_db':regpath+'success.true_novel_psnr_drop_max_db',
  'true_novel_absrel_ratio_max':regpath+'success.true_novel_absrel_ratio_max','integrity_and_budget_valid':regpath+'success.integrity_and_budget_valid',
  'failure_conditions':regpath+'failure[0..2]','map_ci_upper_failure_lte':regpath+'failure[0] parsed after exact clause validation',
  'ap50_ci_upper_failure_lt':regpath+'failure[1] parsed after exact clause validation','otherwise':regpath+'otherwise',
  'epoch':cp+'epoch','completed_updates':cp+'completed_updates','new_exposures':cp+'new_exposures','source_exposure':cp+'source_exposure',
  'model_exposure':cp+'model_exposure','training_code_sha':cp+'code_sha','checkpoint_sha256':'checkpoint_epoch8.pt SHA256'}
 normalized_success={'delta_map_min':reg['success']['delta_map_min'],
  'map_ci_lower_gt':reg['success']['delta_map_paired_scene_bootstrap_95ci_lower_gt'],
  'delta_ap50_min':reg['success']['delta_ap50_min'],'delta_pq_min':reg['success']['delta_pq_min'],
  'context_psnr_drop_max_db':reg['success']['context_psnr_drop_max_db'],
  'true_novel_psnr_drop_max_db':reg['success']['true_novel_psnr_drop_max_db'],
  'true_novel_absrel_ratio_max':reg['success']['true_novel_absrel_ratio_max'],
  'integrity_and_budget_valid':reg['success']['integrity_and_budget_valid'],
  'map_ci_upper_failure_lte':0.0,'ap50_ci_upper_failure_lt':-.01}
 return {'status':'PASS','registration_path':str(files['evaluation_registration.json'].resolve()),
  'registration_sha256':sha256(files['evaluation_registration.json']),
  'source_files':{n:{'path':str(p.resolve()),'sha256':sha256(p)} for n,p in files.items()},
  'checkpoint_path':str(ckpt.resolve()),'checkpoint_sha256':checkpoint_sha256,'checkpoint_metadata':observed,
  'identity':{'source_checkpoint_sha256':reg['source_checkpoint_sha256'],'plan_sha256':reg['fixed_training_plan_sha256'],
   'training_code_sha':run['code_sha'],'epoch':8,'completed_updates':1008,'new_exposures':8064,'source_exposure':50064,
   'model_exposure':58128,'physical_world_size':4,'logical_global_slots':8,'gradient_accumulation_steps':2,
   'alpha':.01,'per_window_exposures_exactly_eight':True},
  'success_conditions':normalized_success,'failure_conditions':[dict(text=s,**FAILURE_MAP[s]) for s in FAILURE_EXPECTED],
  'otherwise_clause':reg['otherwise'],'field_mappings':mappings,
  'protocol_matches_locked_prompt':True,'b_and_d_shared_parser':'scripts.object_locus_r3d_registration.normalize_r3d_registration'}
def registered_outcome(registration,metrics,*,protocol_complete):
 if not protocol_complete:return {'status':'INVALID','algorithm_conclusion':None,'reason':'registered protocol/budget/receipt contract incomplete'}
 if registration.get('status')!='PASS' or registration.get('protocol_matches_locked_prompt') is not True:
  return {'status':'INVALID','algorithm_conclusion':None,'reason':'registration/source validation failed'}
 required=('delta_map','map_ci_lower','map_ci_upper','delta_ap50','ap50_ci_upper','delta_pq','context_psnr_drop_db','true_novel_psnr_drop_db','true_novel_absrel_ratio')
 missing=[k for k in required if metrics.get(k) is None]
 if missing:return {'status':'INCOMPLETE','algorithm_conclusion':None,'reason':'required registered metrics are missing','missing_metrics':missing}
 s=registration['success_conditions'];f=registration['failure_conditions']
 reconstruction_failed=(metrics['context_psnr_drop_db']>s['context_psnr_drop_max_db'] or
  metrics['true_novel_psnr_drop_db']>s['true_novel_psnr_drop_max_db'] or
  metrics['true_novel_absrel_ratio']>s['true_novel_absrel_ratio_max'])
 failure=(metrics['map_ci_upper']<=s['map_ci_upper_failure_lte'] or metrics['ap50_ci_upper']<s['ap50_ci_upper_failure_lt'] or reconstruction_failed)
 success=(metrics['delta_map']>=s['delta_map_min'] and metrics['map_ci_lower']>s['map_ci_lower_gt'] and
  metrics['delta_ap50']>=s['delta_ap50_min'] and metrics['delta_pq']>=s['delta_pq_min'] and not reconstruction_failed)
 if success:status='SUCCESS'
 elif failure:status='FAILURE'
 else:status='INCONCLUSIVE'
 return {'status':status,'algorithm_conclusion':status,'reason':None,'metrics':dict(metrics),
  'registered_success_conditions':s,'registered_failure_conditions':[x['text'] for x in f]}
