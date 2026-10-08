#!/usr/bin/env python3
"""Create immutable attempt-level protocol and provenance records."""
import argparse,hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import numpy as np,torch
DEFAULT_ATTEMPT=Path('/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt02')
ATTEMPT=DEFAULT_ATTEMPT
BASE=Path('/space/mawb/ssst')
FILES={
 'checkpoint':Path('/space/mawb/ssst/workspace_group_plus/object_locus_gc_sweep_v1/gc001/checkpoint_epoch8.pt'),
 'source_training_plan':Path('/space/mawb/ssst/group_plus/object_locus_gc_sweep_v1/training_plan.json'),
 'data_manifest':Path('/space/mawb/ssst/group_plus/object_locus_panoptic_v1_8gpu/data_manifest.json'),
 'four_arm_cohort':Path('/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1/four_arm_evaluation_retry01/cohort_manifest.json'),
 'four_arm_window_identities':Path('/space/mawb/ssst/group_plus/object_locus_competition_gc001_v1/four_arm_evaluation_retry01/window_identities.json'),
 'focus20':Path('/space/mawb/ssst/group_plus/object_locus_instance_attribution_v1/attempts/attempt00/results/focus20_cases.jsonl'),
 'task_prompt':Path('/space/mawb/ssst/docs/object_locus_frozen_representation_diagnostic_v1_codex_prompt.md'),
}
EXPECTED={'checkpoint':'72f7440b5b3cf2fd877dfc534ae5c4a409c76c9884729fe23247008aab8c0d58',
 'source_training_plan':'0a7c8173b3dcc187d7ce3074e69d9c8649204a933262ded3a773b08898650ea8',
 'data_manifest':'a03e6842e9657d512a0de2b9dd20ed73aa9bcc61ec8f4fd8d33ed07e8cc11249'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 p=ATTEMPT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=str)+'\n')
def main():
 global ATTEMPT
 ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=Path,default=DEFAULT_ATTEMPT);ATTEMPT=ap.parse_args().attempt
 if not ATTEMPT.is_dir():raise RuntimeError(f'attempt directory missing: {ATTEMPT}')
 managed=('source_manifest.json','git_provenance.json','protocol.json','data_contract.json','environment_preflight.json')
 existing=[n for n in managed if (ATTEMPT/n).exists()]
 if existing:raise RuntimeError(f'attempt preparation outputs already exist; refusing overwrite: {existing}')
 for name,p in FILES.items():
  if not p.is_file():raise FileNotFoundError(p)
  digest=sha(p)
  if name in EXPECTED and digest!=EXPECTED[name]:raise RuntimeError(f'{name} SHA mismatch: {digest}')
 assets={name:{'path':str(p),'sha256':sha(p),'size':p.stat().st_size} for name,p in FILES.items()}
 provenance={
  'base_branch':'object-locus-gc-sweep-v1','base_sha':'9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0',
  'official_evaluation_source_branch':'object-locus-competition-gc001-v1','official_evaluation_source_sha':'68e2376ffef5171206f458d82a063aac35e36891',
  'official_exporter_blob_sha':'5858ca77275a305abe09260b9cb10612b2c801b2',
  'official_source_file_blobs':{
   path:subprocess.check_output(['git','rev-parse',f'68e2376ffef5171206f458d82a063aac35e36891:{path}'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
   for path in ['scripts/export_object_locus_v3_set_official.py','scripts/eval_object_locus_v3_set.py',
    'scripts/eval_object_locus_gc_competition_four_arm.py','scripts/summarize_object_locus_gc_competition_four_arm.py',
    'scripts/invoke_siu3r_official_evaluator.py','scripts/object_locus_gc_sweep_runtime.py','scripts/object_locus_panoptic_v1_runtime.py']},
  'siu3r_commit':'8ea80166be76854f938e90521f1a5b688b755c87',
  'checkpoint_training_code_sha':'9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0',
  'task_branch':'object-locus-frozen-representation-diagnostic-v1',
  'execution_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
  'remote_origin':subprocess.check_output(['git','remote','get-url','origin'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
  'source_functions':['assemble_panoptic','_save_packed','write_official_pair','_targets','_prediction_candidate_data','_candidate_stats','official CPU Evaluator invocation','official packed AP scene bootstrap'],
  'source_assets':assets}
 protocol={'name':'Object-Locus Frozen Representation Diagnostic V1','status':'PREPARED',
  'checkpoint':assets['checkpoint'],'checkpoint_sha256':EXPECTED['checkpoint'],'alpha':.01,'epoch':8,
  'completed_updates':1008,'new_exposures':8064,'source_exposure':50064,'model_exposure':58128,'forward_step':58128,'beta':.1,
  'data':{'train_scenes':128,'train_windows':1008,'train_forward_count_per_window':1,'dev_scenes':8,'dev_windows':8,'test_windows':24,'test_split':'val32_excluding_dev8_scenes','test_preset_reuse':'exploratory prior research cohort'},
  'labels':{'query_count':100,'head_classes':'thing internal 2..19 -> labels 0..17; no-object label18; ambiguous -1','iou_match_threshold':.5,'negative_max_iou_strictly_below':.1,'oracle_thresholds':[.5,.75]},
  'heads':{'H0':'original cached classifier, no training','H1':'LN256 eps 1e-5 + Linear256->19 on q','H2':'LN256 eps 1e-5 + Linear256->19 on opacity-mass pooled z','H3':'independent LN256(q), LN256(z), concat512 + Linear512->19','seeds':[20261,20262,20263],'primary_seed':20261},
  'training':{'optimizer':'AdamW','lr':.001,'weight_decay_linear_weight':.0001,'other_decay':0,'batch_windows':16,'updates_per_epoch':63,'epochs_max':50,'early_stop':'dev context CE; minimum10, patience10, improvement1e-8','clip_grad_norm':1.},
  'task_protocol_sha256':assets['task_prompt']['sha256'],'bootstrap':{'scenes':24,'resamples':2000,'seed':2026,'replacement':'scene-level paired; same index matrix for all comparisons/scopes/class metrics'},
  'forbidden_changes':['GC001 model weights/buffers','full val32 or full 1860 windows','extra pooling/probes/heads/calibration','R3D or comp arms','shared environment/dependencies']}
 save('source_manifest.json',{'assets':assets,'checkpoint_identity':{'sha256':EXPECTED['checkpoint'],'alpha':.01,'epoch':8,'completed_updates':1008,'new_exposures':8064,'source_exposure':50064,'model_exposure':58128,'code_sha':'9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0'},'siu3r_commit':'8ea80166be76854f938e90521f1a5b688b755c87'})
 save('git_provenance.json',provenance);save('protocol.json',protocol)
 save('data_contract.json',{'source_sha_checks':'PASS','four_arm_test_identity_check':'deferred to extractor','scene_disjointness':'deferred to extractor','query_count':100,'train_windows':1008,'dev_windows':8,'test_windows':24,'status':'PENDING'})
 try: gpu=subprocess.check_output(['nvidia-smi','-L'],text=True)
 except Exception as e:gpu=str(e)
 try: slurm=subprocess.check_output(['scontrol','show','node','3dimage-11'],text=True)
 except Exception as e:slurm=str(e)
 env={'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,
      'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_listing':gpu,'node_resources':slurm,
      'model_python':'/space/mawb/anaconda3/envs/tokengs/bin/python','official_python':'/space/mawb/SIU3R/.venv_gpu_v4/bin/python',
      'device_policy':'extract single Slurm GPU; probes/evaluator CUDA_VISIBLE_DEVICES empty; FP32; TF32 disabled','cpu_threads':4}
 save('environment_preflight.json',env)
 if not (ATTEMPT/'slurm').exists():(ATTEMPT/'slurm').mkdir()
 print(json.dumps({'prepared':True,'execution_sha':provenance['execution_sha'],'assets':assets},indent=2))
if __name__=='__main__':main()
