"""Prepare independent Baseline8 run; original model/feature/label math is frozen."""
from pathlib import Path
import sys,json,subprocess,shutil,os,ast
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.gs_quality_core import SOURCE_SHA,FEATURE_NAMES,write_json,digest,GSQualityHead
from scripts.gs_quality_runtime import PIN_PATH,PIN_SHA
from scripts.invoke_siu3r_official_evaluator import SIU3R_REPO,SIU3R_COMMIT
BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
RUN=BASE/'frozen_baseline8_gs_quality_20261025_v3_runtimefix'
BASE4=BASE/'frozen_baseline4_gs_quality_20261024_v4_cachefinalized'
BASE8=BASE/'v4init_isolated_extend8_recovery_20261023_v2_jsonfix/Baseline'


def main():
    if RUN.exists():raise FileExistsError('Run exists; inspect rather than duplicate')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty snapshot')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tests=Path('/tmp/gs8_quality_tests_20261025.log').read_text()
    if 'Ran 13 tests' not in tests or '\nOK\n' not in tests:raise ValueError('Tests incomplete')
    unchanged=['tokengs','scripts/reciprocity_panoptic.py','scripts/gs_quality_runtime.py','scripts/export_unified_query_checkpoint.py','scripts/train_dual_stream_v1.py','scripts/frozen_iou_head_core.py']
    old='d8bb0a97c3908932f2e0b1d39bb6f9c65aa33e01'
    if subprocess.check_output(['git','diff','--name-only',old,'HEAD','--']+unchanged,cwd=ROOT,text=True).strip():raise ValueError('Original model/render/label math changed')
    before=ast.parse(subprocess.check_output(['git','show',old+':scripts/gs_quality_core.py'],cwd=ROOT,text=True));after=ast.parse((ROOT/'scripts/gs_quality_core.py').read_text())
    for name in ['graph_stats','geometry_features','GSQualityHead','view_gt_diagnostics']:
        a=next(n for n in before.body if getattr(n,'name',None)==name);b=next(n for n in after.body if getattr(n,'name',None)==name)
        if ast.dump(a)!=ast.dump(b):raise ValueError('Feature/head definition changed')
    receipt=json.loads((BASE8/'COMPLETION_AUDIT_20261023.json').read_text());frozen=json.loads((BASE8/'eval/FINAL_CHECKPOINT_FROZEN.json').read_text())
    source=BASE8/'eval/source_checkpoint.pt'
    if receipt['status']!='PASSED' or receipt['final_checkpoint_sha256']!=SOURCE_SHA or frozen['sha256']!=SOURCE_SHA or digest(source)!=SOURCE_SHA:raise ValueError('Baseline8 source identity differs')
    checkpoint=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    if checkpoint['progress']!=dict(epoch=8,update_in_epoch=0,full_updates=8344) or len(checkpoint['model'])!=632 or len(checkpoint['optimizer']['state'])!=604:raise ValueError('Source endpoint is not original Baseline8')
    if checkpoint['meta']['architecture']!='OMEGA_V4INIT_NO_ANCHOR_CA_ISOLATED_V4_RECIPE_V1':raise ValueError('Wrong architecture')
    del checkpoint
    if digest(PIN_PATH)!=PIN_SHA:raise ValueError('Renderer binary differs')
    cfg4=json.loads((BASE4/'configuration.json').read_text())
    for path,key in [(BASE4/'cache_cohort.json','cache_cohort_sha256'),(BASE4/'scene_split.json','scene_split_sha256'),(Path(cfg4['official_cohort']),'official_cohort_sha256')]:
        if digest(path)!=cfg4[key]:raise ValueError('Original data split/cohort changed')
    split=json.loads((BASE4/'scene_split.json').read_text());windows=json.loads((BASE4/'cache_cohort.json').read_text())['records']
    if len(windows)!=10197 or [len(split[k]) for k in ['train','dev','official_eval']]!=[1072,119,312]:raise ValueError('Cohort counts differ')
    if set(split['train'])&set(split['dev']) or (set(split['train'])|set(split['dev']))&set(split['official_eval']):raise ValueError('Scene leakage')
    if len({w['scene']+'_context'+'_'.join(map(str,w['context'])) for w in windows[8337:]})!=1860:raise ValueError('Formal names not unique')
    if subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()!=SIU3R_COMMIT or subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip():raise ValueError('Evaluator differs')
    if shutil.disk_usage(BASE).free<100*1024**3:raise ValueError('Need >100GiB free for new independent cache/exports')
    if [sum(p.numel() for p in GSQualityHead(a).parameters()) for a in ['A','C']]!=[66561,74753]:raise ValueError('Head architecture differs')
    RUN.mkdir();(RUN/'cache').mkdir();os.link(source,RUN/'source_checkpoint.pt')
    (RUN/'source_checkpoint.sha256').write_text(SOURCE_SHA+'  '+str(RUN/'source_checkpoint.pt')+'\n')
    (RUN/'code.sha').write_text(code+'\n')
    for name in ['scene_split.json','cache_cohort.json']:shutil.copyfile(BASE4/name,RUN/name)
    shutil.copyfile('/tmp/gs8_quality_tests_20261025.log',RUN/'cpu_tests.log')
    cfg=dict(code_snapshot=str(ROOT),code_sha=code,source_sha256=SOURCE_SHA,source_epoch=8,source_progress=dict(epoch=8,update_in_epoch=0,full_updates=8344),
      official_cohort=cfg4['official_cohort'],official_cohort_sha256=cfg4['official_cohort_sha256'],cache_cohort_sha256=cfg4['cache_cohort_sha256'],scene_split_sha256=cfg4['scene_split_sha256'],
      renderer_path=PIN_PATH,renderer_sha256=PIN_SHA,features=FEATURE_NAMES,seed=31415,trained_arms=['A','C'],world_size=8,
      lr=.001,batch_size=1024,max_epochs=100,patience=12,calibration_gamma=.25,formal_selection='dev MSE only',
      target='original Context class-agnostic binary IoU, unchanged from epoch4 GS experiment',
      historical_baseline8_metrics=str(BASE8/'eval/evaluation/metrics.json'),baseline4_comparison=str(BASE4/'RESULTS.json'),
      backbone_updates=0,reuse_epoch4_features_or_targets=False,scene_split_caveat='dev is head-held-out, backbone-trained scenes',
      interpretation='A8/C8 paired within epoch8; epoch4 comparison changes source checkpoint and reruns frozen forwards',
      allowed_nodes=['3dimage-'+str(n) for n in [11,12,13,14,17,18]],partitions=['3090','4090'],single_node_per_stage=True,
      replaces_jobs=[71646,71647,71648,71649])
    write_json(RUN/'configuration.json',cfg)
    write_json(RUN/'PREFLIGHT.json',dict(status='PASSED',code_sha=code,source_sha256=SOURCE_SHA,source_endpoint_verified=True,
      source_architecture_tensor_optimizer_schema_verified=True,head_and_GS_feature_functions_AST_identical=True,tests=13,
      pinned_renderer_binary_verified=True,prior_node13_compatibility='same partial_1406 renderer completed earlier Baseline16 diagnostic on node13',
      new_GPU_execution_verified=False,source_data_split_hashes_verified=True,new_cache_required=True,backbone_updates=0))
    print(json.dumps(cfg,indent=2))


if __name__=='__main__':main()
