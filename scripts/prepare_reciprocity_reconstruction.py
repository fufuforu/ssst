"""CPU source/data/teacher validation and exclusive new independent run roots."""
from pathlib import Path
import sys,json,os,shutil,subprocess,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.membership_geometry_core import *
from scripts.gs_quality_core import PARENT,FEATURE_NAMES
from scripts.gs_quality_runtime import PIN_PATH,PIN_SHA,verify_renderer
BASELINE16_SHA='126085013533dcc4a8eaf1c0a8a6eae2ba25af2d8d2c6c7288055f0412172e62'
M=BASE/'membership_guided_geometry_20261024_v3_ddpgate'
Q=BASE/'frozen_baseline4_gs_quality_20261024_v3_freshpair'
R=BASE/'baseline16_reconstruction_gap_20261024_v1'
OLD=BASE/'membership_guided_geometry_20261023_v2_sdpafix'

def model_restore_check(saved):
    from torch import nn
    from tokengs.options import config_defaults
    from tokengs.models.v4init_isolated_noca import V4InitIsolatedNoCACore
    from tokengs.models.unified_superpoints import UnifiedQueryConfig
    from scripts.train_unified_query_v1 import restore_state
    from scripts.v4init_isolated_recipe import optimizer_groups
    torch.manual_seed(42);model=nn.Module();model.gfm_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256));model.core=V4InitIsolatedNoCACore(config_defaults['train_siu3r_unified_query_v1'],UnifiedQueryConfig(**saved['meta']['config']))
    restore_state(model,saved['model'])
    if len(model.state_dict())!=632 or any(not torch.equal(v,saved['model'][n]) for n,v in model.state_dict().items()):raise ValueError('Model restore differs')
    optim=torch.optim.AdamW(optimizer_groups(model),betas=(.9,.95),eps=1e-8);optim.load_state_dict(copy.deepcopy(saved['optimizer']))
    actual=optim.state_dict();original=saved['optimizer']
    if actual['param_groups']!=original['param_groups'] or any(not torch.equal(v,original['state'][k][key]) for k,state in actual['state'].items() for key,v in state.items()):raise ValueError('Adam restore differs')
    return dict(model_tensors=632,model_exact=True,adam_states=len(actual['state']),adam_exact=True,geometry_trainable_names=freeze(model))

def main():
    from scripts.train_dual_stream_v1 import cohorts,build_batch,MANIFEST,HELDOUT
    from scripts.dual_stream_training_policy import epoch_order,rank_microbatches
    from tokengs.options import config_defaults
    if any(p.exists() for p in (M,Q,R)):raise FileExistsError('New run already exists; inspect rather than duplicate')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Snapshot dirty')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();log=Path('/tmp/reciprocity_reconstruction_cpu_20261024.log').read_text()
    if 'Ran 17 tests' not in log or '\nOK\n' not in log:raise ValueError('CPU tests incomplete')
    torch.set_num_threads(4)
    source=OLD/'source_checkpoint.pt'
    if sha(source)!=SOURCE_SHA:raise ValueError('Baseline4 SHA differs')
    saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True);validate_source(saved)
    for path,key in ((MANIFEST,'manifest_sha256'),(HELDOUT,'heldout_sha256')):
        if sha(path)!=saved['meta'][key]:raise ValueError('Source data differs')
    restoration=model_restore_check(saved);del saved
    teacher=OLD/'teacher';manifest=json.loads((teacher/'export_complete.json').read_text())
    if manifest['source_sha256']!=SOURCE_SHA or manifest['windows']!=8337 or not manifest['teacher_versions_unchanged']:raise ValueError('Teacher provenance differs')
    windows,_,_=cohorts();official=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json');evaluation=json.loads(official.read_text())['records']
    for i,item in enumerate(manifest['records']):
        path=teacher/item['file']
        if item['window_index']!=i or item['scene']!=windows[i]['scene'] or sha(path)!=item['sha256']:raise ValueError('Teacher manifest/hash differs')
        rec=torch.load(path,map_location='cpu',weights_only=False)
        if rec['source_sha256']!=SOURCE_SHA or rec['window_index']!=i or rec['features'].shape[:2]!=rec['source_xyz_scale'].shape[:2] or rec['source_xyz_scale'].shape[-1]!=6 or rec['target'].shape!=(2,256,256):raise ValueError('Teacher shape/identity differs')
        if not all(torch.isfinite(rec[k]).all() for k in ('features','source_xyz_scale','cam_view','intrinsics')):raise ValueError('Teacher nonfinite')
    print('teacher all8337 hashes and shapes verified',flush=True)
    for rank in range(4):
        smoke=json.loads((teacher/f'GPU_GRADIENT_SMOKE_rank{rank:02d}.json').read_text())
        if smoke['status']!='PASSED' or not smoke['reconstruction_backward'] or smoke['attention_backend']['math_bias_calls']<=0:raise ValueError('Prior teacher GPU smoke absent')
    split=json.loads((PARENT/'scene_split.json').read_text());sets=[set(split[k]) for k in ('train','dev','official_eval')]
    if list(map(len,sets))!=[1072,119,312] or any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2))):raise ValueError('Scene split/leakage')
    if sets[0]|sets[1]!={w['scene'] for w in windows} or sets[2]!={w['scene'] for w in evaluation}:raise ValueError('Cohort scenes differ')
    renderer=verify_renderer(PIN_PATH)
    if shutil.disk_usage(BASE).free<180*1024**3:raise ValueError('Insufficient combined cache/export disk budget')
    b16=BASE/'baseline_extend16_20261023_v1/eval/source_checkpoint.pt'
    if sha(b16)!=BASELINE16_SHA:raise ValueError('Baseline16 SHA differs')
    saved16=torch.load(b16,map_location='cpu',weights_only=False,mmap=True)
    if saved16['progress']!=dict(epoch=16,update_in_epoch=0,full_updates=16688):raise ValueError('Baseline16 progress differs')
    restore16=model_restore_check(saved16);del saved16
    # Prediction-independent selection: one window per scene, low/high frame
    # displacement and pair-IoU extremes represented within each scene cohort.
    chosen=[]
    for label,allowed in [('train16',sets[0]),('internal16',sets[1])]:
        pool=[dict(w) for w in windows if w['scene'] in allowed]
        low=sorted(pool,key=lambda w:(abs(w['context'][1]-w['context'][0]),-(w.get('pair_iou') or 0),w['scene']))
        high=sorted(pool,key=lambda w:(-abs(w['context'][1]-w['context'][0]),w.get('pair_iou') or 0,w['scene']))
        selected=[];seen=set()
        for candidates in (low,high):
            count=0
            for w in candidates:
                if w['scene'] in seen:continue
                w=dict(w,diagnostic_split=label);selected.append(w);seen.add(w['scene']);count+=1
                if count==8:break
        if len(selected)!=16:raise ValueError('Small-set selection incomplete')
        chosen.extend(selected)
    opt=config_defaults['train_siu3r_unified_query_v1'];batch_evidence=[]
    for w in chosen:
        batch=build_batch(opt,w,torch.device('cpu'));w['input_rgb_sha256']=tensor_sha(batch['images_all'])
        if list(batch['images_input'].shape)!=[1,2,3,256,256] or batch['frame_ids'][0].tolist()!=w['context']+w['novel']:raise ValueError('Real batch differs')
        c2w=torch.linalg.inv(batch['cam_view_all'].transpose(-1,-2))[0]
        w['provider_context_baseline']=float((c2w[1,:3,3]-c2w[0,:3,3]).norm())
        batch_evidence.append(dict(name=w['scene'],split=w['diagnostic_split'],rgb_sha256=w['input_rgb_sha256'],gt_camera_sha256=tensor_sha(c2w),GT_seen_by_backbone=True))
    order=epoch_order(8337,4);first=[rank_microbatches(order,0,r,4) for r in range(4)]
    # Audit real first-update Context inputs against the reused teacher, CPU only.
    for ids in first:
        for index in ids:
            batch=build_batch(opt,windows[index],torch.device('cpu'));record=torch.load(teacher/manifest['records'][index]['file'],map_location='cpu',weights_only=False)
            if tensor_sha(batch['images_input'])!=record['rgb_sha256']:raise ValueError('Teacher first-update RGB changed')
    for run,src,source_sha in ((M,source,SOURCE_SHA),(Q,source,SOURCE_SHA),(R,b16,BASELINE16_SHA)):
        run.mkdir();os.link(src,run/'source_checkpoint.pt');(run/'source_checkpoint.sha256').write_text(source_sha+'  '+str(run/'source_checkpoint.pt')+'\n');(run/'code.sha').write_text(code+'\n');shutil.copyfile('/tmp/reciprocity_reconstruction_cpu_20261024.log',run/'cpu_tests.log')
        write(run/'CPU_TESTS.json',dict(status='PASSED',tests=17,gpu_verified=False))
    oldcfg=json.loads((OLD/'configuration.json').read_text());shutil.copyfile(OLD/'training_cohort.json',M/'training_cohort.json')
    (M/'teacher').symlink_to(teacher,target_is_directory=True)
    mcfg=dict(oldcfg,teacher_renderer=dict(path='/space/mawb/tokengs_siu3r_joint_v1/.torch_extensions_v4/gsplat_cuda/gsplat_cuda.so',sha256=sha('/space/mawb/tokengs_siu3r_joint_v1/.torch_extensions_v4/gsplat_cuda/gsplat_cuda.so')),code_sha=code,code_snapshot=str(ROOT),retry_of=str(OLD),teacher_read_only_reuse=str(teacher),teacher_manifest_sha256=sha(teacher/'export_complete.json'),
        gate='Four-rank real DDP, two microbatches, one actual Adam step per arm, throw away then restore Baseline4 anew',gate_updates_not_in_formal1043=True)
    write(M/'configuration.json',mcfg);write(M/'TEACHER_REUSE_AUDIT.json',dict(status='PASSED',windows=8337,all_hashes_shapes_verified=True,source_sha256=SOURCE_SHA,teacher_manifest_sha256=sha(teacher/'export_complete.json'),prior_GPU_smoke_reused=True,new_DDP_gate_pending=True))
    shutil.copyfile(PARENT/'scene_split.json',Q/'scene_split.json');write(Q/'cache_cohort.json',dict(records=windows+evaluation))
    qcfg=dict(code_sha=code,code_snapshot=str(ROOT),source_sha256=SOURCE_SHA,official_cohort=str(official),official_cohort_sha256=sha(official),scene_split_sha256=sha(Q/'scene_split.json'),
        cache_cohort_sha256=sha(Q/'cache_cohort.json'),renderer_path=PIN_PATH,renderer_sha256=PIN_SHA,seed=31415,trained_arms=['A','B','C','D'],historical_tokens_targets_reused=False,
        geometry_features_gt_free=True,features=FEATURE_NAMES,world_size=4,lr=.001,batch_size=1024,max_epochs=100,patience=12,calibration_gamma=.25,formal_selection='dev only',target='original Context class-agnostic binary IoU unchanged across all four heads',
        D='Exact cross-scene whole-descriptor permutation separately within train/dev/official, no labels for shuffling; cohort negative control')
    write(Q/'configuration.json',qcfg)
    write(R/'diagnostic_cohort.json',dict(records=chosen,selection='8 smallest and 8 largest Context frame displacement per split, distinct scenes; pairIoU tiebreak',backbone_seen_all=True,formal1860_used_for_selection=False))
    write(R/'configuration.json',dict(code_sha=code,code_snapshot=str(ROOT),source_sha256=BASELINE16_SHA,cohort_sha256=sha(R/'diagnostic_cohort.json'),renderer=renderer,
        train_windows=16,internal_windows=16,internal_scenes_seen_by_backbone=True,appearance_updates=100,appearance_lr=.01,appearance_parameters='Gaussian RGB only',appearance_target='Context RGB only',
        camera='GT pose+intrinsics after orientation-constrained two-Context Sim3; reject degenerate or residual threshold failure',LPIPS_pilot_submitted=False))
    write(R/'INPUT_AUDIT.json',dict(records=batch_evidence))
    evidence=dict(status='PASSED',code_sha=code,source4=restoration,source16=restore16,teacher_all8337_verified=True,scene_overlap=0,renderer=renderer,
        real_smallset_batches=32,real_teacher_first_update_batches=8,first_update_indices_by_rank=first,gpu_verified=False,cpu_tests=17,official_cohort_sha256=sha(official))
    for run in (M,Q,R):write(run/'PREFLIGHT.json',evidence)
    print(json.dumps(dict(runs=list(map(str,(M,Q,R))),preflight=evidence),indent=2),flush=True)
if __name__=='__main__':main()
