"""Prepare three independent recovery directories and audit real model/Adam restoration."""
from pathlib import Path
import sys,os,json,gc,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from tokengs.options import config_defaults
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.v4init_isolated_noca import V4InitIsolatedNoCACore
from scripts.v4init_isolated_recipe import optimizer_groups,check_meta as check_original
from scripts.v4init_extend8_policy import tensor_tree_equal,validate_source,START,END
from scripts.v4init_recovery_policy import recovery_meta
from scripts.olda4_ablation_pipeline import read,write,digest
from scripts.train_dual_stream_v1 import cohorts,MANIFEST,HELDOUT,build_batch
from scripts.dual_stream_training_policy import epoch_order,rank_microbatches
BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
OLD=BASE/'v4init_isolated_extend8_20261022_v1'
RUN=BASE/'v4init_isolated_extend8_recovery_20261023_v1'
SOURCE_CODE='ce8446d102ac1d9a7347c8525ec38935d4922de3'


def link_once(source,target):
    if target.exists():assert os.path.samefile(source,target)
    else:os.link(source,target)


def main():
    torch.set_num_threads(4)
    RUN.mkdir(exist_ok=True)
    assert not (RUN/'SUBMISSION_REQUESTED.json').exists(),'Refuse preparing submitted run'
    audits={};windows,_,_=cohorts()
    assert len(windows)==8337 and len({w['scene'] for w in windows})==1191
    for label in ['Baseline','U005','Views2ctx4']:
        old=OLD/label/'train';cfg=read(old/'configuration.json');original_root=Path(cfg['source_training_root'])
        original_cfg=read(old/'source_configuration.json');original_path=original_root/'checkpoint_epoch4.pt'
        original=torch.load(original_path,map_location='cpu',weights_only=False,mmap=True)
        assert original['progress']==dict(epoch=4,update_in_epoch=0,full_updates=START)
        check_original(original['meta'],original_cfg)
        source4_sha=digest(original_path)
        assert source4_sha==read(Path(cfg['source_eval_root'])/'FINAL_CHECKPOINT_FROZEN.json')['sha256']
        branch=RUN/label
        for stage in ['train','eval']:(branch/stage).mkdir(parents=True,exist_ok=True)
        train=branch/'train';assert not (train/'STARTED.json').exists()
        mode='continue' if label=='Views2ctx4' else 'recover'
        newcfg=dict(cfg,code_snapshot=str(ROOT),code_sha='PREFLIGHT',training_root=str(train),restart_mode=mode,
                    continuation_sha256=source4_sha,previous_failed_training_root=str(old),checkpoint_retention='initial+latest+final_only; no intermediate epoch checkpoints')
        model_cfg=UnifiedQueryConfig(**read(old/'model_config.json'));torch.manual_seed(42)
        model=nn.Module();model.gfm_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256))
        opt=config_defaults['train_siu3r_unified_query_v1'];nv=4 if label=='Views2ctx4' else 2
        model.core=V4InitIsolatedNoCACore(opt.evolve(num_views=2+nv),model_cfg)
        optimizer=torch.optim.AdamW(optimizer_groups(model),betas=(.9,.95),eps=1e-8)
        write(train/'source_epoch4_meta.json',original['meta'])
        newcfg['source_epoch4_meta_sha256']=digest(train/'source_epoch4_meta.json')
        if mode=='recover':
            source=old/'checkpoint_latest.pt';frozen=train/'checkpoint_recovery_source.pt';link_once(source,frozen)
            saved=torch.load(frozen,map_location='cpu',weights_only=False,mmap=True)
            step=5500 if label=='Baseline' else START
            assert saved['progress']==dict(epoch=step//1043,update_in_epoch=step%1043,full_updates=step)
            newcfg.update(recovery_source=str(source),recovery_sha256=digest(frozen),recovery_source_code_sha=SOURCE_CODE,recovery_progress=dict(saved['progress']))
            current=dict(copy.deepcopy(original['meta']),code_sha='PREFLIGHT',epochs=8,learning_rate=3e-5)
            actual=recovery_meta(saved,newcfg,read(train/'source_epoch4_meta.json'),current,optimizer,model)
            assert actual==dict(saved['meta'],code_sha='PREFLIGHT')
        else:
            saved=original;source=original_path
            validate_source(saved,original_cfg,model,optimizer)
            newcfg['recovery_progress']=dict(saved['progress'])
        target=model.state_dict()
        with torch.no_grad():
            for k,v in saved['model'].items():target[k].copy_(v)
        assert all(torch.equal(target[k],v) for k,v in saved['model'].items())
        optimizer.load_state_dict(saved['optimizer']);assert tensor_tree_equal(optimizer.state_dict(),saved['optimizer'])
        train_windows=windows
        if nv==4:
            from scripts.training_windows_2ctx4novel import load_training_windows
            train_windows,identity=load_training_windows(cfg['training_window_manifest'],windows,digest(MANIFEST))
            assert identity['training_window_manifest_sha256']==cfg['training_window_manifest_sha256']
        assert digest(MANIFEST)==cfg['manifest_sha256'] and digest(HELDOUT)==cfg['heldout_sha256']
        progress=saved['progress'];order=epoch_order(len(windows),progress['epoch'])
        indices=[rank_microbatches(order,progress['update_in_epoch'],rank,8)[0] for rank in range(8)]
        batch=build_batch(opt.evolve(num_views=2+nv),train_windows[indices[0]],torch.device('cpu'))
        assert list(batch['images_input'].shape)==[1,2,3,256,256] and list(batch['images_all'].shape)==[1,2+nv,3,256,256]
        audit=dict(status='PASSED',restart_mode=mode,source=str(source),source_sha256=newcfg.get('recovery_sha256',source4_sha),original_epoch4_sha256=source4_sha,progress=progress,remaining_updates=END-progress['full_updates'],model_tensors=len(saved['model']),adam_states=len(saved['optimizer']['state']),adam_steps=sorted({float(s['step']) for s in saved['optimizer']['state'].values()}),rank_rng_count=len(saved['rank_rng']),all_model_tensors_restored_exact=True,all_adam_moments_and_steps_restored_exact=True,next_rank_indices=indices,next_batch_shapes={k:list(batch[k].shape) for k in ['images_input','images_all','frame_ids']},gpu_first_update_verified=False)
        audits[label]=audit
        for stage in ['train','eval']:
            root=branch/stage;ecfg=dict(newcfg)
            write(root/'configuration.json',ecfg);write(root/'source_configuration.json',original_cfg);write(root/'model_config.json',model_cfg.to_dict())
            write(root/'SOURCE_LAYOUT.json',read(old/'SOURCE_LAYOUT.json'));write(root/'PREFLIGHT.json',audit)
        print(label,json.dumps(audit),flush=True)
        del saved,original,model,optimizer,target,batch;gc.collect()
    available=os.statvfs(RUN);free=available.f_bavail*available.f_frsize
    # Three initial+latest states, atomic-save headroom and three official exports.
    budget=32*2**30
    assert free>budget,('Insufficient capacity',free,budget)
    write(RUN/'CAPACITY_PLAN.json',dict(available_bytes=free,planned_additional_upper_budget_bytes=budget,checkpoint_retention='initial+latest+atomic-temp+final-hardlink',intermediate_epochs_retained=False,evaluation_exports_included=True,other_jobs_not_reserved=True))
    identity={}
    for relative in ['tokengs/models/v4init_isolated_noca.py','tokengs/models/superpoint_structural_variants.py','tokengs/models/unified_query_v1.py','tokengs/models/unified_query_loss.py','scripts/v4init_extend8_policy.py','scripts/v4init_isolated_recipe.py','scripts/dual_stream_training_policy.py','scripts/training_windows_2ctx4novel.py','scripts/export_unified_query_checkpoint.py','scripts/score_dual_stream_v1.py']:
        oldfile=Path('/space/mawb/ssst_v4init_extend8_code_20261022_v1')/relative
        assert (ROOT/relative).read_bytes()==oldfile.read_bytes();identity[relative]=digest(ROOT/relative)
    write(RUN/'MODEL_RECIPE_SOURCE_IDENTITY.json',identity)
    write(RUN/'CPU_SOURCE_VALIDATION.json',dict(status='PASSED',arms=audits,source_execution_code=SOURCE_CODE,no_gpu_validation_claim=True))


if __name__=='__main__':main()
