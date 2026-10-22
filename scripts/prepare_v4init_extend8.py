"""Audit each live source ONCE and prepare afterok continuations; no polling."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,json,os,gc,hashlib,copy
from torch import nn
from types import SimpleNamespace
from tokengs.options import config_defaults
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.v4init_isolated_noca import V4InitIsolatedNoCACore
from scripts.v4init_isolated_recipe import optimizer_groups
from scripts.v4init_extend8_policy import POLICY,START,END,validate_source,tensor_tree_equal,transition_meta,source_parameter_digest
from scripts.olda4_ablation_pipeline import read,write,digest
from scripts.train_dual_stream_v1 import cohorts,MANIFEST,HELDOUT,build_batch
BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
RUN=BASE/'v4init_isolated_extend8_20261022_v1'
SOURCES=[('Baseline',BASE/'v4init_isolated_noca_v4recipe_20261022_v1',71001,71002,'310d49d04b6ac1feda55439bb21b6c88a8fc5882'),
 ('U005',BASE/'v4init_isolated_weight_views_20261022_v1/U005',71009,71010,'ef98a6f9d8ffdba3c39c5cb4660eb65726bd1f2e'),
 ('Views2ctx4',BASE/'v4init_isolated_weight_views_20261022_v1/Views2ctx4',71012,71013,'ef98a6f9d8ffdba3c39c5cb4660eb65726bd1f2e')]

def main():
    torch.set_num_threads(4)
    if RUN.exists():raise ValueError('Refusing duplicate preparation')
    RUN.mkdir(parents=True);audits={};windows,_,_=cohorts();assert len(windows)==8337 and len({w['scene'] for w in windows})==1191
    for label,source,job,eval_job,source_code in SOURCES:
        branch=RUN/label;branch.mkdir();source_cfg=read(source/'train/configuration.json')
        assert source_cfg['code_sha']==source_code and read(source/'train/SUBMITTED.json')['job']==job
        assert read(source/'eval/SUBMITTED.json')['job']==eval_job
        assert digest(MANIFEST)==source_cfg['manifest_sha256'] and digest(HELDOUT)==source_cfg['heldout_sha256']
        pinned=branch/'preflight_observed_source.pt';os.link(source/'train/checkpoint_latest.pt',pinned)
        saved=torch.load(pinned,map_location='cpu',weights_only=False,mmap=True)
        cfg=UnifiedQueryConfig(**read(source/'train/model_config.json'));torch.manual_seed(42)
        model=nn.Module();model.architecture_name=source_cfg['architecture'];model.config=cfg
        model.frozen_omega=SimpleNamespace(source_identity=saved['meta']['omega_identity']);model.sam_proposals=SimpleNamespace(source_identity=saved['meta']['sam_identity'])
        model.gfm_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256));opt=config_defaults['train_siu3r_unified_query_v1']
        nv=4 if label=='Views2ctx4' else 2
        model.core=V4InitIsolatedNoCACore(opt.evolve(num_views=2+nv),cfg)
        specs=optimizer_groups(model);optimizer=torch.optim.AdamW(specs,betas=(.9,.95),eps=1e-8)
        evidence=validate_source(saved,source_cfg,model,optimizer,require_complete=False)
        target=model.state_dict()
        with torch.no_grad():
            for k,v in saved['model'].items():target[k].copy_(v)
        assert all(torch.equal(target[k],v) for k,v in saved['model'].items())
        optimizer.load_state_dict(saved['optimizer'])
        assert tensor_tree_equal(optimizer.state_dict(),saved['optimizer'])
        # Exercise the SAME metadata transition used by every training rank.
        current=dict(saved['meta'],code_sha='PREPARATION_PLACEHOLDER',epochs=8,learning_rate=3e-5)
        transition_meta(saved['meta'],current,optimizer,checkpoint=str(pinned),sha256=digest(pinned))
        # Original source optimizer moment/step bytes are preserved after LR rebasing.
        assert tensor_tree_equal(optimizer.state_dict()['state'],saved['optimizer']['state'])
        layout=dict(model={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in saved['model'].items()},
            optimizer_groups=[{k:(list(v) if isinstance(v,tuple) else v) for k,v in g.items()} for g in saved['optimizer']['param_groups']])
        assert len(layout['model'])==632
        audit=dict(evidence,source_checkpoint_observed_once=str(source/'train/checkpoint_latest.pt'),
            observed_checkpoint_sha256=digest(pinned),model_tensor_count=632,all_model_tensors_restored_exact=True,
            all_adam_moments_and_steps_restored_exact=True,post_rebase_adam_moments_and_steps_unchanged=True,
            source_final_sha_deferred_until_afterok=True,source_job=job,source_code_sha=source_code,gpu_continuation_verified=False)
        # Validate a real epoch5 batch from the unchanged deterministic order.
        from scripts.dual_stream_training_policy import epoch_order
        train_windows=windows
        if nv==4:
            from scripts.training_windows_2ctx4novel import load_training_windows
            train_windows,identity=load_training_windows(source_cfg['training_window_manifest'],windows,digest(MANIFEST))
            assert identity['training_window_manifest_sha256']==source_cfg['training_window_manifest_sha256']
        first=epoch_order(len(train_windows),4)[0];batch=build_batch(opt.evolve(num_views=2+nv),train_windows[first],torch.device('cpu'))
        assert list(batch['images_input'].shape)==[1,2,3,256,256] and list(batch['images_all'].shape)==[1,2+nv,3,256,256]
        assert batch['frame_ids'][0].tolist()==train_windows[first]['context']+train_windows[first]['novel']
        audit['first_continued_window_index']=first;audit['real_epoch5_batch_shapes']={k:list(batch[k].shape) for k in ('images_input','images_all','images_output','frame_ids')}
        audits[label]=audit
        for stage in ('train','eval'):
            r=branch/stage;r.mkdir();config=dict(source_cfg)
            config.update(recipe=f'V4init_{label}_epoch4_to8_Adam_preserved',branch=label,code_snapshot=str(ROOT),
                source_training_root=str(source/'train'),source_eval_root=str(source/'eval'),source_code_sha=source_code,
                source_train_job=job,source_eval_job=eval_job,training_root=str(branch/'train'),epochs=8,checkpoint_epoch=8,
                full_updates=END,added_updates=END-START,learning_rate=3e-5,geometry_refinement_lr=3e-6,
                continuation_policy=POLICY,continuation_sha256=None,continuation_start_updates=START,
                continuation_added_updates=END-START,lr_warmup=100,lr_floor=3e-6,geometry_lr_floor=3e-7,
                optimizer_reset=False,understanding_warmup_restarted=False,code_sha=None)
            config.pop('smoke_root',None)
            write(r/'configuration.json',config);write(r/'source_configuration.json',source_cfg);write(r/'model_config.json',cfg.to_dict());write(r/'SOURCE_LAYOUT.json',layout);write(r/'PREFLIGHT.json',audit)
        print(label,json.dumps(audit),flush=True)
        del saved,model,target,optimizer,specs,layout,batch;gc.collect();pinned.unlink()
    write(RUN/'CPU_SOURCE_VALIDATION.json',dict(status='PASSED',source_observations=3,arms=audits,source_final_sha_deferred_until_afterok=True,training_windows=8337,training_scenes=1191,source_epoch=4,target_epoch=8,source_updates=START,added_updates=END-START,full_updates=END,gpu_verified=False))
    print('ALL THREE SOURCE AUDITS PASSED',flush=True)
if __name__=='__main__':main()
