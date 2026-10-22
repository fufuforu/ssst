"""Prepare exactly two authorized one-factor controls; never submit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,json,os,gc,shutil
from torch import nn
from types import SimpleNamespace
from tokengs.options import config_defaults
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.v4init_isolated_noca import V4InitIsolatedNoCACore
from scripts.v4init_isolated_recipe import ARCH,KEY,LOW_ARCH,LOW_KEY,VIEWS_ARCH,VIEWS_KEY,recipe_identity,initialize_from_v4init_zero,optimizer_groups
from scripts.olda4_ablation_pipeline import digest,write
from scripts.train_dual_stream_v1 import cohorts,MANIFEST,HELDOUT,build_batch
from scripts.training_windows_2ctx4novel import DEFAULT_MANIFEST,load_training_windows
from scripts.smoke_v4init_weight_views import validate_batch_contract
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_weight_views_20261022_v1')
SOURCE=RUN.parent/'oldA4_v4init_noca_ablation_20261022_v1/V4init/train/checkpoint_initial.pt'
COHORT=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json')
BASELINE=RUN.parent/'v4init_isolated_noca_v4recipe_20261022_v1'
ARMS=[('U005',LOW_ARCH,LOW_KEY,2),('Views2ctx4',VIEWS_ARCH,VIEWS_KEY,4)]
SOURCE_SHA='bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397'

def main():
    if RUN.exists():raise ValueError('Refusing duplicate preparation')
    assert digest(SOURCE)==SOURCE_SHA
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True);cfg=UnifiedQueryConfig(**saved['meta']['config'])
    base_cfg=json.loads((BASELINE/'train/configuration.json').read_text())
    assert base_cfg['initial_checkpoint_sha256']==SOURCE_SHA and base_cfg['code_sha']=='310d49d04b6ac1feda55439bb21b6c88a8fc5882'
    windows,_,_=cohorts();assert len(windows)==8337 and len({w['scene'] for w in windows})==1191
    assert all(len(w['context'])==len(w['novel'])==2 for w in windows)
    extended,extension=load_training_windows(DEFAULT_MANIFEST,windows,digest(MANIFEST));assert len(extended)==8337
    # Verify the real decoder batches, not only manifest bookkeeping.
    from scripts.dual_stream_training_policy import epoch_order
    index=epoch_order(len(windows),0)[0];opt=config_defaults['train_siu3r_unified_query_v1']
    four=build_batch(opt,windows[index],torch.device('cpu'));six=build_batch(opt.evolve(num_views=6),extended[index],torch.device('cpu'))
    contracts={'U005':validate_batch_contract(four,windows[index]),'Views2ctx4':validate_batch_contract(six,extended[index])}
    for key in ('images_input',):assert torch.equal(four[key],six[key]),key
    for key in ('images_all','semantic_label_all','instance_label_all','masks_all'):
        for old_i,frame in enumerate(four['frame_ids'][0].tolist()):
            new_i=six['frame_ids'][0].tolist().index(frame)
            assert torch.equal(four[key][:,old_i],six[key][:,new_i]),(key,frame)
    assert contracts['Views2ctx4']['supervised_views']==6
    cohort=json.loads(COHORT.read_text())['records'];assert len(cohort)==1860 and len({w['scene'] for w in cohort})==312
    RUN.mkdir(parents=True);manifest=RUN/'training_windows_2ctx4novel.json';shutil.copyfile(DEFAULT_MANIFEST,manifest)
    extended,extension=load_training_windows(manifest,windows,digest(MANIFEST))
    audits={};tensor_signatures=[]
    for label,arch,key,nv in ARMS:
        torch.manual_seed(42);model=nn.Module();model.architecture_name=arch;model.config=cfg
        model.frozen_omega=SimpleNamespace(source_identity=saved['meta']['omega_identity'])
        model.sam_proposals=SimpleNamespace(source_identity=saved['meta']['sam_identity'])
        model.gfm_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256))
        model.core=V4InitIsolatedNoCACore(opt.evolve(num_views=2+nv),cfg)
        hashes=dict(manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),world=8)
        evidence=initialize_from_v4init_zero(saved,model,**hashes)
        assert tuple(len(evidence[k]) for k in ('copied_keys','removed_keys','new_keys'))==(632,32,0)
        state=model.state_dict();assert all(torch.equal(state[k],saved['model'][k]) for k in evidence['copied_keys'])
        assert not hasattr(model.core,'understanding_adapters') and not hasattr(model.core.initializer,'reconstruction')
        assert all(not hasattr(l,'anchor_attention') and hasattr(l,'image_attention') for l in model.core.panoptic.layers.values())
        for case in ('trained','adam','data','rng','extra','missing_retained','removed_shape','initial_provenance'):
            bad=dict(saved,meta=dict(saved['meta']),model=dict(saved['model']),progress=dict(saved['progress']),optimizer=dict(saved['optimizer']))
            if case=='trained':bad['progress']['full_updates']=1
            if case=='adam':bad['optimizer']['state']={0:{'step':1}}
            if case=='data':bad['meta']['manifest_sha256']='wrong'
            if case=='rng':bad['rank_rng']=bad['rank_rng'][:1]
            if case=='extra':bad['model']['unexpected']=torch.zeros(1)
            if case=='missing_retained':del bad['model'][evidence['copied_keys'][0]]
            if case=='removed_shape':bad['model'][evidence['removed_keys'][0]]=torch.zeros(1)
            if case=='initial_provenance':bad['meta']['initial_sha256']='wrong'
            try:initialize_from_v4init_zero(bad,model,**hashes)
            except ValueError:pass
            else:raise AssertionError('Invalid migration accepted: '+case)
        groups=optimizer_groups(model);ids=[id(p) for g in groups for p in g['params']];assert len(ids)==len(set(ids))==len(list(model.parameters()))
        geom=[n for g in groups if g['family']=='geometry' for n in g['parameter_names']];assert len(geom)==48
        changes={k for k,v in recipe_identity().items() if recipe_identity(key)[k]!=v}
        assert changes==({'understanding_weight'} if nv==2 else {'training_view_protocol'})
        audit=dict(status='PASSED',source_sha256=SOURCE_SHA,source_updates=0,empty_adam=True,retained_tensors=632,removed_tensors=32,new_tensors=0,
            retained_tensors_bitwise_equal=True,invalid_migration_cases_rejected=8,geometry_parameter_tensors=geom,
            optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in groups],trainable_parameters=sum(p.numel() for p in model.parameters()),
            recipe_changed_fields=sorted(changes),real_cpu_batch_contract=contracts[label],original_views_rgb_and_gt_bitwise_preserved=True,migration=evidence,gpu_verified=False)
        audits[label]=audit;branch=RUN/label;branch.mkdir()
        for stage in ('smoke','train','eval'):
            r=branch/stage;r.mkdir()
            config=dict(recipe=f'V4init_isolated_{label}_full4_official1860',branch=label,architecture=arch,model_key=key,code_snapshot=str(ROOT),
                training_root=str(branch/'train'),smoke_root=str(branch/'smoke'),checkpoint_epoch=4,epochs=4,full_updates=4172,global_batch=8,world_size=8,
                learning_rate=1e-4,lr_warmup=100,lr_floor=1e-5,geometry_lr_floor=1e-6,clip=1.,context_views=2,novel_views=nv,total_supervised_views=2+nv,
                manifest=str(MANIFEST),manifest_sha256=hashes['manifest_sha256'],heldout_sha256=hashes['heldout_sha256'],cohort=str(COHORT),cohort_sha256=digest(COHORT),
                initial_checkpoint=str(r/'checkpoint_untrained_initial.pt'),initial_checkpoint_sha256=SOURCE_SHA,source_v4init_zero=str(SOURCE),
                original_A4_zero_sha256=saved['meta']['initial_sha256'],checkpoint_every=50,epoch_probes=True,baseline_root=str(BASELINE),baseline_training_job=71001,baseline_eval_job=71002,
                official_commit='8ea80166be76854f938e90521f1a5b688b755c87',**recipe_identity(key))
            if nv==4:config.update(extension)
            write(r/'configuration.json',config);write(r/'model_config.json',cfg.to_dict());write(r/'PREFLIGHT.json',audit)
            if stage!='eval':os.link(SOURCE,r/'checkpoint_untrained_initial.pt');(r/'untrained_initial.sha256').write_text(SOURCE_SHA+'\n')
        del model,state,groups;gc.collect()
    write(RUN/'CPU_VALIDATION.json',dict(status='PASSED',source_sha256=SOURCE_SHA,arms=audits,training_windows=8337,training_scenes=1191,
        views_manifest_sha256=digest(manifest),all_8337_parent_contexts_and_original_novel_preserved=True,global_batch=8,updates=4172,gpu_verified=False))
    print(json.dumps(dict(status='PASSED',run=str(RUN),arms={k:{x:v for x,v in a.items() if x not in ('migration','optimizer_groups','geometry_parameter_tensors')} for k,a in audits.items()})),flush=True)
if __name__=='__main__':main()
