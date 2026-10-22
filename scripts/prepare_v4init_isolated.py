"""Audit the exact V4init zero, then prepare one independent smoke/train/eval chain."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,json,os,gc
from torch import nn
from types import SimpleNamespace
from tokengs.options import config_defaults
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.v4init_isolated_noca import V4InitIsolatedNoCACore
from scripts.v4init_isolated_recipe import ARCH,KEY,recipe_identity,initialize_from_v4init_zero,optimizer_groups
from scripts.olda4_ablation_pipeline import digest,write
from scripts.train_dual_stream_v1 import cohorts,MANIFEST,HELDOUT
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_noca_v4recipe_20261022_v1')
SOURCE=RUN.parent/'oldA4_v4init_noca_ablation_20261022_v1/V4init/train/checkpoint_initial.pt'
COHORT=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json')

def main():
    if RUN.exists():raise ValueError('Refusing duplicate preparation')
    source_sha=digest(SOURCE);saved=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True)
    cfg=UnifiedQueryConfig(**saved['meta']['config']);torch.manual_seed(42)
    model=nn.Module();model.architecture_name=ARCH;model.config=cfg
    model.frozen_omega=SimpleNamespace(source_identity=saved['meta']['omega_identity'])
    model.sam_proposals=SimpleNamespace(source_identity=saved['meta']['sam_identity'])
    model.gfm_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256))
    model.core=V4InitIsolatedNoCACore(config_defaults['train_siu3r_unified_query_v1'],cfg)
    hashes=dict(manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),world=8)
    evidence=initialize_from_v4init_zero(saved,model,**hashes)
    counts=tuple(len(evidence[k]) for k in ('copied_keys','removed_keys','new_keys'));assert counts==(632,32,0),counts
    state=model.state_dict();assert all(torch.equal(state[k],saved['model'][k]) for k in evidence['copied_keys'])
    initial_keys=[k for k in state if k.startswith('core.v4_initialization.')];assert len(initial_keys)==9
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
    windows,_,_=cohorts();assert len(windows)==8337 and len({w['scene'] for w in windows})==1191
    assert all(len(w['context'])==len(w['novel'])==2 for w in windows)
    cohort=json.loads(COHORT.read_text())['records'];assert len(cohort)==1860 and len({w['scene'] for w in cohort})==312
    groups=optimizer_groups(model);ids=[id(p) for g in groups for p in g['params']];assert len(ids)==len(set(ids))==len(list(model.parameters()))
    geometry_names=[n for g in groups if g['family']=='geometry' for n in g['parameter_names']]
    assert len(geometry_names)==48,len(geometry_names)
    audit=dict(status='PASSED',source_checkpoint=str(SOURCE),source_sha256=source_sha,source_updates=0,empty_adam=True,
        retained_tensors=632,removed_tensors=32,new_tensors=0,retained_tensors_bitwise_equal=True,
        v4_initialization_tensors_preserved=initial_keys,invalid_migration_cases_rejected=8,
        geometry_parameter_tensors=geometry_names,optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in groups],
        trainable_parameters=sum(p.numel() for p in model.parameters()),migration=evidence,gpu_verified=False)
    RUN.mkdir(parents=True)
    for stage in ('smoke','train','eval'):
        r=RUN/stage;r.mkdir()
        config=dict(recipe='V4init_NoCA_isolated_earlyV4_full4_official1860',branch='V4init_NoCA_isolated_V4recipe',architecture=ARCH,model_key=KEY,code_snapshot=str(ROOT),
            training_root=str(RUN/'train'),smoke_root=str(RUN/'smoke'),checkpoint_epoch=4,epochs=4,full_updates=4172,global_batch=8,world_size=8,
            learning_rate=1e-4,lr_warmup=100,lr_floor=1e-5,geometry_lr_floor=1e-6,clip=1.,
            manifest=str(MANIFEST),manifest_sha256=hashes['manifest_sha256'],heldout_sha256=hashes['heldout_sha256'],cohort=str(COHORT),cohort_sha256=digest(COHORT),
            initial_checkpoint=str(r/'checkpoint_untrained_initial.pt'),initial_checkpoint_sha256=source_sha,source_v4init_zero=str(SOURCE),
            original_A4_zero_sha256=saved['meta']['initial_sha256'],checkpoint_every=50,epoch_probes=True,
            official_commit='8ea80166be76854f938e90521f1a5b688b755c87',**recipe_identity())
        write(r/'configuration.json',config);write(r/'model_config.json',cfg.to_dict());write(r/'PREFLIGHT.json',audit)
        if stage!='eval':os.link(SOURCE,r/'checkpoint_untrained_initial.pt');(r/'untrained_initial.sha256').write_text(source_sha+'\n')
    write(RUN/'CPU_VALIDATION.json',dict(audit,training_windows=8337,training_scenes=1191,view_protocol='2context+2novel',global_batch=8,updates=4172))
    print(json.dumps({k:v for k,v in audit.items() if k not in ('migration','optimizer_groups','geometry_parameter_tensors')}),flush=True)
if __name__=='__main__':main()
