"""Prepare independent old A4 V4-init / No-CA runs; no job submission."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import torch
from torch import nn
import gc,json,os,unittest,copy
from types import SimpleNamespace
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.olda4_initialization_ablation import OldA4V4InitCore,OldA4NoCACore,V4_ARCH,NOCA_ARCH
from tokengs.options import config_defaults
from scripts.olda4_ablation_initial_state import initialize_olda4_ablation,recipe_identity
from scripts.olda4_ablation_pipeline import digest,write
from scripts.train_dual_stream_v1 import cohorts,MANIFEST,HELDOUT

RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/oldA4_v4init_noca_ablation_20261022_v1')
SOURCE=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_untrained_initial.pt')
SOURCE_SHA='3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744'
COHORT=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json')


def main():
    assert not RUN.exists(),'Refuse duplicate preparation'
    suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_olda4_ablation')
    result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful()
    assert digest(SOURCE)==SOURCE_SHA
    blob=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True);cfg=UnifiedQueryConfig(**blob['meta']['config'])
    windows,_,_=cohorts();assert len(windows)==8337 and len({w['scene'] for w in windows})==1191 and all(len(w['context'])==len(w['novel'])==2 for w in windows)
    assert len(json.loads(COHORT.read_text())['records'])==1860
    RUN.mkdir(parents=True);audits={}
    for label,core,arch,key,expected in [('V4init',OldA4V4InitCore,V4_ARCH,'dual_olda4_v4init',(655,38,9)),('NoCA',OldA4NoCACore,NOCA_ARCH,'dual_olda4_noca',(661,32,0))]:
        torch.manual_seed(42);model=nn.Module();model.architecture_name=arch;model.config=cfg
        model.frozen_omega=SimpleNamespace(source_identity=blob['meta']['omega_identity']);model.sam_proposals=SimpleNamespace(source_identity=blob['meta']['sam_identity'])
        model.gfm_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256));model.core=core(config_defaults['train_siu3r_unified_query_v1'],cfg)
        evidence=initialize_olda4_ablation(blob,model,manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),world=8)
        counts=tuple(len(evidence[k]) for k in ('copied_keys','removed_keys','new_keys'));assert counts==expected,counts
        state=model.state_dict();assert all(torch.equal(state[k],blob['model'][k]) for k in evidence['copied_keys'])
        assert not hasattr(model.core,'understanding_adapters')
        assert all(hasattr(l,'image_attention') and hasattr(l,'anchor_attention')==(label=='V4init') for l in model.core.panoptic.layers.values())
        if label=='NoCA':assert hasattr(model.core.initializer,'reconstruction')
        else:assert not hasattr(model.core.initializer,'reconstruction')
        # A trained endpoint, wrong data, or arbitrary missing/extra tensors must fail explicitly.
        for case in ('trained','adam','data','rng','extra','missing_retained','removed_shape'):
            bad=dict(blob,meta=dict(blob['meta']),model=dict(blob['model']),progress=dict(blob['progress']),optimizer=dict(blob['optimizer']))
            if case=='trained':bad['progress']['full_updates']=1
            if case=='adam':bad['optimizer']['state']={0:{'step':1}}
            if case=='data':bad['meta']['manifest_sha256']='wrong'
            if case=='rng':bad['rank_rng']=bad['rank_rng'][:1]
            if case=='extra':bad['model']['unexpected']=torch.zeros(1)
            if case=='missing_retained':del bad['model'][evidence['copied_keys'][0]]
            if case=='removed_shape':bad['model'][evidence['removed_keys'][0]]=torch.zeros(1)
            try:initialize_olda4_ablation(bad,model,manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),world=8)
            except ValueError:pass
            else:raise AssertionError('Invalid migration accepted: '+case)
        audits[label]=dict(status='PASSED',retained_tensors_bitwise_equal=True,retained_tensors=counts[0],removed_tensors=counts[1],new_tensors=counts[2],
            trainable_parameters=sum(p.numel() for p in model.parameters()),invalid_migration_cases_rejected=7,migration=evidence,gpu_verified=False)
        branch=RUN/label;branch.mkdir()
        for stage in ('smoke','train','eval'):
            r=branch/stage;r.mkdir()
            config=dict(recipe=f'old_A4_{label}_joint_full4_official1860',branch=label,architecture=arch,model_key=key,code_snapshot=str(ROOT),
                training_root=str(branch/'train'),smoke_root=str(branch/'smoke'),checkpoint_epoch=4,epochs=4,full_updates=4172,global_batch=8,world_size=8,
                learning_rate=1e-4,lr_warmup=100,lr_floor=1e-5,understanding_weight=.05,understanding_warmup=1043,clip=1.,
                manifest=str(MANIFEST),manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),cohort=str(COHORT),cohort_sha256=digest(COHORT),
                initial_checkpoint=str(r/'checkpoint_untrained_initial.pt'),initial_checkpoint_sha256=SOURCE_SHA,checkpoint_every=50,epoch_probes=True,
                official_commit='8ea80166be76854f938e90521f1a5b688b755c87',**recipe_identity(key))
            write(r/'configuration.json',config);write(r/'model_config.json',cfg.to_dict())
            if stage!='eval':os.link(SOURCE,r/'checkpoint_untrained_initial.pt');(r/'untrained_initial.sha256').write_text(SOURCE_SHA+'\n')
            write(r/'PREFLIGHT.json',audits[label])
        del model,state;gc.collect()
    write(RUN/'CPU_VALIDATION.json',dict(status='PASSED',tests=result.testsRun,branches=audits,source_sha256=SOURCE_SHA,training_windows=8337,training_scenes=1191,
        view_protocol='2context+2novel',global_batch=8,updates=4172,gpu_verified=False))
    print(json.dumps(dict(status='PASSED',run=str(RUN),branches={k:{n:v for n,v in r.items() if n!='migration'} for k,r in audits.items()})),flush=True)

if __name__=='__main__':main()
