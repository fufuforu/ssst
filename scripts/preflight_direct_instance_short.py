"""Real source/asset/data and common-zero-parameter comparability, CPU only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,hashlib,json,subprocess
import torch
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from scripts.train_dual_stream_v1 import cohorts,build_batch,MANIFEST,HELDOUT,digest

ARCH='unified_direct_instance_attributes'
BASE=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt11_unified_instance_grounding_short_node13_4gpu')
REFERENCE=Path('/space/mawb/ssst/group_plus/fixed_instance_readout_v1/attempt01_67050_67292')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);a=parser.parse_args();root=a.run_root
    if any((root/f).exists() for f in ('preflight.json','submission.json','STARTED.json')):raise ValueError('Refusing reuse')
    torch.set_num_threads(4);torch.manual_seed(42)
    opt=config_defaults['train_siu3r_unified_query_v1'];cfg=UnifiedQueryConfig();model=build_structural_model(ARCH,opt,cfg)
    old=torch.load(BASE/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    assert old['progress']['full_updates']==0 and not old['optimizer']['state']
    state={k:v for k,v in model.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))}
    common=sorted(set(state)&set(old['model']));removed=sorted(set(old['model'])-set(state));added=sorted(set(state)-set(old['model']))
    assert all(torch.equal(state[k],old['model'][k]) for k in common),'Common zero-step weights differ'
    legacy=('core.query_mask.','core.child_index.','core.child_feature.','core.stuff_feature.','core.instance_head.')
    new=('core.instance_attribute_head.','core.instance_prototype_head.','core.stuff_attribute_head.','core.instance_logit_scale','core.instance_logit_bias')
    assert removed and added and all(k.startswith(legacy) for k in removed) and all(k.startswith(new) for k in added)
    assert old['meta']['config']==cfg.to_dict()
    assert model.frozen_omega.source_identity==old['meta']['omega_identity'] and model.sam_proposals.source_identity==old['meta']['sam_identity']
    for identity in (model.frozen_omega.source_identity,model.sam_proposals.source_identity):assert digest(Path(identity['path']))==identity['sha256']
    assert old['meta']['manifest_sha256']==digest(MANIFEST) and old['meta']['heldout_sha256']==digest(HELDOUT)
    groups=model.parameter_groups();ids=[id(p) for ps in groups.values() for p in ps]
    assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
    assert not any(p.requires_grad for module in (model.frozen_omega,model.sam_proposals) for p in module.parameters())
    windows,chosen,heldout=cohorts();train=[windows[i] for i in chosen]
    assert len(train)==16 and len({w['scene'] for w in train})==16 and old['meta']['training_windows']==train
    ref=json.loads((REFERENCE/'preflight.json').read_text());assert ref['windows']==train[:4]+heldout[:4]
    assert json.loads((REFERENCE/'COMPLETE.json').read_text())['windows']==16
    gt_root=Path(ref['gt_cache_root']);samples=[]
    # All 16 training windows + 4 heldout inputs, not a synthetic substitute.
    for scope,selected in [('train',train),('heldout',heldout[:4])]:
        for index,window in enumerate(selected):
            batch=build_batch(opt,window,'cpu');rgb=batch['images_input']
            assert rgb.shape==(1,2,3,256,256) and torch.isfinite(batch['images_all']).all()
            assert rgb.min()>=0 and rgb.max()<=1 and torch.equal(rgb,batch['images_all'][:,:2])
            if scope=='heldout' or index<4:
                cache_index=index+(4 if scope=='heldout' else 0)
                cached=torch.load(BASE/'probes/final'/f'{cache_index:02d}.pt',map_location='cpu',weights_only=False)
                gt=torch.load(gt_root/'probes/final'/f'{cache_index:02d}_coupling.pt',map_location='cpu',weights_only=False)
                assert torch.equal(rgb,cached['context_rgb']) and cached['scene']==window['scene']
                assert torch.equal(batch['semantic_label_all'][:,:2],gt['gt_semantic']) and torch.equal(batch['instance_label_all'][:,:2],gt['gt_instance'])
            samples.append(dict(scope=scope,scene=window['scene'],context=window['context'],rgb_sha256=hashlib.sha256(rgb.numpy().tobytes()).hexdigest()))
    record=dict(status='PASSED',architecture_key=ARCH,architecture=model.architecture_name,config=cfg.to_dict(),
        extra_checkpoint_metadata=model.extra_checkpoint_metadata,code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        omega_identity=model.frozen_omega.source_identity,sam_identity=model.sam_proposals.source_identity,
        manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),training_windows=train,training_indices=chosen,heldout_probes=heldout[:4],samples=samples,
        baseline_run=str(BASE),baseline_initial_sha256=digest(BASE/'checkpoint_initial.pt'),common_initial_tensors_exact=len(common),
        common_parameter_names=common,removed_legacy_parameters=removed,new_parameter_names=added,
        baseline_reference=str(REFERENCE),baseline_reference_summary_sha256=digest(REFERENCE/'SUMMARY.json'),
        trainable_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad],
        parameter_counts={k:sum(p.numel() for p in ps) for k,ps in groups.items()},real_cpu_assets_loaded=True,
        seed=42,source_checkpoint=None,initial_updates=0,initial_optimizer_states=0,world_size=4,global_batch=8,short_updates=256,
        learning_rate=1e-4,understanding_weight=.05,understanding_warmup=20,lr_warmup=20,smoke_updates_discarded=2,
        official128=False,auto_full_training=False,cuda_verified=False)
    root.mkdir(parents=True,exist_ok=True);(root/'model_config.json').write_text(json.dumps(cfg.to_dict(),indent=2)+'\n')
    (root/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status='PASSED',common_tensors=len(common),removed=len(removed),added=len(added),real_windows=len(samples),code_sha=record['code_sha']),indent=2))


if __name__=='__main__':main()
