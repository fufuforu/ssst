"""Real CPU asset/data check for smoke -> fresh Unified instance short training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,hashlib,json,subprocess
import torch
from tokengs.models.unified_instance_grounding import LocusGSInstanceGroundedUnified
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from scripts.train_dual_stream_v1 import cohorts,build_batch,MANIFEST,HELDOUT,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--architecture',default='unified_instance_grounding',choices=('unified_instance_grounding','unified_instance_lift','unified_instance_boundary','unified_instance_bidirectional'))
    parser.add_argument('--baseline-run',type=Path);args=parser.parse_args()
    root=args.run_root
    if any((root/name).exists() for name in ('submission.json','STARTED.json','preflight.json')):
        raise ValueError('Refusing duplicate preflight/run')
    torch.set_num_threads(4);torch.manual_seed(42)
    cfg=UnifiedQueryConfig();opt=config_defaults['train_siu3r_unified_query_v1']
    model=build_structural_model(args.architecture,opt,cfg)
    assert not any(p.requires_grad for module in (model.frozen_omega,model.sam_proposals) for p in module.parameters())
    groups=model.parameter_groups();ids=[id(p) for ps in groups.values() for p in ps]
    assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
    names=[name for name,p in model.named_parameters() if p.requires_grad]
    assert len([n for n in names if n.startswith('core.instance_head.')])==4
    baseline_audit=None
    if args.architecture!='unified_instance_grounding':
        if args.baseline_run is None:raise ValueError('Coupling controls require the completed baseline identity')
        baseline=args.baseline_run
        if json.loads((baseline/'COMPLETE.json').read_text())['updates']!=256:raise ValueError('Baseline incomplete')
        initial=torch.load(baseline/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
        if initial['progress']['full_updates']!=0 or initial['optimizer']['state']:raise ValueError('Baseline initial state not fresh')
        state=model.state_dict()
        assert len(initial['model'])==305
        assert all(k in state and torch.equal(v,state[k]) for k,v in initial['model'].items()),'Common initial weights differ'
        assert initial['meta']['config']==cfg.to_dict()
        assert initial['meta']['omega_identity']==model.frozen_omega.source_identity and initial['meta']['sam_identity']==model.sam_proposals.source_identity
        assert initial['meta']['manifest_sha256']==digest(MANIFEST) and initial['meta']['heldout_sha256']==digest(HELDOUT)
        baseline_audit=dict(run=str(baseline),initial_sha256=digest(baseline/'checkpoint_initial.pt'),
            common_initial_tensors_exact=305,new_tensors=sorted(set(names)-set(initial['model'])),baseline_retrained=False)
    windows,chosen,heldout=cohorts();train=[windows[i] for i in chosen]
    if baseline_audit is not None:assert initial['meta']['training_windows']==train
    previous=json.loads((Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt01_esam_many_short_node18_8gpu')/'preflight.json').read_text())
    assert chosen==previous['short_train_indices'] and len({w['scene'] for w in train})==16
    samples=[]
    for scope,window in [('train',train[0]),('train',train[-1]),('heldout',heldout[0])]:
        batch=build_batch(opt,window,'cpu');rgb=batch['images_input']
        assert rgb.shape==(1,2,3,256,256) and torch.isfinite(batch['images_all']).all()
        assert rgb.min()>=0 and rgb.max()<=1 and torch.equal(rgb,batch['images_all'][:,:2])
        samples.append(dict(scope=scope,scene=window['scene'],rgb_sha256=hashlib.sha256(batch['images_all'].contiguous().numpy().tobytes()).hexdigest()))
    record=dict(status='PASSED',architecture=model.architecture_name,architecture_key=args.architecture,config=cfg.to_dict(),baseline_comparability=baseline_audit,
        extra_checkpoint_metadata=model.extra_checkpoint_metadata,code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        real_cpu_assets_loaded=True,omega_identity=model.frozen_omega.source_identity,sam_identity=model.sam_proposals.source_identity,
        manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),training_windows=train,training_indices=chosen,
        heldout_probes=heldout[:4],samples=samples,world_size=4,global_batch=8,short_updates=256,
        parameter_counts={k:sum(p.numel() for p in ps) for k,ps in groups.items()},trainable_parameter_names=names,
        source_checkpoint=None,initial_updates=0,initial_optimizer_states=0,seed=42,
        learning_rate=1e-4,understanding_weight=.05,understanding_warmup=20,lr_warmup=20,
        smoke_updates_discarded=2,official128=False,auto_full_training=False,cuda_verified=False)
    root.mkdir(parents=True,exist_ok=True)
    (root/'model_config.json').write_text(json.dumps(cfg.to_dict(),indent=2)+'\n')
    (root/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('training_windows','heldout_probes','trainable_parameter_names')},indent=2))


if __name__=='__main__':main()
