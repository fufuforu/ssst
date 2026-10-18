"""Real CPU asset/data check for smoke -> fresh Unified instance short training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,hashlib,json,subprocess
import torch
from tokengs.models.unified_instance_grounding import LocusGSInstanceGroundedUnified
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from scripts.train_dual_stream_v1 import cohorts,build_batch,MANIFEST,HELDOUT,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args()
    root=args.run_root
    if any((root/name).exists() for name in ('submission.json','STARTED.json','preflight.json')):
        raise ValueError('Refusing duplicate preflight/run')
    torch.set_num_threads(4);torch.manual_seed(42)
    cfg=UnifiedQueryConfig();opt=config_defaults['train_siu3r_unified_query_v1']
    model=LocusGSInstanceGroundedUnified(opt,config=cfg)
    assert not any(p.requires_grad for module in (model.frozen_omega,model.sam_proposals) for p in module.parameters())
    groups=model.parameter_groups();ids=[id(p) for ps in groups.values() for p in ps]
    assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
    names=[name for name,p in model.named_parameters() if p.requires_grad]
    assert len([n for n in names if n.startswith('core.instance_head.')])==4
    windows,chosen,heldout=cohorts();train=[windows[i] for i in chosen]
    previous=json.loads((Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt01_esam_many_short_node18_8gpu')/'preflight.json').read_text())
    assert chosen==previous['short_train_indices'] and len({w['scene'] for w in train})==16
    samples=[]
    for scope,window in [('train',train[0]),('train',train[-1]),('heldout',heldout[0])]:
        batch=build_batch(opt,window,'cpu');rgb=batch['images_input']
        assert rgb.shape==(1,2,3,256,256) and torch.isfinite(batch['images_all']).all()
        assert rgb.min()>=0 and rgb.max()<=1 and torch.equal(rgb,batch['images_all'][:,:2])
        samples.append(dict(scope=scope,scene=window['scene'],rgb_sha256=hashlib.sha256(batch['images_all'].contiguous().numpy().tobytes()).hexdigest()))
    record=dict(status='PASSED',architecture=model.architecture_name,config=cfg.to_dict(),
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
