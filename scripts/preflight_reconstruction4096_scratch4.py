"""Actual CPU zero migration and Baseline4 compatibility; never fake a GPU pass."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,copy,json,hashlib
import torch
from scripts.reconstruction4096_scratch_policy import initialize,optimizer,KEY,ZERO_SHA
from scripts.reconstruction_abcd_policy import task_state,exact_equal,EXPANDED
from scripts.v4init_isolated_recipe import initialize_from_v4init_zero
from tokengs.models.superpoint_structural_variants import build_structural_model,MODELS
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from tests.test_unified_query_v1 import FakeOmega,FakeSAM,FakeRenderer


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args()
    run=args.run_root;cfg=json.loads((run/'configuration.json').read_text())
    assert sha(cfg['source_checkpoint'])==cfg['source_sha256']==ZERO_SHA
    saved=torch.load(cfg['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    torch.manual_seed(42);opt=config_defaults['train_siu3r_unified_query_v1'];config=UnifiedQueryConfig(**saved['meta']['config'])
    def make(key):
        omega=FakeOmega();omega.source_identity=saved['meta']['omega_identity']
        sam=FakeSAM();sam.source_identity=saved['meta']['sam_identity']
        return build_structural_model(key,opt,config,omega=omega,sam=sam,renderer=FakeRenderer())
    model=make(KEY);proof=initialize(model,saved,cfg);adam=optimizer(model)
    assert len(proof['copied_keys'])==631 and len(proof['removed_keys'])==32 and not adam.state
    baseline=make('dual_v4init_isolated_noca')
    initialize_from_v4init_zero(saved,baseline,manifest_sha256=cfg['manifest_sha256'],heldout_sha256=cfg['heldout_sha256'],world=8)
    old,new=task_state(baseline),task_state(model);assert set(old)==set(new) and len(new)==632
    for name,value in old.items():assert exact_equal(value,new[name][:1024] if name==EXPANDED else new[name]),name
    baseline_trainable={n for n,p in baseline.named_parameters() if p.requires_grad}
    assert baseline_trainable=={n for n,p in model.named_parameters() if p.requires_grad}
    constructors={cls.architecture_name:cls for cls in MODELS.values()};assert constructors[model.architecture_name] is type(model)
    rejected=[]
    for label,changes in [('trained',dict(progress=dict(epoch=4,update_in_epoch=0,full_updates=4172))),
                          ('nonempty_adam',dict(optimizer=dict(state={0:{'step':torch.tensor(1)}}))),
                          ('missing_key',dict(model={n:v for n,v in saved['model'].items() if n!='gfm_projection.0.weight'}))]:
        bad=dict(saved,**changes)
        try:initialize(model,bad,cfg)
        except ValueError:rejected.append(label)
        else:raise AssertionError('Bad source accepted: '+label)
    from scripts.dual_stream_training_policy import lr_factor,epoch_order
    for step in (0,1,99,100,1042,4171):
        rates={g['family']:g['base_lr']*lr_factor('full',step,4172) for g in adam.param_groups}
        assert abs(rates['geometry']/rates['main']-.1)<1e-12
    assert len(epoch_order(8337,0))//8==1043
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(run/'tensorboard'));writer.add_text('run/prepared','4096 tokens; zero-step source; full4; awaiting GPU gate',0);writer.close()
    result=dict(status='PASSED',gpu_forward_backward_verified=False,source_sha256=ZERO_SHA,
                actual_task_tensors=632,copied_exact=631,first1024_token_rows_exact=True,
                trainable_names_equal_baseline4=True,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                optimizer_empty=True,geometry_lr_ratio=.1,rejected_sources=rejected,initialization=proof,
                frozen_asset_loading='Synthetic CPU wrappers ONLY; identities checked against source; actual loading verified by queued GPU gate')
    (run/'CPU_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='initialization'}),flush=True)


if __name__=='__main__':main()
