"""Eight-rank real CUDA engineering gate; both updates are discarded."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,json,os,random,gc
from unittest.mock import patch
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tokengs.models.unified_query_gs_evidence as module
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from scripts.gs_evidence_full import SPECS,read,write,code_sha,validate_smoke,digest
from scripts.train_dual_stream_v1 import build_batch
from scripts.train_unified_query_v1 import Task,restore_state
from scripts.unified_query_validation import parameter_digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=tuple(SPECS),required=True);args=parser.parse_args()
    run=SPECS[args.group]['run'];pre=read(run/'preflight.json')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==8
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl',device_id=device)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert pre['code_sha']==code_sha() and pre['initial_sha256']==digest(run/'checkpoint_untrained_initial.pt')
    if rank==0:
        assert not (run/'SMOKE_STARTED.json').exists() and not (run/'STARTED.json').exists()
        write(run/'SMOKE_STARTED.json',dict(code_sha=code_sha(),discarded_updates=2))
    dist.barrier()
    opt=config_defaults['train_siu3r_unified_query_v1']
    model=build_structural_model(pre['architecture_key'],opt,UnifiedQueryConfig(**pre['config'])).to(device).train()
    saved=torch.load(run/'checkpoint_untrained_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    restore_state(model,saved['model']);del saved
    initial=parameter_digest(model);before=model.core.child_feature[0].weight.detach().clone()
    position_absent=True;gfm_nonzero=False;evidence_finite=True;zero=0.
    original=module.sample_gs_evidence;captured=[]
    def sampling(*args,**kwargs):
        result=original(*args,**kwargs);captured.append((result[0],args[1]));return result
    for index in (rank,rank+8):
        batch=build_batch(opt,pre['smoke_windows'][index],device)
        with patch.object(module,'sample_gs_evidence',side_effect=sampling):pred=model.generate(batch['images_input'])
        assert len(captured)==1;e,g=captured.pop()
        assert e.shape==(1,pred['num_thing_queries']*64,256)
        position_gradient,gfm_gradient=torch.autograd.grad(e.square().mean(),(g,model.gfm_projection[1].weight),allow_unused=True)
        position_absent &= position_gradient is None
        assert gfm_gradient is not None and torch.isfinite(gfm_gradient).all()
        gfm_nonzero |= bool(gfm_gradient.abs().sum()>0)
        evidence_finite &= bool(torch.isfinite(e).all());zero=pred['evidence_zero_fraction']
        del batch,pred,e,g,position_gradient,gfm_gradient
        gc.collect();torch.cuda.empty_cache()
    assert parameter_digest(model)==initial and all(p.grad is None for p in model.parameters())
    grouped={}
    for p in model.parameters():
        if p.requires_grad:grouped.setdefault(0. if p.ndim<2 or getattr(p,'_no_weight_decay',False) else .05,[]).append(p)
    optimizer=torch.optim.AdamW([dict(params=ps,weight_decay=wd) for wd,ps in grouped.items()],lr=1e-4,betas=(.9,.95),eps=1e-8)
    task=DDP(Task(model),device_ids=[local],find_unused_parameters=True);steps=[];all_finite=True;coverage_positive=False
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for param_group in optimizer.param_groups:param_group['lr']=1e-4*(step+1)/100
        batch=build_batch(opt,pre['smoke_windows'][rank+step*8],device)
        loss,numbers=task(batch,step*8,.05*(step+1)/1043);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        assert model.core.child_feature[0].weight.grad[:,-256:].abs().sum()>0
        grad_norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
        optimizer.step();coverage_positive |= numbers.get('loss_coverage',0)>0
        steps.append(dict(loss=numbers['loss'],grad_norm=grad_norm,parameter_sha256=parameter_digest(model)))
        print(json.dumps(dict(rank=rank,smoke_step=step+1,**steps[-1])),flush=True)
        del batch,loss
    row=dict(rank=rank,probe_window_indices=[rank,rank+8],initial_parameter_sha256=initial,steps=steps,all_finite_gradients=all_finite,
             child_feature_changed=not torch.equal(before,model.core.child_feature[0].weight),
             adam_steps=sorted({float(v['step']) for v in optimizer.state.values()}),
             frozen_no_grad=not any(p.grad is not None for m in (model.frozen_omega,model.sam_proposals) for p in m.parameters()),
             sampling_position_gradient_absent=position_absent,sampling_gfm_gradient_nonzero=gfm_nonzero,
             evidence_finite=evidence_finite,evidence_zero_fraction=zero,coverage_positive=coverage_positive)
    rows=[None]*8;dist.all_gather_object(rows,row)
    receipt=dict(status='PASSED',code_sha=code_sha(),initial_sha256=pre['initial_sha256'],ranks=rows,discarded_updates=2)
    validate_smoke(receipt,pre)
    if rank==0:write(run/'SMOKE_PASSED.json',receipt)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
