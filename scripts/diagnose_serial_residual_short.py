"""Diagnose only the residual model; reuse 66637 baseline without rerunning it."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,gc,json,os,random
import numpy as np
import torch
import torch.distributed as dist
from scripts.diagnose_serial_query_masks import observe
from scripts.train_unified_query_v1 import restore_state
from scripts.train_dual_stream_v1 import digest,MANIFEST,HELDOUT
from scripts.serial_residual_initial import RESIDUAL_ARCH
from tokengs.models.superpoint_structural_variants import LocusGSSerialResidualQuery
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);args=p.parse_args();root=args.run_root
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world!=4:raise ValueError('Residual comparison uses the same four diagnostic ranks')
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    preflight=json.loads((root/'preflight.json').read_text());baseline=Path(preflight['baseline_diagnostics'])/'serial'
    if digest(MANIFEST)!=preflight['manifest_sha256'] or digest(HELDOUT)!=preflight['heldout_sha256']:raise ValueError('Diagnostic data changed')
    if digest(Path(preflight['source_final']))!=preflight['source_final_sha256']:raise ValueError('Baseline checkpoint changed')
    folder=root/'diagnostics'/'residual';folder.mkdir(parents=True,exist_ok=True)
    if rank==0:
        started=root/'diagnostics'/'STARTED.json'
        if started.exists():raise ValueError('Refusing duplicate residual diagnosis')
        started.write_text(json.dumps(dict(parameter_updates=0,optimizer_created=False,baseline_reused=True))+'\n')
    dist.barrier()
    checkpoint=root/'checkpoint_latest.pt';checkpoint_sha=digest(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);meta=saved['meta']
    if meta['architecture']!=RESIDUAL_ARCH or not meta['short_validation'] or saved['progress']['full_updates']!=256 or len(saved['model'])!=386:raise ValueError('Wrong residual endpoint')
    opt=config_defaults['train_siu3r_unified_query_v1'];model=LocusGSSerialResidualQuery(opt,config=UnifiedQueryConfig(**meta['config'])).to(device)
    if model.frozen_omega.source_identity!=meta['omega_identity'] or model.sam_proposals.source_identity!=meta['sam_identity']:raise ValueError('Frozen assets differ')
    restore_state(model,saved['model']);model.eval().requires_grad_(False)
    for index,window in enumerate(preflight['windows']):
        if index%world!=rank:continue
        path=baseline/f'{index:02d}.json'
        if digest(path)!=preflight['baseline_window_sha256'][str(index)]:raise ValueError('Reused baseline window changed')
        reference=json.loads(path.read_text())
        row=observe(model,opt,window,index,'train' if index<16 else 'heldout',device,True,index==rank,folder,reference_folder=baseline)
        for key in ('scene','context','input_rgb_sha256','gt_ids','gt_classes'):
            if row[key]!=reference[key]:raise ValueError('Baseline/residual identity mismatch: '+key)
        row['identity_residual_gate']=float(model.core.identity_gate_logit.sigmoid())
        (folder/f'{index:02d}.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(rank=rank,index=index,scene=row['scene'],raw_iou=row['stage_metrics']['raw_alpha']['source_supported_only']['best_iou_mean'])),flush=True)
    if not all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in saved['model'].items()):raise ValueError('Diagnostic changed model weights')
    if any(p.grad is not None for p in model.parameters()):raise ValueError('Diagnostic produced parameter gradients')
    (root/'diagnostics'/f'UNCHANGED_rank{rank:02}.json').write_text(json.dumps(dict(unchanged=True,parameters=386,parameter_updates=0,optimizer_created=False,checkpoint_sha256=checkpoint_sha))+'\n')
    del model,saved;gc.collect();torch.cuda.empty_cache();dist.barrier()
    if rank==0:(root/'diagnostics'/'INFERENCE_COMPLETE.json').write_text(json.dumps(dict(windows=32,parameter_updates=0,baseline_reused=True))+'\n')
    dist.destroy_process_group()


if __name__=='__main__':main()
