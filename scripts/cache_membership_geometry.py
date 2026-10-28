"""Use the original strict Baseline4 loader; freeze a train-only teacher cache."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import torch
import torch.distributed as dist
from scripts.membership_geometry_core import *
from scripts.dual_stream_frozen_cache import atomic_torch_save
from scripts.membership_geometry_sdpa import install,evidence

def export(model,opt,windows,root,rank,world,device,*,expected_windows):
    from scripts.train_dual_stream_v1 import cohorts,build_batch
    from scripts.train_unified_query_v1 import rng,restore_rng
    root=Path(root);run=root.parent;cfg=json.loads((run/'configuration.json').read_text())
    if sha(run/'training_cohort.json')!=cfg['training_cohort_sha256']:raise ValueError('Training cohort hash differs')
    original,_,official=cohorts()
    if windows!=original or len(windows)!=8337 or world!=4:raise ValueError('Original train cohort/four ranks required')
    forbidden={w['scene'] for w in json.loads(Path(cfg['official_cohort']).read_text())['records']}
    if forbidden&{w['scene'] for w in windows}:raise ValueError('Train/eval scene overlap')
    model.requires_grad_(False);model.eval();versions={n:p._version for n,p in model.named_parameters()};records=[]
    for i in range(rank,len(windows),world):
        batch=build_batch(opt,windows[i],device)
        with torch.no_grad():record=teacher_record(model.generate(batch['images_input']),batch,model.config)
        if i==rank:
            state=rng();freeze(model)
            with torch.autograd.detect_anomaly(check_nan=True):
                pred=model.generate(batch['images_input']);loss,ce,dice=geometry_mask_loss(model,pred,record)
                loss.backward()
            family={prefix:sum(float(p.grad.detach().float().square().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)**.5 for prefix in PREFIXES}
            if not all(v>0 and torch.isfinite(torch.tensor(v)) for v in family.values()):raise ValueError('Mask gradient did not reach both geometry heads')
            if pred['gaussian_membership'].requires_grad:raise ValueError('Understanding unexpectedly attached')
            if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith(PREFIXES)):raise ValueError('New loss leaked outside geometry heads')
            mask_numbers=dict(loss=float(loss),ce=float(ce),dice=float(dice),geometry_gradient_norms=family)
            model.zero_grad(set_to_none=True);del pred,loss,ce,dice
            restore_rng(state)
            with torch.autograd.detect_anomaly(check_nan=True):
                pred=model.generate(batch['images_input']);recon=reconstruction(model,pred,batch);recon.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float('inf'),error_if_nonfinite=True)
            if not float(norm)>0 or evidence()['math_bias_calls']==0:raise ValueError('Reconstruction/SDPA guard not exercised')
            if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith(PREFIXES)):raise ValueError('Reconstruction gradient leaked')
            write(root/f'GPU_GRADIENT_SMOKE_rank{rank:02d}.json',dict(status='PASSED',**mask_numbers,
                reconstruction_backward=True,reconstruction_loss=float(recon),reconstruction_gradient_norm=float(norm),attention_backend=evidence(),
                membership_detached=True,frozen_parameter_grads_none=True,optimizer_updates=0))
            model.zero_grad(set_to_none=True);model.requires_grad_(False);model.eval();restore_rng(state);del pred,recon
        path=root/'windows'/f'{i:05d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
        record.update(window_index=i,scene=windows[i]['scene'],context=windows[i]['context'],source_sha256=SOURCE_SHA)
        atomic_torch_save(record,path)
        records.append(dict(window_index=i,file=str(path.relative_to(root)),sha256=sha(path),scene=record['scene'],matched_instances=record['matched_instances'],gt_instances=record['gt_instances']))
        if len(records)%25==0:write(root/f'progress_rank{rank:02d}.json',dict(windows=len(records)))
        del batch,record
    if any(p.grad is not None or p.requires_grad or p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('Teacher changed')
    gathered=[None]*world;dist.all_gather_object(gathered,records)
    if rank==0:
        records=sorted([r for rows in gathered for r in rows],key=lambda r:r['window_index'])
        if [r['window_index'] for r in records]!=list(range(8337)):raise ValueError('Cache coverage differs')
        write(root/'export_complete.json',dict(status='FIXED_TEACHER_COMPLETE',windows=8337,scenes=1191,parameter_updates=0,source_sha256=SOURCE_SHA,
              gpu_gradient_smoke_ranks=4,teacher_versions_unchanged=True,records=records))
    dist.barrier()

if __name__=='__main__':
    from scripts import export_unified_query_checkpoint as loader
    install()
    loader.export=export;loader.main()
