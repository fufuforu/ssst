"""Only the prescribed single3090 and40-update global8 validation gates."""
import argparse,json
import torch
import torch.distributed as dist
from scripts.object_locus_panoptic_full1201_runtime import *
from scripts.smoke_object_locus_panoptic_v1 import state_digest

def main():
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['single','eight']);args=parser.parse_args();single=args.phase=='single'
 configure(REPORT/'smoke_runtime',RUN/'smoke_temporary');device=base.init_distributed();rank,world=base.rank_world()
 if world!=(1 if single else 8):raise RuntimeError('smoke world mismatch')
 if not single and json.loads((REPORT/'single_smoke.json').read_text())['status']!='PASS':raise RuntimeError('single gate missing')
 manifest,plan,_,_=assets();model,opt=base.build_model(device);optimizer=base.build_optimizer(model);logs=[];windows=[];torch.cuda.reset_peak_memory_stats()
 for update in range(1 if single else 40):
  wi=plan['orders'][0][0] if single else plan['orders'][0][8*update+rank];batch=base.build_batch(opt,manifest['windows'][wi],device)
  out,row=train_step(model,opt,optimizer,batch,update,check_rec_under=not single and update==39)
  for key in ('F_m','q_pre','gaussians','gaussian_membership','region_mass','alpha'):
   if not torch.isfinite(out['prediction'][key]).all():raise FloatingPointError('smoke output '+key)
  if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('smoke gradient')
  logs.append(row);windows.append(wi);print(json.dumps(dict(rank=rank,update=update+1,window_id=wi,loss_recon=row['loss_recon'],loss_understanding=row['loss_understanding'])),flush=True);del out,batch
 if single:
  from scripts.eval_object_locus_panoptic_v1 import evaluate_windows
  model.understanding_step=8
  result,_,_=evaluate_windows(model,opt,[manifest['windows'][wi]],1,'single_smoke',REPORT/'smoke_eval',device,base.build_batch,official=True,panels=True)
  base.write_json(REPORT/'single_smoke.json',dict(status='PASS',discarded=True,optimizer_updates=1,window_id=wi,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),steps=logs,evaluator=result))
 else:
  error=None
  if any(a['weight_norm']<=0 or a['delta_norm']<=0 for a in logs[-1]['activity'].values()):error=RuntimeError('inactive injection')
  if not logs[-1]['rec_to_pretrained']:error=RuntimeError('no rec to pretrained gradient')
  base.synchronized_check(error,device,'smoke40',40)
  payload=dict(rank=rank,windows=windows,digest=state_digest(model,optimizer),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),activity=logs[-1]['activity'],last=logs[-1]);ranks=[None]*8;dist.all_gather_object(ranks,payload)
  if len({x['digest'] for x in ranks})!=1:raise RuntimeError('rank optimizer/model mismatch')
  if sorted(w for x in ranks for w in x['windows'])!=sorted(plan['orders'][0][:320]):raise RuntimeError('smoke exposure mismatch')
  if rank==0:base.write_json(REPORT/'eight_smoke.json',dict(status='PASS',discarded=True,optimizer_updates=40,exposures=320,ranks=ranks))
  dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
