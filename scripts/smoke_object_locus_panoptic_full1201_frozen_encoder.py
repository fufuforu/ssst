"""Only the prescribed single3090 and40-update global8 validation gates."""
import argparse,json
import torch
import torch.distributed as dist
from scripts.object_locus_panoptic_full1201_frozen_encoder_runtime import *
from scripts.smoke_object_locus_panoptic_v1 import state_digest

def required_gradient_paths(model,*,require_child=True):
 groups={
  'adapter':lambda n:n.startswith('understanding.adapter.'),
  'pixel_decoder':lambda n:n.startswith('understanding.mask2former.pixel_decoder.'),
  'query_decoder':lambda n:n.startswith('understanding.mask2former.transformer_module.'),
  'object_child':lambda n:n.startswith('panoptic.child_mlp.'),
  'reconstruction_encoder':lambda n:n.startswith(('patch_embed.','enc_dec_backbone.encoder.')),
  'reconstruction_decoder':lambda n:n.startswith('enc_dec_backbone.decoder_blocks.'),
  'anchor_geometry':lambda n:n.startswith(('anchor_decoder.mu','anchor_decoder.rho')),
 }
 result={key:any(predicate(name) and parameter.grad is not None for name,parameter in model.named_parameters())
     for key,predicate in groups.items()}
 if not require_child:
  result['object_child']='checked by 40-update smoke after understanding warm-up activates'
 required=[value for key,value in result.items() if key!='object_child' or require_child]
 if not all(required): raise RuntimeError(f'missing required gradient path: {result}')
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['single','eight']);args=parser.parse_args();single=args.phase=='single'
 configure(REPORT/'smoke_runtime',RUN/'smoke_temporary');device=base.init_distributed();rank,world=base.rank_world()
 if world!=(1 if single else 8):raise RuntimeError('smoke world mismatch')
 if not single and json.loads((REPORT/'single_smoke.json').read_text())['status']!='PASS':raise RuntimeError('single gate missing')
 manifest,plan,_,_=assets();model,opt=build_model(device);optimizer=build_optimizer(model);initial_encoder_digest=frozen_state_digest(model);logs=[];windows=[];torch.cuda.reset_peak_memory_stats()
 expected={f'understanding.encoder.{n}' for n,_ in model.understanding.encoder.named_parameters()}
 actual={n for n,p in model.named_parameters() if not p.requires_grad}
 if actual!=expected: raise RuntimeError(f'frozen parameter set mismatch: expected={len(expected)} actual={len(actual)}')
 model.train()
 if model.understanding.encoder.training: raise RuntimeError('encoder did not remain eval after train()')
 for update in range(1 if single else 40):
  wi=plan['orders'][0][0] if single else plan['orders'][0][8*update+rank];batch=base.build_batch(opt,manifest['windows'][wi],device)
  out,row=train_step(model,opt,optimizer,batch,update,check_rec_under=not single and update==39)
  for key in ('F_m','q_pre','gaussians','gaussian_membership','region_mass','alpha'):
   if not torch.isfinite(out['prediction'][key]).all():raise FloatingPointError('smoke output '+key)
  if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('smoke gradient')
  if any(p.grad is not None for p in model.understanding.encoder.parameters()): raise RuntimeError('frozen encoder has gradients')
  paths=required_gradient_paths(model,require_child=False)
  logs.append(row);windows.append(wi);print(json.dumps(dict(rank=rank,update=update+1,window_id=wi,loss_recon=row['loss_recon'],loss_understanding=row['loss_understanding'])),flush=True);del out,batch
 if frozen_state_digest(model)!=initial_encoder_digest: raise RuntimeError('frozen encoder changed during smoke')
 if single:
  from scripts.eval_object_locus_panoptic_v1 import evaluate_windows
  model.understanding_step=8
  result,_,_=evaluate_windows(model,opt,[manifest['windows'][wi]],1,'single_smoke',REPORT/'smoke_eval',device,base.build_batch,official=True,panels=True)
  base.write_json(REPORT/'single_smoke.json',dict(status='PASS',discarded=True,optimizer_updates=1,window_id=wi,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),steps=logs,evaluator=result,frozen_encoder_state_sha256=initial_encoder_digest,frozen_encoder_unchanged=True,frozen_encoder_parameters=len(expected),trainable_parameters=sum(1 for p in model.parameters() if p.requires_grad),required_gradient_paths=paths))
 else:
  error=None
  if any(a['weight_norm']<=0 or a['delta_norm']<=0 for a in logs[-1]['activity'].values()):error=RuntimeError('inactive injection')
  if not logs[-1]['rec_to_pretrained']:error=RuntimeError('no rec to pretrained gradient')
  paths=required_gradient_paths(model)
  base.synchronized_check(error,device,'smoke40',40)
  payload=dict(rank=rank,windows=windows,digest=state_digest(model,optimizer),frozen_encoder_state_sha256=frozen_state_digest(model),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),activity=logs[-1]['activity'],required_gradient_paths=paths,last=logs[-1]);ranks=[None]*8;dist.all_gather_object(ranks,payload)
  if len({x['digest'] for x in ranks})!=1:raise RuntimeError('rank optimizer/model mismatch')
  if sorted(w for x in ranks for w in x['windows'])!=sorted(plan['orders'][0][:320]):raise RuntimeError('smoke exposure mismatch')
  if any(x['frozen_encoder_state_sha256']!=initial_encoder_digest for x in ranks): raise RuntimeError('rank encoder state changed')
  if rank==0:base.write_json(REPORT/'eight_smoke.json',dict(status='PASS',discarded=True,optimizer_updates=40,exposures=320,ranks=ranks,frozen_encoder_unchanged=True,frozen_encoder_state_sha256=initial_encoder_digest))
  dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
