"""Fresh fixed-plan training with only the MASt3R understanding encoder frozen."""
import json,os,subprocess
import numpy as np
import torch
import torch.distributed as dist
from scripts.object_locus_panoptic_full1201_frozen_encoder_runtime import *

def save(model,optimizer,opt,u,code,hashes,provenance,counts,initial_encoder_digest):
 rank,world=base.rank_world()
 if frozen_state_digest(model)!=initial_encoder_digest: raise RuntimeError('frozen encoder state changed at checkpoint')
 rng=base.capture_rng();rng['window_exposure_counts']=counts.tolist();all_rng=[None]*world if rank==0 else None
 dist.gather_object(rng,all_rng,dst=0)
 if rank==0:
  payload=checkpoint_payload(model,optimizer,opt,u,code,hashes,provenance,all_rng)
  payload['frozen_encoder']={'module_path':'understanding.encoder','parameter_names':list(model._frozen_understanding_encoder_names),
      'parameter_count':sum(p.numel() for p in model.understanding.encoder.parameters()),'state_sha256':initial_encoder_digest}
  atomic_save(RUN/f'checkpoint_epoch_{u//EPOCH_UPDATES:02}.pt',payload)
 dist.barrier()

def main():
 configure();device=base.init_distributed();rank,world=base.rank_world()
 if world!=8: raise RuntimeError('formal world must be 8')
 code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 if subprocess.check_output(['git','status','--porcelain'],text=True).strip(): raise RuntimeError('formal checkout must be clean')
 receipt=json.loads((REPORT/'git_provenance.json').read_text())
 if receipt['training_sha']!=code or not receipt['clean']: raise RuntimeError('clean pushed SHA receipt required')
 for phase in ('single','eight'):
  if json.loads((REPORT/f'{phase}_smoke.json').read_text())['status']!='PASS': raise RuntimeError('failed smoke '+phase)
 manifest,plan,hashes,provenance=assets()
 model,opt=build_model(device);optimizer=build_optimizer(model);u=0;counts=np.zeros(N,dtype=np.int64)
 initial_encoder_digest=frozen_state_digest(model)
 if rank==0:
  mapping=json.loads((REPORT/'weights_mapping.json').read_text())['mapping']
  frozen_mapping=[row for row in mapping if isinstance(row.get('target'),str) and row['target'].startswith('understanding.encoder.')]
  frozen_names=list(model._frozen_understanding_encoder_names)
  if {row['target'].removeprefix('understanding.encoder.') for row in frozen_mapping}!={n.removeprefix('understanding.encoder.') for n in frozen_names}:
   raise RuntimeError('encoder pretrained mapping and frozen names differ')
  frozen={'module_path':'understanding.encoder','parameter_names':frozen_names,
    'parameter_count':sum(p.numel() for p in model.understanding.encoder.parameters()),
    'pretrained_loading_keys':[row['source'] for row in frozen_mapping],
    'pretrained_loading_map':[dict(source=row['source'],target=row['target']) for row in frozen_mapping],
    'parameter_tensor_count':len(frozen_names),
    'state_sha256_at_initialization':initial_encoder_digest}
  base.write_json(REPORT/'frozen_encoder_manifest.json',frozen)
 if rank==0:
  base.write_json(REPORT/'execution_status.json',dict(status='FRESH_INITIALIZING',job_id=os.environ['SLURM_JOB_ID'],git_sha=code,
   manifest_sha256=hashes['manifest.json'],candidate_train_scenes=1201,actual_train_scenes=manifest['S'],N=N,U=EPOCH_UPDATES,P=plan['P']))
  base.write_json(REPORT/'run_manifest.json',dict(experiment='object_locus_panoptic_full1201_frozen_encoder_8gpu',
   job_id=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,git_sha=code,science_sha=BASE,fresh=True,
   global_batch=8,per_rank_batch=1,epochs=8,optimizer_updates=TOTAL,window_exposures=EXPOSURES,
   hashes=hashes,weights_provenance=provenance,reconstruction_checkpoint_step=47500,
   mast3r_component='MASt3R encoder',panoptic_checkpoint_epoch=60,strict_transfer_counts=dict(reconstruction=450,encoder=292,adapter=187,mask_decoder=326),
   frozen_encoder_module='understanding.encoder',frozen_encoder_state_sha256=initial_encoder_digest,
   padding_window_ids=plan['padding_window_ids'],checkpoint_epochs=SAVE,full_evaluations_launched=False,
   unregistered_scientific_changes=[]))
 base.write_json(REPORT/f'initial_encoder_rank{rank}.json',dict(state_sha256=initial_encoder_digest,
    parameter_count=sum(p.numel() for p in model.understanding.encoder.parameters()),
    trainable_numel=sum(p.numel() for p in model.parameters() if p.requires_grad),
    frozen_numel=sum(p.numel() for p in model.parameters() if not p.requires_grad)))
 dist.barrier()
 if u==0: save(model,optimizer,opt,0,code,hashes,provenance,counts,initial_encoder_digest)
 model.train()
 if model.understanding.encoder.training: raise RuntimeError('frozen encoder left eval after model.train()')
 while u<TOTAL:
  epoch,k=divmod(u,EPOCH_UPDATES);wi=plan['orders'][epoch][8*k+rank];error=None;batch=None
  try: batch=base.build_batch(opt,manifest['windows'][wi],device)
  except Exception as exc: error=exc
  base.synchronized_check(error,device,'batch',u,batch)
  output,row=train_step(model,opt,optimizer,batch,u);row.update(rank=rank,window_id=wi,identity=manifest['windows'][wi],padding=8*k+rank>=N)
  u+=1;counts[wi]+=1
  with (REPORT/f'training_rank{rank}.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
  del output,batch
  if rank==0:
   base.write_json(REPORT/'progress.json',dict(completed_updates=u,completed_exposures=u*8,epoch=u/EPOCH_UPDATES,finite_optimizer_update=True))
   if u==1:
    base.write_json(REPORT/'startup_confirmation.json',dict(status='FRESH_FORMAL_FINITE_UPDATE_1',job_id=os.environ['SLURM_JOB_ID'],
      git_sha=code,epoch0_path=str(RUN/'checkpoint_epoch_00.pt'),completed_updates=1,completed_exposures=8,
      loss_recon=row['loss_recon'],loss_understanding=row['loss_understanding'],finite=True,full_evaluations=False))
    base.write_json(REPORT/'execution_status.json',dict(status='TRAINING_STARTED',job_id=os.environ['SLURM_JOB_ID'],
      git_sha=code,actual_train_scenes=manifest['S'],N=N,U=EPOCH_UPDATES,P=plan['P'],full_evaluations_launched=False))
    print('FRESH_FORMAL_FINITE_UPDATE_1 epoch0_saved',flush=True)
  if u%EPOCH_UPDATES==0 and u//EPOCH_UPDATES in SAVE:
   save(model,optimizer,opt,u,code,hashes,provenance,counts,initial_encoder_digest)
 totals=torch.tensor(counts,device=device,dtype=torch.int64);dist.all_reduce(totals)
 if totals.cpu().tolist()!=plan['actual_expected_counts']: raise RuntimeError('actual exposure count mismatch')
 if rank==0: base.write_json(REPORT/'training_complete.json',dict(updates=TOTAL,exposures=EXPOSURES,epochs=8,
   window_counts=totals.cpu().tolist(),padding_window_ids=plan['padding_window_ids'],evaluation_status='WAIT_USER_INSTRUCTION'))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__': main()
