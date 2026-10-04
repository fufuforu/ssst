"""Fixed full1201 execution profile; scientific runtime/model are reused verbatim."""
from pathlib import Path
import hashlib,json,math,os,dataclasses
import numpy as np
import torch
from scripts import object_locus_panoptic_v1_runtime as base
BASE='7300b6ff6ae963ea7228ae7bc0c7d0e0446eefa3'
REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
RUN=Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu')
SAVE=(0,1,2,4,6,8)
_plan=json.loads((REPORT/'training_plan.json').read_text()) if (REPORT/'training_plan.json').exists() else {}
N=_plan.get('N',0);EPOCH_UPDATES=_plan.get('U',0);TOTAL=8*EPOCH_UPDATES;EXPOSURES=64*EPOCH_UPDATES
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def lr_multiplier(update):
 t=int(update)+1
 if not 1<=t<=TOTAL:raise ValueError('update out of range')
 return t/200 if t<=200 else .1+.9*(1+math.cos(math.pi*(t-200)/(TOTAL-200)))/2

def configure(report=REPORT,run=RUN):
 global N,EPOCH_UPDATES,TOTAL,EXPOSURES
 plan=json.loads((REPORT/'training_plan.json').read_text());N=plan['N'];EPOCH_UPDATES=plan['U'];TOTAL=8*EPOCH_UPDATES;EXPOSURES=64*EPOCH_UPDATES
 # Only execution destinations and user-prescribed LR clock change. GC/under/beta
 # remain the unchanged train_one_step's exposure=8*update formulas.
 base.REPORTS=Path(report);base.RUN=Path(run);base.lr_multiplier=lr_multiplier
 base.REPORTS.mkdir(parents=True,exist_ok=True);base.RUN.mkdir(parents=True,exist_ok=True)
def assets(verify_weights=True):
 hashes=json.loads((REPORT/'asset_hashes.json').read_text())
 for name,h in hashes.items():
  if sha(REPORT/name)!=h:raise RuntimeError('fixed asset changed: '+name)
 manifest=json.loads((REPORT/'manifest.json').read_text());plan=json.loads((REPORT/'training_plan.json').read_text());provenance=json.loads((REPORT/'weights_provenance.json').read_text())
 if verify_weights:
  for w in provenance['weights'].values():
   if sha(w['path'])!=w['sha256']:raise RuntimeError('pretrained SHA mismatch '+w['path'])
 return manifest,plan,hashes,provenance

def train_step(model,opt,optimizer,batch,update,**kwargs):
 output,row=base.train_one_step(model,opt,optimizer,batch,update,**kwargs)
 row['epoch']=(update+1)/EPOCH_UPDATES
 return output,row

def atomic_save(path,payload):
 path=Path(path);tmp=path.with_suffix('.tmp');torch.save(payload,tmp)
 test=torch.load(tmp,map_location='cpu',weights_only=False,mmap=True)
 for key in ('completed_updates','completed_exposures','plan_sha256','git_sha'):
  if test[key]!=payload[key]:raise RuntimeError('checkpoint metadata verification '+key)
 if set(test['model'])!=set(payload['model']):raise RuntimeError('checkpoint model keys')
 del test;os.replace(tmp,path)
def checkpoint_payload(model,optimizer,opt,update,sha_code,hashes,provenance,rank_rng):
 return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=dict(completed_updates=update,total_updates=TOTAL,warmup_updates=200,lr_clock='optimizer_update',understanding_clock='window_exposure'),completed_updates=update,completed_exposures=update*8,epoch=update//EPOCH_UPDATES,git_sha=sha_code,science_sha=BASE,plan_sha256=hashes['training_plan.json'],manifest_sha256=hashes['manifest.json'],weights_provenance=provenance,config=dataclasses.asdict(opt),rank_rng=rank_rng,world_size=8)
def restore_checkpoint(blob,model,optimizer,sha_code,hashes):
 if blob['git_sha']!=sha_code or blob['plan_sha256']!=hashes['training_plan.json'] or blob['manifest_sha256']!=hashes['manifest.json'] or blob['world_size']!=8:raise RuntimeError('resume provenance mismatch')
 if blob['scheduler']['total_updates']!=TOTAL or blob['scheduler']['completed_updates']!=blob['completed_updates']:raise RuntimeError('resume schedule mismatch')
 model.load_state_dict(blob['model'],strict=True);optimizer.load_state_dict(blob['optimizer']);return blob['completed_updates']
