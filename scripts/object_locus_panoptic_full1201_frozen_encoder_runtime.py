"""Fixed full1201 execution profile; scientific runtime/model are reused verbatim."""
from pathlib import Path
import hashlib,json,math,os,dataclasses
import numpy as np
import torch
from scripts import object_locus_panoptic_v1_runtime as base
BASE='7300b6ff6ae963ea7228ae7bc0c7d0e0446eefa3'
REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu')
RUN=Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_frozen_encoder_8gpu')
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


# The frozen experiment imports this runtime explicitly.  Default paths and
# behavior above remain the original trainable-encoder experiment.
def freeze_understanding_encoder(model):
 encoder=model.understanding.encoder
 names=[f'understanding.encoder.{name}' for name,_ in encoder.named_parameters()]
 if not names: raise RuntimeError('understanding encoder has no parameters')
 for parameter in encoder.parameters(): parameter.requires_grad_(False)
 encoder.eval()
 model._frozen_understanding_encoder_names=tuple(names)
 return names


def frozen_state_digest(model):
 import hashlib
 encoder=model.understanding.encoder
 h=hashlib.sha256()
 for name,value in sorted(encoder.state_dict().items()):
  h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
  h.update(memoryview(value.detach().cpu().contiguous().numpy()))
 return h.hexdigest()


def build_model(device='cpu',*,report=True):
 from tokengs.models import object_locus_panoptic_v1_pretrained as pretrained
 original=pretrained.PretrainedUnderstanding
 class FrozenUnderstanding(original):
  def __init__(self,mapping):
   super().__init__(mapping)
   self.encoder.eval()
   self.frozen_parameter_names=tuple(f'understanding.encoder.{n}' for n,_ in self.encoder.named_parameters())
  def train(self,mode=True):
   super().train(mode)
   self.encoder.eval()
   return self
  def forward(self,context_images):
   if context_images.shape[1:]!=(2,3,256,256): raise ValueError(f'expected context [B,2,3,256,256], got {context_images.shape}')
   b=context_images.shape[0]
   import torch.nn.functional as F
   image=F.interpolate(context_images.flatten(0,1),size=(512,512),mode='bilinear',align_corners=False)*2-1
   with torch.no_grad(): layers=self.encoder(image)
   scales=self.adapter(image,layers)
   scales=[x.reshape(b,2,1024,*x.shape[-2:]) for x in scales]
   if [x.shape[-2:] for x in scales]!=[(128,128),(64,64),(32,32),(16,16)]: raise RuntimeError('adapter scale contract failed')
   pixel=self.mask2former.pixel_decoder(scales,output_hidden_states=True)
   query=self.mask2former.transformer_module(word_embeddings=None,multi_scale_features=pixel.multi_scale_features,
       mask_features=pixel.mask_features,output_hidden_states=True,output_attentions=False)
   qpre=query.intermediate_hidden_states[-1].transpose(0,1);fm=pixel.mask_features
   if qpre.shape!=(b,100,256) or fm.shape!=(b,2,256,128,128): raise RuntimeError(f'understanding shapes {qpre.shape}/{fm.shape}')
   return fm,qpre
 pretrained.PretrainedUnderstanding=FrozenUnderstanding
 try:
  model,opt=base.build_model(device,report=report)
 finally:
  pretrained.PretrainedUnderstanding=original
 names=freeze_understanding_encoder(model)
 return model,opt


def build_optimizer(model):
 optimizer=base.build_optimizer(model)
 frozen={id(p) for p in model.understanding.encoder.parameters()}
 if any(id(p) in frozen for group in optimizer.param_groups for p in group['params']):
  raise RuntimeError('frozen encoder entered optimizer')
 return optimizer


def train_step(model,opt,optimizer,batch,update,**kwargs):
 output,row=base.train_one_step(model,opt,optimizer,batch,update,**kwargs)
 row['epoch']=(update+1)/EPOCH_UPDATES
 row['frozen_parameter_count']=sum(1 for _ in model.understanding.encoder.parameters())
 row['trainable_parameter_count']=sum(1 for p in model.parameters() if p.requires_grad)
 return output,row
