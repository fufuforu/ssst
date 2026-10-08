#!/usr/bin/env python3
"""One isolated CUDA worker for a single H1/H2/H3 head and its fixed seeds."""
import argparse,csv,json,os,platform,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from scripts.train_object_locus_frozen_probe import Readout,load_split

SEEDS=(20261,20262,20263)
def weighted_loss(logits,y):
    v=y>=0
    if not v.any(): return None
    wt=torch.ones(19,device=logits.device,dtype=logits.dtype);wt[18]=.1
    ce=F.cross_entropy(logits[v],y[v],reduction='none')
    return (ce*wt[y[v]]).sum()/wt[y[v]].sum()
def evaluate(model,rows,device):
    model.eval();num=den=0.;jc=jn=cc=cn=0
    with torch.no_grad():
      for st in range(0,len(rows),16):
        part=rows[st:st+16];q=torch.as_tensor(np.stack([r['q'] for r in part]),device=device);z=torch.as_tensor(np.stack([r['z'] for r in part]),device=device)
        y=torch.as_tensor(np.stack([r['labels'] for r in part]),device=device);log=model(q,z);v=y>=0
        if v.any():
          wt=torch.ones(19,device=device);wt[18]=.1;num+=float(F.cross_entropy(log[v],y[v],weight=wt,reduction='sum'));den+=float(wt[y[v]].sum())
        pos=(y>=0)&(y<18);jp=log.argmax(-1);cp=log[:,:,:18].argmax(-1) if log.ndim==3 else log[:,:18].argmax(-1)
        jc+=int(((jp==y)&pos).sum());jn+=int(pos.sum());cc+=int(((cp==y)&pos).sum());cn+=int(pos.sum())
    return {'ce':num/den if den else None,'joint19_accuracy':jc/jn if jn else None,'conditional18_accuracy':cc/cn if cn else None,'positive_queries':jn}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--head',required=True,choices=['H1','H2','H3']);ap.add_argument('--device',default='cuda:0');ap.add_argument('--attempt',type=Path,required=True);a=ap.parse_args();root=a.attempt
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError(f'worker must see exactly one assigned CUDA device, sees {torch.cuda.device_count()}')
 dev=torch.device(a.device);torch.cuda.set_device(dev);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rec=json.loads((root/'extraction_complete.json').read_text())
 if rec.get('complete') is not True or rec.get('cached_windows')!=1040:raise RuntimeError('GC001 feature receipt is incomplete')
 train=load_split(root,'train');devrows=load_split(root,'dev')
 if len(train)!=1008 or len(devrows)!=8:raise RuntimeError('fixed train/dev cache mismatch')
 if not any(np.any((r['labels']>=0)&(r['labels']<18)) for r in train) or not any(np.any(r['labels']==18) for r in train):raise RuntimeError('train requires explicit positive and negative examples')
 # Fixed two-update smoke; throw away model and optimizer afterwards.
 torch.manual_seed(20261);sm=Readout(a.head).cpu().float();sm.to(dev);before=[p.detach().clone() for p in sm.parameters()]
 so=torch.optim.AdamW([{'params':[sm.linear.weight],'weight_decay':1e-4},{'params':[p for n,p in sm.named_parameters() if n!='linear.weight'],'weight_decay':0.}],lr=1e-3,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
 for i in (0,1):
  q=torch.as_tensor(train[i]['q'],device=dev);z=torch.as_tensor(train[i]['z'],device=dev);y=torch.as_tensor(train[i]['labels'],device=dev);loss=weighted_loss(sm(q,z),y);so.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(sm.parameters(),1.,error_if_nonfinite=True);so.step()
 if not any(not torch.equal(x,p) for x,p in zip(before,sm.parameters())):raise RuntimeError('head smoke did not update parameters')
 del sm,so;torch.cuda.empty_cache()
 startup={'status':'PASS','head':a.head,'device':str(dev),'cuda_name':torch.cuda.get_device_name(dev),'two_step_smoke':True,'formal_optimizer_steps':0,'slurm_job_id':os.getenv('SLURM_JOB_ID')}
 (root/f'head_{a.head}_startup.json').write_text(json.dumps(startup,indent=2)+'\n')
 rows_all=[]
 for seed in SEEDS:
  torch.manual_seed(seed);np.random.seed(seed);model=Readout(a.head).cpu().float().to(dev)
  params=[{'params':[model.linear.weight],'weight_decay':1e-4},{'params':[p for n,p in model.named_parameters() if n!='linear.weight'],'weight_decay':0.}]
  opt=torch.optim.AdamW(params,lr=1e-3,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
  best=float('inf');best_epoch=0;best_state=None;pat=0;updates=0;hist=[];reason='max_epochs'
  for epoch in range(50):
   model.train();order=np.random.default_rng(seed+epoch).permutation(1008);epoch_num=epoch_den=0.;seen=effective=0
   for st in range(0,1008,16):
    inds=order[st:st+16];part=[train[int(i)] for i in inds];q=torch.as_tensor(np.stack([r['q'] for r in part]),device=dev);z=torch.as_tensor(np.stack([r['z'] for r in part]),device=dev);y=torch.as_tensor(np.stack([r['labels'] for r in part]),device=dev)
    opt.zero_grad(set_to_none=True);log=model(q,z);loss=weighted_loss(log,y);seen+=len(part);effective+=int((y>=0).sum())
    if loss is None:continue
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite train loss')
    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();updates+=1
    with torch.no_grad(): wt=torch.ones(19,device=dev);wt[18]=.1;v=y>=0;epoch_num+=float((F.cross_entropy(log[v],y[v],reduction='none')*wt[y[v]]).sum());epoch_den+=float(wt[y[v]].sum())
   tm=evaluate(model,train,dev);dm=evaluate(model,devrows,dev);ce=dm['ce']
   if ce is None:raise RuntimeError('dev context CE missing')
   if ce<best-1e-8:
    best=ce;best_epoch=epoch+1;pat=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   else:pat+=1
   row={'head':a.head,'seed':seed,'epoch':epoch+1,'train_ce':epoch_num/epoch_den if epoch_den else None,'dev_ce':ce,'train_joint19_accuracy':tm['joint19_accuracy'],'dev_joint19_accuracy':dm['joint19_accuracy'],'train_conditional18_accuracy':tm['conditional18_accuracy'],'dev_conditional18_accuracy':dm['conditional18_accuracy'],'effective_train_queries':effective,'windows_seen':seen,'optimizer_updates':updates,'best_epoch':best_epoch,'stopped_reason':None}
   hist.append(row);rows_all.append(row)
   if seed==20261 and updates>=20 and startup['formal_optimizer_steps']==0:
    startup.update({'formal_optimizer_steps':updates,'written_unix':time.time()});(root/f'head_{a.head}_startup.json').write_text(json.dumps(startup,indent=2)+'\n')
   if epoch+1>=10 and pat>=10:reason='patience10';break
  if best_state is None:raise RuntimeError('no best checkpoint produced')
  d=root/'heads'/a.head/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True);torch.save({'head':a.head,'seed':seed,'epoch':best_epoch,'state_dict':best_state,'architecture':a.head},d/'best.pt')
  torch.save({'head':a.head,'seed':seed,'epoch':len(hist),'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'architecture':a.head},d/'final.pt')
  hist[-1]['stopped_reason']=reason;(d/'training_history.json').write_text(json.dumps(hist,indent=2)+'\n');(d/'manifest.json').write_text(json.dumps({'head':a.head,'seed':seed,'best_epoch':best_epoch,'final_epoch':len(hist),'stop_reason':reason,'best_dev_ce':best},indent=2)+'\n')
  del model,opt;torch.cuda.empty_cache()
 with (root/f'training_scalars_{a.head}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows_all[0]));w.writeheader();w.writerows(rows_all)
 (root/f'head_{a.head}_complete.json').write_text(json.dumps({'status':'PASS','head':a.head,'seeds':list(SEEDS),'best_checkpoints':9//3,'head_checkpoints':3,'all_seeds_complete':True,'rows':len(rows_all)},indent=2)+'\n')
if __name__=='__main__':main()
