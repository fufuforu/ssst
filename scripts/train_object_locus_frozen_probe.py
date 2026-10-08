#!/usr/bin/env python3
"""CPU-only frozen-feature H1/H2/H3 diagnostic head training."""
from __future__ import annotations
import argparse, csv, json, os, time
import hashlib
import platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.object_locus_frozen_probe_contract import labels_from_context

ATTEMPT=Path(os.environ.get('TASK_ATTEMPT_ROOT','/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt02'))
SEEDS=(20261,20262,20263)

class Readout(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.kind=kind
        if kind in ('H1','H2'):
            self.ln=nn.LayerNorm(256,eps=1e-5); self.linear=nn.Linear(256,19)
        elif kind=='H3':
            self.lnq=nn.LayerNorm(256,eps=1e-5); self.lnz=nn.LayerNorm(256,eps=1e-5); self.linear=nn.Linear(512,19)
        else: raise ValueError(kind)
        nn.init.xavier_uniform_(self.linear.weight); nn.init.zeros_(self.linear.bias)
    def forward(self,q,z):
        if self.kind=='H1': x=self.ln(q)
        elif self.kind=='H2': x=self.ln(z)
        else: x=torch.cat((self.lnq(q),self.lnz(z)),dim=-1)
        return self.linear(x)

def load_split(attempt,split):
    if split=='train':
        manifest=json.loads((attempt/'cohort_manifest.json').read_text())['train']
        root=attempt/'cache/train'
        out=[]
        for i,w in enumerate(manifest):
            p=root/f'{i:04d}.npz'
            with np.load(p) as d:
                labs=labels_from_context(d['iou'],d['gt_classes'].tolist())['labels']
                out.append({'q':d['q'].copy(),'z':d['z'].copy(),'labels':labs,'pclass':d['pclass'].copy(),'logits':d['logits'].copy(),'window':w})
        return out
    manifest=json.loads((attempt/'cohort_manifest.json').read_text())[split]
    rows=[]
    for i,w in enumerate(manifest):
        prefix=f'{split}_{i:04d}_{w["scene"]}_c{"_".join(map(str,w["context"]))}'
        with np.load(attempt/'features'/f'{prefix}.npz') as d: q,z=d['q'].copy(),d['z'].copy()
        with np.load(attempt/'cache/dev_test'/f'{prefix}_iou.npz') as d:
            labs=labels_from_context(d['iou'],d['gt_classes'].tolist())['labels']
        rows.append({'q':q,'z':z,'labels':labs,'window':w})
    return rows

def batches(data,order,batch_size=16):
    for start in range(0,len(order),batch_size): yield order[start:start+batch_size]

def weighted_ce(logits,labels):
    valid=labels>=0
    if not valid.any(): return None
    weights=torch.ones(19,dtype=logits.dtype,device=logits.device); weights[18]=.1
    losses=F.cross_entropy(logits[valid],labels[valid],reduction='none')
    sample=weights[labels[valid]]
    return (losses*sample).sum()/sample.sum()

def evaluate(head,data):
    head.eval(); ce_num=0.; ce_den=0.; correct=joint=cond=pos=0
    with torch.no_grad():
        for row in data:
            q=torch.from_numpy(row['q']); z=torch.from_numpy(row['z']); y=torch.from_numpy(row['labels'])
            logits=head(q,z); mask=y>=0
            if mask.any():
                weights=torch.ones(19); weights[18]=.1
                loss=F.cross_entropy(logits[mask],y[mask],weight=weights,reduction='sum')
                ce_num+=float(loss); ce_den+=float(weights[y[mask]].sum())
            positive=(y>=0)&(y<18); pred=logits.argmax(-1); c_pred=logits[:,:18].argmax(-1)
            joint+=int(((pred==y)&positive).sum()); pos+=int(positive.sum())
            cond+=int(((c_pred==y)&positive).sum())
    return {'ce':ce_num/ce_den if ce_den else None,'joint_acc':joint/pos if pos else None,
            'conditional_acc':cond/pos if pos else None,'positive_queries':pos}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--attempt',type=Path,default=ATTEMPT); args=ap.parse_args(); root=args.attempt
    if torch.cuda.is_available(): raise RuntimeError('CPU job must set CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    envpath=root/'cpu_runtime.json'
    env={'python':__import__('sys').version,'executable':__import__('sys').executable,'torch':torch.__version__,
         'platform':platform.platform(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
         'torch_cuda_available':torch.cuda.is_available(),'torch_threads':torch.get_num_threads(),
         'omp_num_threads':os.environ.get('OMP_NUM_THREADS'),'mkl_num_threads':os.environ.get('MKL_NUM_THREADS'),
         'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_job_partition':os.environ.get('SLURM_JOB_PARTITION'),
         'slurm_job_node':os.environ.get('SLURMD_NODENAME'),'slurm_cpus':os.environ.get('SLURM_CPUS_PER_TASK'),
         'slurm_mem':os.environ.get('SLURM_MEM_PER_NODE')}
    envpath.write_text(json.dumps(env,indent=2)+'\n')
    receipt=json.loads((root/'extraction_complete.json').read_text())
    if receipt.get('status')!='PASS' or receipt.get('cached_windows')!=1040: raise RuntimeError('extraction receipt incomplete')
    cm=json.loads((root/'cache_manifest.json').read_text())
    cb=cm['combined_feature_bundle'];cbp=Path(cb['path']);h=hashlib.sha256()
    with cbp.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    if cbp.stat().st_size!=cb['size'] or h.hexdigest()!=cb['sha256']:raise RuntimeError('combined dev/test vector SHA mismatch')
    for row in cm['files']:
        for path_key,hash_key,size_key in [('path','sha256','size'),('feature_path','feature_sha256','feature_size'),('iou_path','iou_sha256','iou_size')]:
            if path_key not in row: continue
            path=Path(row[path_key]);h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1<<20),b''):h.update(block)
            if path.stat().st_size!=row[size_key] or h.hexdigest()!=row[hash_key]: raise RuntimeError(f'cache SHA mismatch: {path}')
    train=load_split(root,'train'); dev=load_split(root,'dev')
    if len(train)!=1008 or len(dev)!=8: raise RuntimeError('train/dev cache count mismatch')
    if not any(np.any((r['labels']>=0)&(r['labels']<18)) for r in train) or not any(np.any(r['labels']==18) for r in train):
        raise RuntimeError('train split needs at least one positive and explicit negative')
    out=root/'heads'; out.mkdir(parents=True,exist_ok=True)
    scalar_path=root/'training_scalars.csv'; scalar_path.parent.mkdir(parents=True,exist_ok=True)
    # Two-step CPU contract on two cached windows; this state is discarded.
    torch.manual_seed(20261); smoke=Readout('H1').cpu().float()
    smoke_opt=torch.optim.AdamW([{'params':[smoke.linear.weight],'weight_decay':1e-4},
        {'params':[p for n,p in smoke.named_parameters() if n!='linear.weight'],'weight_decay':0.0}],lr=1e-3,betas=(.9,.999),eps=1e-8)
    frozen_before=[(train[i]['q'].copy(),train[i]['z'].copy()) for i in (0,1)]
    smoke_before={n:p.detach().clone() for n,p in smoke.named_parameters()}
    smoke_updates=0
    for i in (0,1):
        q=torch.from_numpy(train[i]['q']).float(); z=torch.from_numpy(train[i]['z']).float()
        y=torch.from_numpy(train[i]['labels']).long(); loss=weighted_ce(smoke(q,z),y)
        if loss is None: continue
        smoke_opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(smoke.parameters(),1.,error_if_nonfinite=True); smoke_opt.step(); smoke_updates+=1
    if smoke_updates != 2: raise RuntimeError('CPU cached two-step smoke lacked effective labels')
    changed=[n for n,p in smoke.named_parameters() if not torch.equal(smoke_before[n],p.detach())]
    optimizer_ids={id(p) for g in smoke_opt.param_groups for p in g['params']}
    if not changed or optimizer_ids!={id(p) for p in smoke.parameters()}:
        raise RuntimeError('CPU smoke optimizer coverage/update contract failed')
    if any(not np.array_equal(train[i]['q'],frozen_before[j][0]) or not np.array_equal(train[i]['z'],frozen_before[j][1]) for j,i in enumerate((0,1))):
        raise RuntimeError('CPU smoke modified frozen features')
    (root/'smoke_cpu.json').write_text(json.dumps({'cpu_two_optimizer_steps':smoke_updates,'head_only_parameters_changed':True,
        'features_unchanged':True,'status':'PASS'},indent=2)+'\n')
    fields=['head','seed','epoch','train_ce','dev_ce','train_joint_acc','dev_joint_acc','train_conditional_acc','dev_conditional_acc','effective_train_queries','windows_seen','optimizer_updates','best_epoch','stopped_reason']
    allrows=[]
    for seed in SEEDS:
      for kind in ('H1','H2','H3'):
        torch.manual_seed(seed); np.random.seed(seed)
        head=Readout(kind).cpu().float()
        params=[{'params':[head.linear.weight],'weight_decay':1e-4},
                {'params':[p for n,p in head.named_parameters() if n!='linear.weight'],'weight_decay':0.0}]
        opt=torch.optim.AdamW(params,lr=1e-3,betas=(.9,.999),eps=1e-8)
        best=float('inf'); best_epoch=None; best_state=None; patience=0; updates=0; reason='max_epochs'
        rows=[]
        for epoch in range(50):
            head.train(); order=np.random.default_rng(seed+epoch).permutation(1008); seen=eff=0
            for inds in batches(train,order):
                q=torch.from_numpy(np.stack([train[i]['q'] for i in inds])).float()
                z=torch.from_numpy(np.stack([train[i]['z'] for i in inds])).float()
                y=torch.from_numpy(np.stack([train[i]['labels'] for i in inds])).long()
                opt.zero_grad(set_to_none=True); logits=head(q,z); loss=weighted_ce(logits,y); seen+=len(inds); eff+=int((y>=0).sum())
                if loss is None: continue
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite probe loss')
                loss.backward(); nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True); opt.step(); updates+=1
            trainm=evaluate(head,train); devm=evaluate(head,dev); dce=devm['ce']
            if dce is None: raise RuntimeError('dev context has no fixed labelled query')
            improved=dce < best-1e-8
            if improved:
                best=float(dce);best_epoch=epoch+1;best_state={k:v.detach().clone() for k,v in head.state_dict().items()};patience=0
                savedir=out/kind/f'seed_{seed}'; savedir.mkdir(parents=True,exist_ok=True)
                torch.save({'head':kind,'seed':seed,'epoch':best_epoch,'state_dict':best_state,'architecture':kind},savedir/'best.pt')
            else: patience+=1
            row={'head':kind,'seed':seed,'epoch':epoch+1,'train_ce':trainm['ce'],'dev_ce':dce,
                'train_joint_acc':trainm['joint_acc'],'dev_joint_acc':devm['joint_acc'],
                'train_conditional_acc':trainm['conditional_acc'],'dev_conditional_acc':devm['conditional_acc'],
                'effective_train_queries':eff,'windows_seen':seen,'optimizer_updates':updates,
                'best_epoch':best_epoch,'stopped_reason':None}
            rows.append(row); allrows.append(row)
            if kind=='H1' and seed==20261 and updates>=20 and not (root/'cpu_startup_confirmation.json').exists():
                cpu={'status':'PASS','head':'H1','seed':seed,'formal_optimizer_steps':updates,
                    'parameters_only':True,'device':'cpu','written_unix':time.time()}
                (root/'cpu_startup_confirmation.json').write_text(json.dumps(cpu,indent=2)+'\n')
                start=root/'startup_confirmation.json'
                combined=json.loads(start.read_text()) if start.exists() else {}
                combined['cpu_probe']=cpu
                start.write_text(json.dumps(combined,indent=2)+'\n')
            if epoch+1>=10 and patience>=10: reason='patience10'; break
        row['stopped_reason']=reason
        savedir=out/kind/f'seed_{seed}'; savedir.mkdir(parents=True,exist_ok=True)
        torch.save({'head':kind,'seed':seed,'epoch':len(rows),'state_dict':head.state_dict(),'architecture':kind},savedir/'final.pt')
        rows[-1]['stopped_reason']=reason
        (savedir/'training_history.json').write_text(json.dumps(rows,indent=2)+'\n')
        # Save head metadata with explicit exact parameter count.
        (savedir/'manifest.json').write_text(json.dumps({'head':kind,'seed':seed,'parameters':sum(p.numel() for p in head.parameters()),
            'best_epoch':best_epoch,'final_epoch':len(rows),'stop_reason':reason,'best_dev_ce':best},indent=2)+'\n')
    with scalar_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(allrows)
    (root/'head_manifest.json').write_text(json.dumps({'seeds':SEEDS,'primary_seed':20261,'heads':{
        'H0':{'input':'q','structure':'original model LN256 eps1e-5 + Linear256->19','parameters':5395,'optimizer':False},
        'H1':{'input':'q','structure':'LN256 eps1e-5 + Linear256->19','parameters':5395},
        'H2':{'input':'z','structure':'LN256 eps1e-5 + Linear256->19','parameters':5395},
        'H3':{'input':'q,z','structure':'two independent LN256 eps1e-5, concat512, Linear512->19','parameters':10771}},
        'parameters':{'H0':5395,'H1':5395,'H2':5395,'H3':10771},
        'epochs_max':50,'updates_per_epoch':63,'optimizer_updates_max_per_head':3150},indent=2)+'\n')

if __name__=='__main__': main()
