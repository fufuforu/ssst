"""Fixed protocol utilities for the single registered official-source run."""
from __future__ import annotations
import hashlib,json,math,os,random,subprocess,sys
from pathlib import Path
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from tokengs.options import config_defaults
from tokengs.data.siu3r_processed import SIU3RProcessedProvider,validate_batch_frame_order
from torch.utils.data import default_collate

BASELINE='b00b94fdc45af6d2f78c55f05671aaa75906204f'
OFFICIAL_SHA='9da24a896c4787d0bd90882fd2c9f001b3e153a5'
EVALUATOR_SHA='8ea80166be76854f938e90521f1a5b688b755c87'
RUN=Path('/space/mawb/ssst/workspace_recon_diag/official_source_scannet_v1/run')
REPORT=Path('/space/mawb/ssst/group_plus/official_source_scannet_v1')
DATA=Path('/space/mawb/SIU3R/data/scannet')
LEGACY=Path('/space/mawb/ssst/workspace_recon_diag/full_train/run_lrcap2e5/best_monitor/model.pt')
LEGACY_SHA='5fcf71b969b01c2603194a85521f95e5e48eafa3f3759ce7839d339caaa9634f'
SCHEDULE=dict(total_updates=50000,warmup=2000,base_lr=1e-4,final_multiplier=.02,cap_from_update=2501,cap=2e-5)
KEEP={0,2500,12500,25000,47500,50000}

def write_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,default=str)+'\n');os.replace(temporary,path)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024**2):h.update(chunk)
    return h.hexdigest()

def verify_vendor():
    root=REPO/'third_party/locusgs_official'
    m=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    assert m['commit']==OFFICIAL_SHA
    assert set(m['files'])=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='SOURCE_MANIFEST.json' and '__pycache__' not in p.parts}
    for name,digest in m['files'].items():
        if sha256(root/name)!=digest:raise RuntimeError(f'Upstream source changed: {name}')
    return m

def options():
    return config_defaults['train_siu3r_official_locusgs_recon'].evolve(
        dataset_kwargs={'data_root':str(DATA)},lr=1e-4,pct_start_steps=2000,
        enc_depth=3,dec_depth=12,enc_embed_dim=1024,token_dim=1024,
        enc_num_heads=16,num_gs_tokens=1024,
        img_size=(256,256),patch_size=8,znear=.025,zfar=125.,bg_color='grey')

def seed_all():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)

def move(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(move(v,device) for v in value)
    return value

def lr_at(t):
    if not 0<=t<50000:raise ValueError('0-based optimizer-update index outside registered budget')
    base=1e-4*(t+1)/2000 if t<2000 else 1e-4*(.02+.98*.5*(1+math.cos(math.pi*(t-2000)/48000)))
    return base if t<2500 else min(base,2e-5)

def optimizer_for(model):
    decay=[];nodecay=[];seen=set()
    for name,p in model.named_parameters():
        if not p.requires_grad:raise RuntimeError(f'Frozen parameter prohibited: {name}')
        if id(p) in seen:raise RuntimeError(f'Duplicate optimizer parameter: {name}')
        seen.add(id(p))
        (nodecay if p.ndim==1 or getattr(p,'_no_weight_decay',False) else decay).append(p)
    assert seen=={id(p) for p in model.parameters()}
    return torch.optim.AdamW([{'params':decay,'weight_decay':.05},{'params':nodecay,'weight_decay':0.}],lr=lr_at(0),betas=(.9,.95),eps=1e-8)

def verify_data():
    lists={split:sorted(p.name for p in (DATA/split).iterdir() if p.is_dir()) for split in ('train','val')}
    if len(lists['train'])!=1201 or len(lists['val'])!=312 or set(lists['train'])&set(lists['val']):
        raise RuntimeError('Dataset scene list/count has changed; STOP')
    records=json.loads((DATA/'val_pair.json').read_text())
    if len(records)!=1860:raise RuntimeError('Official val_pair record count has changed; STOP')
    info={}
    for split,names in lists.items():
        p=REPORT/f'{split}_scenes.txt';payload=('\n'.join(names)+'\n').encode()
        if p.exists() and p.read_bytes()!=payload:raise RuntimeError('Scene list changed since preflight')
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
        info[split]={'count':len(names),'sha256':sha256(p),'scenes':names}
    info['val_pair_sha256']=sha256(DATA/'val_pair.json');write_json(REPORT/'data_manifest.json',info)
    return lists

def get_batch(provider,index,device):
    batch=default_collate([provider[index]])
    validate_batch_frame_order(batch,provider.last_pair,phase='official-wrapper')
    return move(batch,device),dict(provider.last_pair)

def monitor_batches(opt,device):
    entries=[]
    for i,scene in enumerate(('scene0011_00','scene0246_00','scene0458_01','scene0621_00')):
        p=SIU3RProcessedProvider(opt,root=str(DATA/'val'),subset=[scene],training=True)
        p.pair_rng.seed(42+1000+i);batch,pair=get_batch(p,0,device)
        entries.append((batch,pair))
    return entries

@torch.no_grad()
def monitor(model,entries,step):
    from tokengs.models.input_types import split_data,ModelInput,ModelInputDecoder
    from tokengs.models.canonical_recon import ssim_loss
    training=model.training;model.eval();rows=[]
    for batch,pair in entries:
        value,_=split_data(batch,model.opt)
        d=ModelInputDecoder(cam_view=batch['cam_view_all'],intrinsics=batch['intrinsics_all'])
        output=model.forward_reconstruction_only(ModelInput(value.encoder,d),d)
        pred=output['render']['images_pred'];gt=batch['images_all'];row={'step':step,'pair':pair}
        for scope,sl in [('context',slice(0,2)),('novel',slice(2,None))]:
            a,b=pred[:,sl],gt[:,sl]
            row[scope+'_PSNR']=float(-10*(a-b).square().mean().clamp_min(1e-12).log10())
            row[scope+'_SSIM']=float(1-2*ssim_loss(a.reshape(-1,3,256,256),b.reshape(-1,3,256,256)))
        rows.append(row)
    model.train(training);return rows

def finite_update(model,optimizer,batch,t):
    optimizer.zero_grad(set_to_none=True)
    output,metrics=model.step_loss(batch,step=t)
    for name,v in metrics.items():
        if not torch.isfinite(v).all():raise FloatingPointError(f'Nonfinite metric before update: {name}')
    for name,v in [('gaussians',output['gaussians']),*output['render'].items()]:
        if name=='means2d_pred':continue # Undefined culled renderer coordinates are never used in loss.
        if not torch.isfinite(v).all():raise FloatingPointError(f'Nonfinite output before update: {name}')
    metrics['loss'].backward()
    for name,p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError(f'Nonfinite gradient: {name}')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    for group in optimizer.param_groups:group['lr']=lr_at(t)
    optimizer.step()
    return output,metrics,float(norm)

def source_sha():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()

def parameter_hash(model):
    h=hashlib.sha256()
    for name,p in model.state_dict().items():
        h.update(name.encode());h.update(str(p.dtype).encode());h.update(str(tuple(p.shape)).encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
