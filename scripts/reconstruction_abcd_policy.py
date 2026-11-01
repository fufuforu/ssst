"""Strict name-based model/Adam migration; no permissive partial loading."""
import copy,hashlib
import torch
from tokengs.models.reconstruction_abcd import ARCHS
B8_SHA='bd1afe7e27ca01533ff5bfef6cd218a99c4f2294de08399a40cd2acf6eb98bc5'
ZERO_SHA='bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397'
BASE_ARCH='OMEGA_V4INIT_NO_ANCHOR_CA_ISOLATED_V4_RECIPE_V1'
EXPANDED='core.v4_initialization.gs_tokens'

def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def exact_equal(left,right):
    # Checkpoint tensors are CPU/mmap; target weights/moments may live on
    # any rank's CUDA device. Compare exact values on a common device, without
    # dtype conversion or tolerance, and never retain a GPU copy of the source.
    if left.shape!=right.shape or left.dtype!=right.dtype:return False
    return torch.equal(left.detach().cpu(),right.detach().cpu())

def task_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))}

def expanded(value,kind):
    if value.shape!=(1024,1024):raise ValueError('Unexpected expansion shape')
    if kind=='parameter':
        generator=torch.Generator(device='cpu').manual_seed(20261025)
        added=value.repeat(3,1)+torch.randn(3072,1024,generator=generator,dtype=value.dtype)*1e-4
    else:added=torch.zeros(3072,1024,dtype=value.dtype)
    return torch.cat((value,added),0)

def initialize(model,saved,arm):
    zero=arm=='D';expected=0 if zero else 8344
    if saved['progress']!=dict(epoch=0 if zero else 8,update_in_epoch=0,full_updates=expected):raise ValueError('Source progress differs')
    if saved['meta']['architecture']!=('OMEGA_OLD_A4_V4_RECON_INITIALIZATION_JOINT_V1' if zero else BASE_ARCH):raise ValueError('Wrong source architecture')
    if zero and saved['optimizer']['state']:raise ValueError('D requires empty zero Adam')
    if model.frozen_omega.source_identity!=saved['meta']['omega_identity'] or (model.sam_proposals.source_identity if model.sam_proposals is not None else None)!=saved['meta']['sam_identity']:raise ValueError('Frozen assets differ')
    source=saved['model'];target=task_state(model)
    copied=[];new=[];changed=[]
    # D only carries zero-step understanding/common inputs. All reconstruction
    # modules, including their adapters, retain fresh seeded initialization.
    def common(name):
        return name.startswith(('gfm_projection.','core.initializer.understanding.','core.semantic_memory.','core.panoptic.'))
    allowed_new={n for n in target if n.startswith('core.appearance.')} if arm=='C' else set()
    if not zero and (set(target)-set(source)!=allowed_new or set(source)-set(target)):raise ValueError('ABC non-whitelisted keys')
    with torch.no_grad():
        for name,value in target.items():
            if zero and not common(name):new.append(name);continue
            if name in allowed_new:new.append(name);continue
            if name not in source:raise ValueError('Missing common zero tensor: '+name)
            old=source[name]
            if arm=='B' and name==EXPANDED:
                value.copy_(expanded(old,'parameter'));changed.append(name)
                if not exact_equal(value[:1024],old):raise ValueError('Original token rows changed')
            else:
                if value.shape!=old.shape:raise ValueError('Unexpected shape: '+name)
                value.copy_(old);copied.append(name)
            if name in copied and not exact_equal(value,old):raise ValueError('Copy mismatch '+name)
    # Freeze AFTER loading Adam layout so all original slots remain represented.
    if arm in ('A','B'):
        reconstruction={id(p) for p in model.core.parameter_groups()['reconstruction']}
        for p in model.parameters():p.requires_grad_(id(p) in reconstruction)
    elif arm=='C':
        ids={id(p) for p in model.core.appearance.parameters()}
        for p in model.parameters():p.requires_grad_(id(p) in ids)
    return dict(status='PASSED',arm=arm,source_updates=expected,copied_keys=copied,new_keys=new,expanded_keys=changed,
        zero_common_module_policy='only understanding/shared zero tensors' if zero else None,
        copied_tensor_sha256={n:tensor_sha(target[n]) for n in copied},trainable_names=[n for n,p in model.named_parameters() if p.requires_grad])

def optimizer(model,saved,arm):
    params=dict(model.named_parameters());groups=[]
    if arm!='D':
        for old in saved['optimizer']['param_groups']:
            names=old['parameter_names']
            if any(n not in params for n in names):raise ValueError('Original optimizer name missing')
            g={k:copy.deepcopy(v) for k,v in old.items() if k not in ('params',)}
            g['params']=[params[n] for n in names];groups.append(g)
        if arm=='C':
            for wd in (0.,.05):
                names=[n for n,p in params.items() if n.startswith('core.appearance.') and (0. if p.ndim<2 else .05)==wd]
                groups.append(dict(params=[params[n] for n in names],parameter_names=names,weight_decay=wd,lr=3e-5,base_lr=3e-5,family='appearance'))
    else:
        for wd in (0.,.05):
            names=[n for n,p in params.items() if p.requires_grad and (0. if p.ndim<2 else .05)==wd]
            groups.append(dict(params=[params[n] for n in names],parameter_names=names,weight_decay=wd,lr=1e-4,base_lr=1e-4,family='main'))
    opt=torch.optim.AdamW(groups,lr=1e-4,betas=(.9,.95),eps=1e-8)
    copied=[];padded=[]
    if arm!='D':
        by_name={n:i for g in saved['optimizer']['param_groups'] for n,i in zip(g['parameter_names'],g['params'])}
        for name,index in by_name.items():
            old=saved['optimizer']['state'].get(index)
            if old is None:continue
            state={}
            for key,value in old.items():
                result=expanded(value,'moment') if arm=='B' and name==EXPANDED and key in ('exp_avg','exp_avg_sq') else value.clone()
                state[key]=result.to(params[name].device) if key!='step' else result.cpu()
                if name==EXPANDED and arm=='B' and key in ('exp_avg','exp_avg_sq'):
                    if not exact_equal(state[key][:1024],value) or torch.count_nonzero(state[key][1024:]):raise ValueError('Adam expansion differs')
                    padded.append(name+'.'+key)
                elif not exact_equal(state[key],value):raise ValueError('Adam copy differs')
            opt.state[params[name]]=state;copied.append(name)
    if arm=='D' and opt.state:raise ValueError('D optimizer not empty')
    return opt,dict(status='PASSED',copied_adam_names=copied,zero_padded_moments=padded,empty_at_zero=arm=='D',frozen_slots_retained=arm!='D')
