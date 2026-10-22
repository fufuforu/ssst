"""Strict V4init isolated epoch4 -> epoch8 continuation, preserving Adam/RNG."""
import math,hashlib
from scripts.v4init_isolated_recipe import MODEL_KEYS,recipe_identity
POLICY='V4INIT_ISOLATED_EPOCH4_TO8_ADAM_PRESERVED_LR_REWARM_V1'
START=4172
END=8344
PEAK=3e-5

def continuation_lr(step,source_main_lr=1e-5):
    offset=step-START
    if not START<=step<END or not 0<source_main_lr<PEAK:raise ValueError('Outside authorized continuation LR range')
    if offset<100:return source_main_lr+(PEAK-source_main_lr)*offset/99
    fraction=(offset-100)/(END-START-101)
    return PEAK*(.1+.9*.5*(1+math.cos(math.pi*fraction)))

def tensor_tree_equal(a,b):
    import torch,numpy as np
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(tensor_tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(tensor_tree_equal(x,y) for x,y in zip(a,b))
    return a==b

def validate_optimizer_layout(optimizer,saved,*,finite=True):
    import torch
    groups=saved['param_groups']
    if len(optimizer.param_groups)!=len(groups):raise ValueError('Adam parameter group count differs')
    all_ids=[];steps=[]
    for new,old in zip(optimizer.param_groups,groups):
        keys=('family','parameter_names','betas','eps','weight_decay','amsgrad','maximize','foreach','capturable','differentiable','fused')
        if any(new.get(k)!=old.get(k) for k in keys) or len(new['params'])!=len(old['params']):raise ValueError('Adam parameter names/order/policy differs')
        all_ids.extend(old['params'])
        for p,index in zip(new['params'],old['params']):
            state=saved['state'].get(index)
            if state is None:continue # Some Write=0 paths are genuinely unused.
            if set(state)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Unexpected Adam state entries')
            step=float(state['step']);steps.append(step)
            if not math.isfinite(step) or step<1:raise ValueError('Invalid Adam step')
            for key in ('exp_avg','exp_avg_sq'):
                t=state[key]
                if t.shape!=p.shape or t.dtype!=p.dtype:raise ValueError('Adam moment shape/dtype differs')
                if finite and not torch.isfinite(t).all():raise ValueError('Nonfinite Adam moments')
    if len(all_ids)!=len(set(all_ids)) or not set(saved['state']).issubset(all_ids) or not steps:raise ValueError('Adam IDs incomplete/duplicated')
    return sorted(set(steps))

def validate_source(saved,source_cfg,model=None,optimizer=None,*,require_complete=True):
    import torch
    from scripts.v4init_isolated_recipe import check_meta as check_source_meta
    check_source_meta(saved['meta'],source_cfg)
    if saved['meta'].get('continuation_policy'):raise ValueError('Source must be original full4')
    p=saved['progress'];step=p['full_updates']
    if require_complete:
        if p!=dict(epoch=4,update_in_epoch=0,full_updates=START):raise ValueError('Source is not completed epoch4')
    elif not 0<step<=START or step!=p['epoch']*1043+p['update_in_epoch']:
        raise ValueError('Invalid observed original training progress')
    if len(saved['rank_rng'])!=8 or any(set(x)!={'python','numpy','torch','cuda'} for x in saved['rank_rng']):raise ValueError('Original eight-rank RNG incomplete')
    if not saved['optimizer']['state']:raise ValueError('Continuation cannot reset Adam')
    steps=[float(v['step']) for v in saved['optimizer']['state'].values()]
    if any(not math.isfinite(s) or s<1 or s>step for s in steps) or max(steps)!=step:raise ValueError('Source Adam clock differs')
    rates={}
    for group in saved['optimizer']['param_groups']:
        family=group['family'];rate=group['lr']
        if family not in ('main','geometry') or family in rates and rates[family]!=rate:raise ValueError('Source LR family differs')
        rates[family]=rate
    if set(rates)!={'main','geometry'} or not math.isclose(rates['geometry']/rates['main'],.1,rel_tol=1e-12):raise ValueError('Source geometry LR ratio differs')
    if require_complete and not math.isclose(rates['main'],1e-5,rel_tol=1e-5):raise ValueError('Source not at full4 cosine endpoint')
    if model is not None:
        from scripts.unified_query_initial_state import validate_trainable_state
        validate_trainable_state(saved['model'],model)
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in saved['model'].values()):raise ValueError('Nonfinite source model')
    if optimizer is not None:validate_optimizer_layout(optimizer,saved['optimizer'])
    return dict(status='PASSED',progress=dict(p),adam_states=len(steps),adam_steps=sorted(set(steps)),source_learning_rates=rates,rank_rng_count=8)

def source_parameter_digest(model,state):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:h.update(name.encode());h.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def transition_meta(old,current,optimizer,*,checkpoint,sha256):
    # Reconstructed runtime identity must equal the original except epochs/LR/code.
    for k in ('initial_checkpoint','initial_sha256','initial_updates'):current[k]=old[k]
    expected=dict(old,code_sha=current['code_sha'],epochs=8,learning_rate=PEAK)
    if current!=expected:raise ValueError('Continuation changed model/assets/data/task recipe')
    starts={g['family']:g['lr'] for g in optimizer.param_groups}
    for group in optimizer.param_groups:group['base_lr']=PEAK*(.1 if group['family']=='geometry' else 1.)
    current.update(geometry_refinement_lr=3e-6,continuation_policy=POLICY,continuation_checkpoint=str(checkpoint),
        continuation_sha256=sha256,continuation_source_code_sha=old['code_sha'],continuation_start_updates=START,
        continuation_added_updates=END-START,understanding_warmup_restarted=False,optimizer_reset=False,
        source_learning_rates=starts,continuation_lr_warmup=100,continuation_lr_floor=3e-6,
        continuation_geometry_lr_floor=3e-7)
    # Record parameter IDs separately in the actual optimizer; meta keeps names.
    current['optimizer_groups']=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups]
    return current

def check_meta(meta,cfg):
    if meta.get('continuation_policy')!=POLICY or meta['architecture']!=cfg['architecture'] or meta['code_sha']!=cfg['code_sha']:raise ValueError('Continuation identity differs')
    expected=dict(recipe_identity(cfg['model_key']),epochs=8,world_size=8,global_batch=8,learning_rate=PEAK,
        geometry_refinement_lr=3e-6,continuation_start_updates=START,continuation_added_updates=END-START,
        understanding_warmup_restarted=False,optimizer_reset=False,continuation_lr_warmup=100,
        continuation_lr_floor=3e-6,continuation_geometry_lr_floor=3e-7)
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Continuation training recipe differs')
    if meta['continuation_source_code_sha']!=cfg['source_code_sha'] or meta['initial_sha256']!=cfg['initial_checkpoint_sha256']:raise ValueError('Source lineage differs')
    if meta['continuation_sha256']!=cfg['continuation_sha256']:raise ValueError('Epoch4 checkpoint digest differs')
    for k in ('manifest_sha256','heldout_sha256'):
        if meta[k]!=cfg[k]:raise ValueError('Continuation data differs')
    if 'training_window_manifest_sha256' in cfg:
        from scripts.training_windows_2ctx4novel import sha256
        if meta.get('training_window_manifest_sha256')!=cfg['training_window_manifest_sha256'] or sha256(cfg['training_window_manifest'])!=cfg['training_window_manifest_sha256']:raise ValueError('Continuation six-view manifest differs')
