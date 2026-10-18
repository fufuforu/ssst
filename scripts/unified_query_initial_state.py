"""Strict zero-step transfer; never accepts an overfitted short-run endpoint."""

def expand_identical_zero_rank_rng(saved,world):
    """Change rank count only for a fresh zero-step state with identical rank RNGs."""
    import copy
    import numpy as np
    import torch
    if saved['progress']!=dict(epoch=0,update_in_epoch=0,full_updates=0) or saved['optimizer']['state']:
        raise ValueError('Rank expansion requires zero updates and empty Adam')
    states=saved['rank_rng'];old=saved['meta']['world_size']
    if not states or len(states)!=old or world<old or world%old:
        raise ValueError('Invalid zero-step rank expansion')
    def equal(a,b):
        if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
        if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    if not all(equal(states[0],s) for s in states):raise ValueError('Source zero-step rank RNGs differ')
    result=dict(saved,meta=dict(saved['meta'],world_size=world),rank_rng=[copy.deepcopy(states[i%old]) for i in range(world)])
    return result

def validate_zero_initial(saved,model,*,manifest_sha256,heldout_sha256,world):
    meta=saved['meta'];progress=saved['progress']
    if progress!=dict(epoch=0,update_in_epoch=0,full_updates=0) or saved['optimizer']['state']:
        raise ValueError('Only zero updates and empty Adam may initialize full training')
    expected=dict(architecture=model.architecture_name,config=model.config.to_dict(),
        manifest_sha256=manifest_sha256,heldout_sha256=heldout_sha256,world_size=world,
        omega_identity=model.frozen_omega.source_identity,
        sam_identity=model.sam_proposals.source_identity if model.sam_proposals is not None else None)
    expected.update(getattr(model,'extra_checkpoint_metadata',{}))
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Zero-step architecture/assets/data/world differ')
    if len(saved['rank_rng'])!=world:raise ValueError('Zero-step RNG rank coverage differs')
