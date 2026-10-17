"""Transfer only the original serial zero-step state; initialize exactly three new tensors."""
import torch

NEW_KEYS={'core.identity_gate_logit','core.identity_projection.weight','core.identity_projection.bias'}
BASE_ARCH='OMEGA_3D_SUPERPOINT_SERIAL_QUERY_ISOLATED_V1'
RESIDUAL_ARCH='OMEGA_3D_SUPERPOINT_SERIAL_QUERY_IDENTITY_RESIDUAL_V1'


def rng_equal(a,b):
    import numpy as np
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(rng_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(rng_equal(x,y) for x,y in zip(a,b))
    return a==b


def prepare_zero(saved,model,*,windows,manifest_sha256,heldout_sha256):
    old=saved['meta']
    if saved['progress']!=dict(epoch=0,update_in_epoch=0,full_updates=0) or saved['optimizer']['state']:
        raise ValueError('Residual source must have zero updates and empty Adam')
    expected=dict(architecture=BASE_ARCH,config=model.config.to_dict(),world_size=4,global_batch=8,
        short_validation=True,epochs=128,epoch_probes=False,learning_rate=1e-4,understanding_weight=.05,understanding_warmup=20,
        training_windows=windows,manifest_sha256=manifest_sha256,heldout_sha256=heldout_sha256,
        omega_identity=model.frozen_omega.source_identity,sam_identity=model.sam_proposals.source_identity)
    if model.architecture_name!=RESIDUAL_ARCH or any(old.get(k)!=v for k,v in expected.items()):
        raise ValueError('Serial zero-step architecture/assets/data/cohort/training policy differs')
    if len(saved['rank_rng'])!=4:raise ValueError('Four-rank RNG coverage required')
    current={k:v for k,v in model.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))}
    if set(current)-set(saved['model'])!=NEW_KEYS or set(saved['model'])-set(current):
        raise ValueError('Only the three identity residual parameters may be new')
    if any(v.shape!=current[k].shape or v.dtype!=current[k].dtype or not torch.isfinite(v).all() for k,v in saved['model'].items()):
        raise ValueError('Original serial tensor shape/dtype/finite check failed')
    if not torch.equal(current['core.identity_projection.weight'],torch.eye(model.config.dim,dtype=current['core.identity_projection.weight'].dtype)) or torch.count_nonzero(current['core.identity_projection.bias']):
        raise ValueError('Residual projection must start at identity')
    if abs(float(current['core.identity_gate_logit'].sigmoid())-.1)>1e-6:raise ValueError('Residual gate must start at .1')
    state=dict(saved['model']);state.update({k:current[k].detach().cpu().clone() for k in NEW_KEYS})
    return dict(saved,model=state,optimizer=dict(state={},param_groups=[]),meta=dict(old,architecture=RESIDUAL_ARCH))
