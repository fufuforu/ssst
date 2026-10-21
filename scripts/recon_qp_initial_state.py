"""Explicit zero-only legacy -> Q+P migration; only the learned Q bank is new."""
BANK_KEY='core.reconstruction_qp.learned_q'

def initialize_recon_qp_from_legacy_zero(saved,model,*,manifest_sha256,heldout_sha256,world):
    import torch
    if model.architecture_name!='OMEGA_3D_SUPERPOINT_RECON_QP_V1':raise ValueError('Wrong target Q+P architecture')
    meta=saved['meta']
    if saved['progress']!=dict(epoch=0,update_in_epoch=0,full_updates=0) or saved['optimizer']['state']:
        raise ValueError('Q+P initialization requires true zero updates and empty Adam')
    expected=dict(architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',config=model.config.to_dict(),
        manifest_sha256=manifest_sha256,heldout_sha256=heldout_sha256,global_batch=8,
        omega_identity=model.frozen_omega.source_identity,
        sam_identity=model.sam_proposals.source_identity if model.sam_proposals is not None else None)
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Legacy Q+P source assets/data/config/architecture differ')
    if meta['world_size']!=8 or len(saved['rank_rng'])!=8 or world not in (1,2,4,8):raise ValueError('Expected original eight-rank zero and a divisor rank count')
    target={k:v for k,v in model.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))}
    source=saved['model']
    if BANK_KEY not in target or set(source)!=(set(target)-{BANK_KEY}):raise ValueError('Only the new reconstruction Q bank may differ')
    if any(target[k].shape!=source[k].shape for k in source):raise ValueError('Legacy tensor shape mismatch')
    with torch.no_grad():
        for k,v in source.items():target[k].copy_(v)
    return dict(status='LEGACY_ZERO_PLUS_LEARNED_RECON_Q',copied_tensor_count=len(source),copied_keys=sorted(source),new_parameter=BANK_KEY,
        learned_q_shape=list(target[BANK_KEY].shape),source_world=8,target_world=world,rank_rng_selection=list(range(world)),
        understanding_parameters_preserved=True,source_updates=0,empty_adam=True)
