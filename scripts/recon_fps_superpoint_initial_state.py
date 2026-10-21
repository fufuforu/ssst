"""Explicit zero-only legacy -> FPS-superpoint migration; only the learned FPS embedding is new."""
BANK_KEY='core.reconstruction_fps.learned_embedding'

def initialize_fps_superpoint_from_legacy_zero(saved,model,*,manifest_sha256,heldout_sha256,world):
    import torch
    if model.architecture_name!='OMEGA_3D_FPS1024_SUPERPOINT_RECON_V1':raise ValueError('Wrong target FPS-superpoint architecture')
    meta=saved['meta']
    if saved['progress']!=dict(epoch=0,update_in_epoch=0,full_updates=0) or saved['optimizer']['state']:
        raise ValueError('FPS-superpoint initialization requires true zero updates and empty Adam')
    expected=dict(architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',config=model.config.to_dict(),
        manifest_sha256=manifest_sha256,heldout_sha256=heldout_sha256,global_batch=8,
        omega_identity=model.frozen_omega.source_identity,
        sam_identity=model.sam_proposals.source_identity if model.sam_proposals is not None else None)
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Legacy FPS-superpoint source assets/data/config/architecture differ')
    if meta['world_size']!=8 or len(saved['rank_rng'])!=8 or world not in (1,2,4,8):raise ValueError('Expected original eight-rank zero and a divisor rank count')
    target={k:v for k,v in model.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))}
    source=saved['model']
    if BANK_KEY not in target or set(source)!=(set(target)-{BANK_KEY}):raise ValueError('Only the new reconstruction FPS embedding may differ')
    if any(target[k].shape!=source[k].shape for k in source):raise ValueError('Legacy tensor shape mismatch')
    with torch.no_grad():
        for k,v in source.items():target[k].copy_(v)
    return dict(status='LEGACY_ZERO_PLUS_FPS_SUPERPOINT_EMBEDDING',copied_tensor_count=len(source),copied_keys=sorted(source),new_parameter=BANK_KEY,
        embedding_shape=list(target[BANK_KEY].shape),source_world=8,target_world=world,rank_rng_selection=list(range(world)),
        understanding_parameters_preserved=True,source_updates=0,empty_adam=True)
