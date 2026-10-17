"""Strict zero-step transfer; never accepts an overfitted short-run endpoint."""

def validate_zero_initial(saved,model,*,manifest_sha256,heldout_sha256,world):
    meta=saved['meta'];progress=saved['progress']
    if progress!=dict(epoch=0,update_in_epoch=0,full_updates=0) or saved['optimizer']['state']:
        raise ValueError('Only zero updates and empty Adam may initialize full training')
    expected=dict(architecture=model.architecture_name,config=model.config.to_dict(),
        manifest_sha256=manifest_sha256,heldout_sha256=heldout_sha256,world_size=world,
        omega_identity=model.frozen_omega.source_identity,
        sam_identity=model.sam_proposals.source_identity if model.sam_proposals is not None else None)
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Zero-step architecture/assets/data/world differ')
    if len(saved['rank_rng'])!=world:raise ValueError('Zero-step RNG rank coverage differs')
