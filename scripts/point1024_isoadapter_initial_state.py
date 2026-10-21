"""Explicit A4 true-zero migration: remove unused recon initializer, add adapters."""
ARCH='OMEGA_POINT1024_SOURCE256_ISOLATED_RESIDUAL_ADAPTER_V1'
REMOVED='core.initializer.reconstruction.'
ADAPTERS='core.understanding_adapters.'


def initialize_point1024_isoadapter_from_legacy_zero(saved,model,*,manifest_sha256,heldout_sha256,world):
    import torch
    from tokengs.models.unified_superpoints import SuperpointQueryInitializer
    if model.architecture_name!=ARCH:raise ValueError('Wrong isolated point1024 target')
    if saved['progress']!=dict(epoch=0,update_in_epoch=0,full_updates=0) or saved['optimizer']['state']:
        raise ValueError('Requires original zero updates and empty Adam')
    meta=saved['meta']
    expected=dict(architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',config=model.config.to_dict(),
        manifest_sha256=manifest_sha256,heldout_sha256=heldout_sha256,global_batch=8,
        omega_identity=model.frozen_omega.source_identity,
        sam_identity=model.sam_proposals.source_identity if model.sam_proposals is not None else None)
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Source architecture/assets/data/config differ')
    if meta['world_size']!=8 or len(saved['rank_rng'])!=8 or world not in (1,2,4,8):raise ValueError('Expected eight-rank original zero')
    target={k:v for k,v in model.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))}
    source=saved['model']
    # Enumerate the exact removed initializer from the validated source config;
    # do not tolerate arbitrary missing keys under the prefix. Preserve RNG.
    with torch.random.fork_rng(devices=[]):
        removed={REMOVED+k:v.shape for k,v in SuperpointQueryInitializer(model.config).state_dict().items()}
    added={ADAPTERS+f'L{layer}.net.{suffix}':shape for layer in (6,8,10,12)
        for suffix,shape in [('0.weight',(64,1024)),('0.bias',(64,)),('2.weight',(1024,64)),('2.bias',(1024,))]}
    if set(source)-set(target)!=set(removed) or set(target)-set(source)!=set(added):raise ValueError('Unexpected added/removed tensors')
    if any(source[k].shape!=shape for k,shape in removed.items()) or any(target[k].shape!=shape for k,shape in added.items()):raise ValueError('Initializer/adapter shape mismatch')
    kept=set(source)&set(target)
    if any(source[k].shape!=target[k].shape for k in kept):raise ValueError('Retained tensor shape mismatch')
    for layer in (6,8,10,12):
        for suffix in ('weight','bias'):
            if torch.count_nonzero(target[ADAPTERS+f'L{layer}.net.2.{suffix}']):raise ValueError('Residual adapters must start at identity')
    with torch.no_grad():
        for key in kept:target[key].copy_(source[key])
    return dict(status='LEGACY_ZERO_TO_POINT1024_ISOLATED_ADAPTER',copied_tensor_count=len(kept),copied_keys=sorted(kept),
        removed_keys=sorted(removed),new_keys=sorted(added),source_world=8,target_world=world,
        rank_rng_selection=list(range(world)),source_updates=0,empty_adam=True,
        seed_feature_source='exact_source_pixel_shared_projection_256',adapter_input_detached=True,
        adapter_residual_output_detached=False,shared_understanding_gradient_disabled=True)
