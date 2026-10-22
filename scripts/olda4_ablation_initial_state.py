"""Strict true-zero migrations only, with exact tensor-key/shape whitelists."""
V4_ARCH='OMEGA_OLD_A4_V4_RECON_INITIALIZATION_JOINT_V1'
NOCA_ARCH='OMEGA_OLD_A4_NO_ANCHOR_CA_GEOMETRIC_JOINT_V1'
MODEL_KEYS={V4_ARCH:'dual_olda4_v4init',NOCA_ARCH:'dual_olda4_noca'}


def recipe_identity(key):
    v4=key=='dual_olda4_v4init'
    if key not in MODEL_KEYS.values():raise ValueError('Unknown old A4 arm')
    return dict(gradient_isolation=False,understanding_adapters=False,training_view_protocol='2context+2novel',
        training_windows=8337,training_scenes=1191,supervised_layers=[6,8,10,12],layer_weights=[.1,.2,.3,.4],
        semantic_memory_resolution=128,understanding_seed_feature_source='old_shared_projected_cloud256',
        reconstruction_initialization='v4_fps1024_learned_base_xyz_radius_rgb_raw_gfm2048' if v4 else 'old_superpoint256_to1024',
        reconstruction_anchor_count=1024 if v4 else 'variable_superpoints',
        object_anchor_attention='learned_cross_attention' if v4 else 'none',object_fm_attention='learned_cross_attention',
        geometry_routing='learned_attention_weights' if v4 else 'normalized_geometric_distance_softmax_v1',write_strength=0.)


def initialize_olda4_ablation(saved,model,*,manifest_sha256,heldout_sha256,world):
    import torch
    from tokengs.models.unified_superpoints import SuperpointQueryInitializer
    if model.architecture_name not in MODEL_KEYS:raise ValueError('Wrong old A4 migration target')
    if saved['progress']!=dict(epoch=0,update_in_epoch=0,full_updates=0) or saved['optimizer']['state']:
        raise ValueError('Requires original zero updates and empty Adam')
    expected=dict(architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',config=model.config.to_dict(),
        manifest_sha256=manifest_sha256,heldout_sha256=heldout_sha256,world_size=8,global_batch=8,
        omega_identity=model.frozen_omega.source_identity,
        sam_identity=model.sam_proposals.source_identity if model.sam_proposals is not None else None)
    if any(saved['meta'].get(k)!=v for k,v in expected.items()):raise ValueError('Original A4 assets/data/config differ')
    if len(saved['rank_rng'])!=8 or world not in (1,2,4,8):raise ValueError('Invalid original rank RNG coverage')
    source=saved['model'];target={k:v for k,v in model.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))}
    removed={};added={}
    if model.architecture_name==V4_ARCH:
        with torch.random.fork_rng(devices=[]):
            removed={'core.initializer.reconstruction.'+k:v.shape for k,v in SuperpointQueryInitializer(model.config).state_dict().items()}
        removed.update({'core.reconstruction_seed_projection.weight':(1024,256),'core.reconstruction_seed_projection.bias':(1024,)})
        prefix='core.v4_initialization.'
        added={prefix+'gs_tokens':(1024,1024),prefix+'anchor_content.0.weight':(1024,7),prefix+'anchor_content.0.bias':(1024,),
            prefix+'anchor_content.2.weight':(1024,1024),prefix+'anchor_content.2.bias':(1024,),
            prefix+'anchor_visual_projection.0.weight':(2048,),prefix+'anchor_visual_projection.0.bias':(2048,),
            prefix+'anchor_visual_projection.1.weight':(1024,2048),prefix+'anchor_visual_projection.1.bias':(1024,)}
    else:
        removed={f'core.panoptic.layers.L{l}.anchor_attention.{projection}.{kind}':shape
            for l in (6,8,10,12) for projection in ('Q','K','V','O')
            for kind,shape in [('weight',(256,256)),('bias',(256,))]}
    if set(source)-set(target)!=set(removed) or set(target)-set(source)!=set(added):raise ValueError('Unexpected migration keys')
    if any(source[k].shape!=v for k,v in removed.items()) or any(target[k].shape!=v for k,v in added.items()):raise ValueError('Migration tensor shape differs')
    kept=set(source)&set(target)
    if any(source[k].shape!=target[k].shape for k in kept):raise ValueError('Retained tensor shape differs')
    with torch.no_grad():
        for k in kept:target[k].copy_(source[k])
    return dict(status='OLD_A4_ZERO_MIGRATED',source_updates=0,empty_adam=True,source_world=8,target_world=world,
        copied_keys=sorted(kept),removed_keys=sorted(removed),new_keys=sorted(added),**recipe_identity(MODEL_KEYS[model.architecture_name]))
