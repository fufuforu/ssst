"""Sealed V4-init/NoCA recipe, exact zero migration and geometry LR policy."""
ARCH='OMEGA_V4INIT_NO_ANCHOR_CA_ISOLATED_V4_RECIPE_V1'
KEY='dual_v4init_isolated_noca'
SOURCE_ARCH='OMEGA_OLD_A4_V4_RECON_INITIALIZATION_JOINT_V1'

def recipe_identity():
    return dict(gradient_isolation=True,gradient_isolation_method='detached_reconstruction_reads_and_mask_geometry',
        understanding_adapters=False,training_view_protocol='2context+2novel',training_windows=8337,training_scenes=1191,
        supervised_layers=[6,12],layer_weights=[1/3,2/3],semantic_memory_resolution=128,
        understanding_seed_feature_source='old_shared_projected_cloud256',
        reconstruction_initialization='v4_fps1024_learned_base_xyz_radius_rgb_raw_gfm2048',reconstruction_anchor_count=1024,
        object_anchor_attention='none',object_fm_attention='learned_cross_attention',
        geometry_routing='normalized_geometric_distance_softmax_v1',write_strength=0.,
        geometry_refinement_lr=1e-5,gradient_clip='global_norm_1_as_early_v4',
        understanding_weight=1.,understanding_warmup=1043)

def removed_keys():
    return {f'core.panoptic.layers.L{l}.anchor_attention.{p}.{k}':shape
        for l in (6,8,10,12) for p in ('Q','K','V','O')
        for k,shape in [('weight',(256,256)),('bias',(256,))]}

def initialize_from_v4init_zero(saved,model,*,manifest_sha256,heldout_sha256,world):
    import torch
    if model.architecture_name!=ARCH:raise ValueError('Wrong target architecture')
    if saved['progress']!=dict(epoch=0,update_in_epoch=0,full_updates=0) or saved['optimizer']['state']:
        raise ValueError('Requires V4init ZERO with empty Adam; trained endpoints forbidden')
    meta=saved['meta']
    expected=dict(architecture=SOURCE_ARCH,config=model.config.to_dict(),manifest_sha256=manifest_sha256,
        heldout_sha256=heldout_sha256,world_size=8,global_batch=8,initial_updates=0,
        initial_sha256='3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744',
        omega_identity=model.frozen_omega.source_identity,
        sam_identity=model.sam_proposals.source_identity if model.sam_proposals is not None else None)
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Source V4init data/config/assets differ')
    if len(saved['rank_rng'])!=8 or world not in (1,2,4,8):raise ValueError('Invalid rank RNG coverage')
    source=saved['model'];target={k:v for k,v in model.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))}
    removed=removed_keys()
    if set(source)-set(target)!=set(removed) or set(target)-set(source):raise ValueError('Only Anchor CA removal allowed')
    if any(tuple(source[k].shape)!=shape for k,shape in removed.items()):raise ValueError('Removed tensor shape differs')
    if any(source[k].shape!=v.shape for k,v in target.items()):raise ValueError('Retained tensor shape differs')
    with torch.no_grad():
        for k,v in target.items():v.copy_(source[k])
    return dict(status='V4INIT_ZERO_MIGRATED',source_updates=0,empty_adam=True,source_world=8,target_world=world,
        copied_keys=sorted(target),removed_keys=sorted(removed),new_keys=[],**recipe_identity())

def optimizer_groups(model):
    # Keep the original weight-decay rule, changing ONLY mu/rho head base LR.
    groups={};names={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        geometry=name.startswith(('core.anchor_decoder.refine_mu.','core.anchor_decoder.refine_rho.'))
        family='geometry' if geometry else 'main';base_lr=1e-5 if geometry else 1e-4
        wd=0. if p.ndim<2 or getattr(p,'_no_weight_decay',False) else .05
        key=(family,base_lr,wd);groups.setdefault(key,[]).append(p);names.setdefault(key,[]).append(name)
    if not any(k[0]=='geometry' for k in groups):raise ValueError('Geometry parameter group absent')
    return [dict(params=ps,lr=lr,base_lr=lr,weight_decay=wd,family=family,parameter_names=names[(family,lr,wd)])
            for (family,lr,wd),ps in groups.items()]

def check_meta(meta,cfg):
    if meta['architecture']!=ARCH or meta['code_sha']!=cfg['code_sha']:raise ValueError('Code/architecture differs')
    for k in ('manifest_sha256','heldout_sha256'):
        if meta[k]!=cfg[k]:raise ValueError('Data identity differs: '+k)
    if meta['initial_sha256']!=cfg['initial_checkpoint_sha256']:raise ValueError('Zero source differs')
    expected=dict(epochs=4,world_size=8,global_batch=8,learning_rate=1e-4,**recipe_identity())
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Recipe identity differs')
