"""Fresh V4 joint4: official frozen backbones, no adaptation or source weights."""
RECIPE='omega_gfm_geometric_sam_pool_v4_fresh_direct_joint4_uwarm1_v1'


def configure_direct_joint(config):
    config.update(recipe=RECIPE,query_pooling='geometric_sam_v4',
        reconstruction_lr=1e-4,geometry_refinement_lr=1e-5,understanding_lr=1e-4,
        understanding_loss_warmup_updates=1043,full_monitor_every_updates=100,
        initialization_policy='official_Omega_SAM_only_all_trainable_branches_fresh')


def validate_direct_arguments(args):
    if args.architecture!='gfm_v4' or args.adapt_epochs or args.transition_from or args.staged_from or args.warm_joint_from or args.geometric_from or args.joint_warmup or args.freeze_reconstruction:
        raise ValueError('Fresh V4 direct joint4 forbids adaptation, source checkpoints and reconstruction freezing')
