"""Same fresh V4 experiment, changing only loss-to-parameter gradient routing."""
RECIPE='omega_gfm_geometric_sam_pool_v4_fresh_joint4_isolated_gradients_uwarm1_v1'
SOURCE_CODE='5a3ff5fa70deb56c8d6bafe390d66f6171b50424'


def configure(config,reconstruction_understanding_scale=0.):
    if reconstruction_understanding_scale not in (0.,.05):raise ValueError("Unsupported routing scale")
    config.update(recipe=RECIPE,gradient_isolation='recon_loss_to_recon_under_loss_to_under',
                  gradient_synchronization='explicit_sum_div_world_fp32_unused_aware')

    if reconstruction_understanding_scale:
        config.update(recipe=RECIPE.replace('isolated_gradients','recon_under005_gradients'),
                      gradient_isolation='recon_plus005_under_to_recon_full_under_to_under',
                      reconstruction_understanding_gradient_scale=reconstruction_understanding_scale)


def validate_initial(saved,current,blob):
    from scripts.gfm_v4_direct_joint_policy import RECIPE as BASE_RECIPE
    expected=dict(current,config=dict(current['config']))
    scale=expected['config'].pop('reconstruction_understanding_gradient_scale',0.)
    if scale not in (0.,.05):raise ValueError('Unsupported routing scale')
    if scale and expected['config']['gradient_isolation']!='recon_plus005_under_to_recon_full_under_to_under':raise ValueError('Routing policy mismatch')
    expected['config'].pop('gradient_isolation');expected['config'].pop('gradient_synchronization')
    expected.update(recipe=BASE_RECIPE,execution_git_sha=SOURCE_CODE)
    expected['config']['recipe']=BASE_RECIPE
    if saved!=expected:raise ValueError('Untrained initial state model/data/schedule differs from baseline')
    p=blob['progress']
    if (p['short_updates'],p['full_updates'],p['epoch'],p['update_in_epoch'],p['phase'])!=(0,0,0,0,'full'):
        raise ValueError('Only the exact zero-training initial checkpoint is allowed')
    if p.get('adapt_updates',0) or any(p['epoch_counts']) or any(p['total_counts']):raise ValueError('Initial checkpoint contains trained exposures')
    if blob['optimizer']['state']:raise ValueError('Initial checkpoint contains optimizer history')
    if len(blob['rank_rng'])!=8 or current['world_size']!=8:raise ValueError('Expected eight source and target ranks')
