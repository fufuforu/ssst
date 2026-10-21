"""Joint backward with 20x lower understanding loss, all other settings unchanged."""
RECIPE='omega_gfm_geometric_sam_pool_v4_fresh_joint4_under005_uwarm1_v1'
SOURCE_CODE='5a3ff5fa70deb56c8d6bafe390d66f6171b50424'


def configure(config):
    config.update(recipe=RECIPE,understanding_loss_scale=.05)


def validate_initial(saved,current,blob):
    from scripts.gfm_v4_direct_joint_policy import RECIPE as BASE_RECIPE
    if current['config'].get('understanding_loss_scale')!=.05 or current['recipe']!=RECIPE:
        raise ValueError('Expected explicit understanding-weight0.05 recipe')
    expected=dict(current,config=dict(current['config']))
    expected['config'].pop('understanding_loss_scale')
    expected.update(recipe=BASE_RECIPE,execution_git_sha=SOURCE_CODE)
    expected['config']['recipe']=BASE_RECIPE
    if saved!=expected:raise ValueError('Weighted source model/data/schedule differs from the ordinary joint baseline')
    p=blob['progress']
    if (p['short_updates'],p['full_updates'],p['epoch'],p['update_in_epoch'],p['phase'])!=(0,0,0,0,'full'):
        raise ValueError('Only the exact zero-training initial checkpoint is allowed')
    if p.get('adapt_updates',0) or any(p['epoch_counts']) or any(p['total_counts']):raise ValueError('Initial checkpoint contains trained exposures')
    if blob['optimizer']['state']:raise ValueError('Initial checkpoint contains optimizer history')
    if len(blob['rank_rng'])!=8 or current['world_size']!=8:raise ValueError('Expected eight source and target ranks')
