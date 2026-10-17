"""Experiment B: completed reconstruction -> SAM-supported Anchor query seed."""
import copy
from scripts.gfm_sam_residual_policy import configure as residual,validate_initial as validate_residual,restore_shared_initial,MODE as RESIDUAL_MODE,RECIPE as RESIDUAL_RECIPE
MODE='updated_anchor_sam_residual_v1'
RECIPE='omega_gfm_v4_updated_anchor_sam_residual_isolated_joint4_v1'
SETTINGS=dict(decoder_execution_order='reconstruction_then_understanding_replay',anchor_seed_content='detached_final_anchors_existing_panoptic_projection',anchor_seed_neighbors=4,anchor_seed_max_distance_over_scene=.1,anchor_seed_temperature_over_scene=.05)


def configure(config):
    residual(config)
    config.update(recipe=RECIPE,sam_query_initialization=MODE,**SETTINGS)
    config['required_gradient_families'].append('anchor_seed_projection')


def validate_initial(saved,current,blob):
    target=copy.deepcopy(current);config=target['config']
    if current['recipe']!=RECIPE or config['recipe']!=RECIPE or config['sam_query_initialization']!=MODE:raise ValueError('Anchor seed recipe differs')
    for key,value in SETTINGS.items():
        if config.pop(key)!=value:raise ValueError('Anchor seed setting differs: '+key)
    if config['required_gradient_families'].count('anchor_seed_projection')!=1:raise ValueError('Missing Anchor seed gradient proof')
    config['required_gradient_families'].remove('anchor_seed_projection')
    config.update(recipe=RESIDUAL_RECIPE,sam_query_initialization=RESIDUAL_MODE);target['recipe']=RESIDUAL_RECIPE
    validate_residual(saved,target,blob)
