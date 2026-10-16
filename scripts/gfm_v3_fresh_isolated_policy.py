"""V3 pooling ablation: retain every shared tensor from the untrained V4 baseline."""
import copy
import torch
from scripts.gfm_geometric_transition import NEW,template,FROZEN
from scripts.gfm_v4_isolated_policy import configure as configure_isolated,validate_initial as validate_v4
RECIPE='omega_gfm_semantic_sam_pool_v3_fresh_joint4_isolated_gradients_uwarm1_v1'


def configure(config):
    from scripts.gfm_v4_direct_joint_policy import configure_direct_joint
    configure_direct_joint(config);configure_isolated(config)
    config.pop('query_pooling',None)
    config['required_gradient_families']=[x for x in config['required_gradient_families'] if x!='geometric_query_pooling']
    config['recipe']=RECIPE


def validate_initial(saved,current,blob):
    if current['recipe']!=RECIPE or current['config']['recipe']!=RECIPE:raise ValueError('V3 recipe mismatch')
    normalized=copy.deepcopy(current)
    configure_isolated(normalized['config']);normalized['recipe']=normalized['config']['recipe']
    normalized['config']['query_pooling']='geometric_sam_v4'
    normalized['config']['required_gradient_families'].append('geometric_query_pooling')
    validate_v4(saved,normalized,blob)


def restore_shared_initial(model,optimizer,blob,optimizer_groups):
    if optimizer.state or blob['optimizer']['state']:raise ValueError('Ablation requires empty Adam')
    target={k:v for k,v in model.state_dict().items() if not k.startswith(FROZEN)};source=blob['model']
    removed=sorted(set(source)-set(target))
    expected_removed={k for k in template(4).state_dict() if k.startswith(NEW)}
    if set(removed)!=expected_removed or set(target)-set(source):raise ValueError('Unexpected V3/V4 key difference')
    if sum(source[k].numel() for k in removed)!=29505:raise ValueError('Unexpected removed pooling size')
    if any(target[k].shape!=source[k].shape for k in target):raise ValueError('Shared tensor shape mismatch')
    old_model=template(4);old_groups=optimizer_groups(old_model)
    if len(old_groups)!=len(blob['optimizer']['param_groups']):raise ValueError('Source optimizer group mismatch')
    old_names={id(p):n for n,p in old_model.named_parameters()};new_names={id(p):n for n,p in model.named_parameters()}
    if len(old_groups)!=len(optimizer.param_groups):raise ValueError('Target optimizer group mismatch')
    for old,saved,new in zip(old_groups,blob['optimizer']['param_groups'],optimizer.param_groups):
        for key in ('family','base_lr','weight_decay'):
            if old[key]!=saved[key] or new[key]!=saved[key]:raise ValueError('Optimizer settings differ')
        if len(old['params'])!=len(saved['params']):raise ValueError('Source optimizer registration mismatch')
        if [old_names[id(p)] for p in old['params'] if old_names[id(p)] not in removed]!=[new_names[id(p)] for p in new['params']]:raise ValueError('Shared optimizer order differs')
    with torch.no_grad():
        for k,value in target.items():value.copy_(source[k])
    return dict(shared_tensor_count=len(target),removed_tensor_names=removed,removed_parameters=29505,
                shared_initial_weights_exact=True,optimizer_states=0)
