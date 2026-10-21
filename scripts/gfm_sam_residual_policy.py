"""Single-variable initializer ablation on the fresh, fully isolated V4 control."""
import copy
import math
import torch
from scripts.gfm_geometric_transition import FROZEN
from scripts.gfm_v4_isolated_policy import configure as isolated,validate_initial as validate_v4
RECIPE='omega_gfm_v4_learned_base_sam_residual_isolated_joint4_v1'
NEW='query_initializer.residual_gate_logit'
MODE='learned_base_rms_normalized_sam_residual_v1'


def configure(config):
    isolated(config)
    config.update(recipe=RECIPE,sam_query_initialization=MODE,sam_residual_initial_gate=.1)
    config['required_gradient_families']=list(config['required_gradient_families'])+['sam_residual_gate']


def validate_initial(saved,current,blob):
    if current['recipe']!=RECIPE or current['config']['recipe']!=RECIPE:raise ValueError('Residual recipe differs')
    target=copy.deepcopy(current);config=target['config']
    if config.pop('sam_query_initialization')!=MODE or config.pop('sam_residual_initial_gate')!=.1:raise ValueError('Residual settings differ')
    if config['required_gradient_families'].count('sam_residual_gate')!=1:raise ValueError('Missing residual gradient proof')
    config['required_gradient_families'].remove('sam_residual_gate')
    isolated(config);target['recipe']=config['recipe']
    validate_v4(saved,target,blob)


def restore_shared_initial(model,optimizer,blob,optimizer_groups):
    from scripts.gfm_geometric_transition import template
    if optimizer.state or blob['optimizer']['state']:raise ValueError('Requires empty Adam')
    target={k:v for k,v in model.state_dict().items() if not k.startswith(FROZEN)};source=blob['model']
    if set(target)-set(source)!={NEW} or set(source)-set(target):raise ValueError('Only scalar residual gate may be added')
    if target[NEW].shape!=torch.Size([]):raise ValueError('Gate must be scalar')
    if any(target[k].shape!=source[k].shape for k in source):raise ValueError('Shared shape differs')
    old=template(4);old_names={id(p):n for n,p in old.named_parameters()};new_names={id(p):n for n,p in model.named_parameters()}
    old_groups=optimizer_groups(old)
    if len(old_groups)!=len(optimizer.param_groups) or len(old_groups)!=len(blob['optimizer']['param_groups']):raise ValueError('Optimizer group count differs')
    for a,b,c in zip(old_groups,optimizer.param_groups,blob['optimizer']['param_groups']):
        if any(a[k]!=b[k] or a[k]!=c[k] for k in ('family','base_lr','weight_decay')):raise ValueError('Optimizer settings differ')
        names_a=[old_names[id(p)] for p in a['params']];names_b=[new_names[id(p)] for p in b['params'] if new_names[id(p)]!=NEW]
        if names_a!=names_b or len(names_a)!=len(c['params']):raise ValueError('Shared optimizer order differs')
    with torch.no_grad():
        for k,value in source.items():target[k].copy_(value)
        target[NEW].fill_(math.log(.1/.9))
    return dict(shared_tensor_count=len(source),new_tensor_names=[NEW],new_parameters=1,shared_initial_weights_exact=True,optimizer_states=0)
