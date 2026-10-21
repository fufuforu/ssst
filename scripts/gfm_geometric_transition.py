"""Audited V3 adaptation2 -> V4 geometric-pooling, with name-based AdamW transfer."""
from unittest.mock import patch
import torch
from torch import nn
from scripts.dual_stream_training_policy import validate_joint_warmup_source

FROZEN=('frozen_omega.','sam_proposals.')
NEW=('query_initializer.local_shape.','query_initializer.point_weight.','query_initializer.shape_projection.')
SOURCE_DIGEST='2b07922b1b2c3e8f5dfa6c1591951e043bc418874f2cea379588003ddde6f3e3'
RECIPE='omega_gfm_geometric_sam_pool_v4_adapt2_frozen_recon_understanding4_uwarm1_v1'


def template(version):
    from tokengs.options import config_defaults
    from tokengs.models.object_locus_gfm_dual_stream_v3 import LocusGSGFMDualStreamV3
    from tokengs.models.object_locus_gfm_dual_stream_v4 import LocusGSGFMDualStreamV4
    constructors={3:LocusGSGFMDualStreamV3,4:LocusGSGFMDualStreamV4}
    with patch('tokengs.models.object_locus_dual_stream_v1.FrozenOmega',side_effect=lambda *a,**k:nn.Identity()),patch('tokengs.models.object_locus_dual_stream_v1.FrozenSAM',side_effect=lambda *a,**k:nn.Identity()),torch.device('meta'):
        return constructors[version](config_defaults[f'train_siu3r_object_locus_gfm_dual_stream_v{version}'])


def validate_source(saved,current,progress):
    if current['recipe']!=RECIPE or current['config'].get('query_pooling')!='geometric_sam_v4' or not current['config'].get('freeze_reconstruction'):
        raise ValueError('Geometric transition requires the explicit frozen V4 recipe')
    normalized=dict(current,config=dict(current['config']))
    normalized['config'].pop('query_pooling')
    families=normalized['config']['required_gradient_families']
    if families.count('geometric_query_pooling')!=1:raise ValueError('Missing geometric gradient proof')
    normalized['config']['required_gradient_families']=[x for x in families if x!='geometric_query_pooling']
    validate_joint_warmup_source(saved,normalized,progress)


def plan_transfer(model,source_template,blob,optimizer_groups):
    source=blob['model']
    old={k:v for k,v in source_template.state_dict().items() if not k.startswith(FROZEN)}
    target={k:v for k,v in model.state_dict().items() if not k.startswith(FROZEN)}
    if set(old)!=set(source):raise ValueError('Source key set differs from real V3 template')
    if set(source)-set(target):raise ValueError('V4 removes shared source tensors')
    added=sorted(set(target)-set(source))
    if not added or any(not k.startswith(NEW) for k in added):raise ValueError('Unexpected new tensors outside geometric pooling')
    for name,value in source.items():
        if old[name].shape!=value.shape or target[name].shape!=value.shape:raise ValueError('Shared tensor shape differs: '+name)
    # Real V3 registration reconstructs the old optimizer index -> name map.
    groups=optimizer_groups(source_template);saved_groups=blob['optimizer']['param_groups']
    if len(groups)!=len(saved_groups):raise ValueError('Source optimizer group count differs')
    names={id(p):n for n,p in source_template.named_parameters()}
    states={};seen=set();config=blob['meta']['config']
    for group,saved in zip(groups,saved_groups):
        if any(group[k]!=saved[k] for k in ('family','weight_decay')) or len(group['params'])!=len(saved['params']):
            raise ValueError('Source optimizer registration/family/decay differs')
        family=group['family']
        rate=config['geometry_refinement_lr'] if family=='geometry' else config['understanding_lr'] if family in ('understanding','queries','panoptic') else config['reconstruction_lr']
        if saved['base_lr']!=rate:raise ValueError('Source optimizer rate differs from source metadata')
        for param,index in zip(group['params'],saved['params']):
            if index in seen:raise ValueError('Duplicate source optimizer index')
            seen.add(index);name=names[id(param)]
            if name not in source or source[name].shape!=param.shape:raise ValueError('Source optimizer parameter mismatch')
            if index not in blob['optimizer']['state']:continue
            state=blob['optimizer']['state'][index]
            for key in ('exp_avg','exp_avg_sq'):
                if key not in state or state[key].shape!=param.shape:raise ValueError('Adam moment shape differs: '+name)
            if 'step' not in state:raise ValueError('Missing Adam clock')
            states[name]=state
    if set(blob['optimizer']['state'])-seen:raise ValueError('Unmapped source optimizer state')
    report=dict(shared_tensor_count=len(source),new_tensor_names=added,
                new_parameters=sum(target[k].numel() for k in added),retained_optimizer_states=len(states),
                source_optimizer_states=len(blob['optimizer']['state']),name_based_optimizer_transfer=True)
    return report,states


def transfer(model,optimizer,source_template,blob,optimizer_groups):
    report,states=plan_transfer(model,source_template,blob,optimizer_groups)
    if optimizer.state:raise ValueError('Expected fresh V4 optimizer before explicit transfer')
    target=model.state_dict()
    with torch.no_grad():
        for name,value in blob['model'].items():target[name].copy_(value)
    for name,param in model.named_parameters():
        if name not in states:continue
        optimizer.state[param]={k:(v.detach().clone() if k=='step' else v.to(device=param.device,dtype=param.dtype).clone()) if torch.is_tensor(v) else v for k,v in states[name].items()}
    if len(optimizer.state)!=report['retained_optimizer_states']:raise ValueError('Optimizer states were not fully transferred')
    if any(p in optimizer.state for n,p in model.named_parameters() if n.startswith(NEW)):
        raise ValueError('New pooling parameters must start with fresh Adam moments')
    return report
