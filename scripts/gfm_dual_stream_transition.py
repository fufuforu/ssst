"""Explicit V1@50 -> GFM V2 transfer, with named AdamW state mapping."""
from unittest.mock import patch
import hashlib
import torch
from torch import nn
from tokengs.models.object_locus_gfm_dual_stream_v2 import REMOVED_ENCODER_PREFIXES,NEW_GFM_PREFIXES


SOURCE50_SHA256='f711044d0241c6fd949353e45bb5a9b92630409dd4df259239084402bc84d3ea'

def verify_source(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()==SOURCE50_SHA256

def legacy_template():
    from tokengs.options import config_defaults
    from tokengs.models.object_locus_dual_stream_v1 import LocusGSDualStreamV1
    # Real old parameter registration/attributes without allocating or loading backbones.
    with patch('tokengs.models.object_locus_dual_stream_v1.FrozenOmega',side_effect=lambda *a,**k:nn.Identity()),patch('tokengs.models.object_locus_dual_stream_v1.FrozenSAM',side_effect=lambda *a,**k:nn.Identity()),torch.device('meta'):
        return LocusGSDualStreamV1(config_defaults['train_siu3r_object_locus_dual_stream_v1'])


def transfer_plan(model,source):
    target={k:v for k,v in model.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))}
    retained=sorted(set(target)&set(source));new=sorted(set(target)-set(source));removed=sorted(set(source)-set(target))
    mismatched=[k for k in retained if target[k].shape!=source[k].shape]
    if mismatched:raise ValueError('Changed retained tensor shapes: '+str(mismatched))
    if any(not k.startswith(NEW_GFM_PREFIXES) for k in new):raise ValueError('Unexpected new tensor outside explicit GFM whitelist')
    if any(not k.startswith(REMOVED_ENCODER_PREFIXES) for k in removed):raise ValueError('Unexpected removed tensor outside old encoder whitelist')
    if not new or not removed:raise ValueError('Expected genuine architecture transition')
    return dict(retained=retained,new=new,removed=removed,
        retained_parameters=sum(target[k].numel() for k in retained),new_parameters=sum(target[k].numel() for k in new),removed_parameters=sum(source[k].numel() for k in removed))


def transfer(model,optimizer,blob,optimizer_groups):
    plan=transfer_plan(model,blob['model'])
    target=model.state_dict()
    with torch.no_grad():
        for name in plan['retained']:target[name].copy_(blob['model'][name])
    # Recover source optimizer IDs using the real old constructor on meta, not
    # guesses based on state_dict order or permissive partial optimizer loading.
    template=legacy_template();groups=optimizer_groups(template)
    names={id(p):name for name,p in template.named_parameters()}
    if len(groups)!=len(blob['optimizer']['param_groups']):raise ValueError('Old optimizer group count differs')
    source_states={}
    for current,saved in zip(groups,blob['optimizer']['param_groups']):
        for key in ('family','base_lr','weight_decay'):
            if current[key]!=saved[key]:raise ValueError('Old optimizer group contract differs: '+key)
        if len(current['params'])!=len(saved['params']):raise ValueError('Old optimizer parameter count differs')
        for param,index in zip(current['params'],saved['params']):
            name=names[id(param)]
            if name not in blob['model'] or param.shape!=blob['model'][name].shape:raise ValueError('Old parameter registration mismatch: '+name)
            if index in blob['optimizer']['state']:source_states[name]=blob['optimizer']['state'][index]
    copied=0
    for name,param in model.named_parameters():
        if name not in source_states or name not in plan['retained']:continue
        state=source_states[name]
        for key in ('exp_avg','exp_avg_sq'):
            if key in state and state[key].shape!=param.shape:raise ValueError('Adam moment shape differs: '+name)
        optimizer.state[param]={key:(value.detach().clone() if key=='step' else value.to(device=param.device,dtype=param.dtype).clone()) if torch.is_tensor(value) else value for key,value in state.items()}
        copied+=1
    plan['retained_optimizer_states']=copied
    return plan


def transition_receipt(metadata,plan):
    """Merge the intentionally repeated state count, rejecting contradictory fields."""
    overlap=set(metadata)&set(plan)
    if overlap-{'retained_optimizer_states'}:raise ValueError('Unexpected duplicated transition receipt field')
    if any(metadata[key]!=plan[key] for key in overlap):raise ValueError('Transition receipt state counts disagree')
    return {**metadata,**plan,'optimizer_moments_retained_by_parameter_name':True,'new_parameters_have_fresh_moments':True}
