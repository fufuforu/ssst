"""Strict identical warm starts, frozen read-only caches, and local Adam audit."""
import hashlib,json
from pathlib import Path
import torch,numpy as np
from collections import OrderedDict
from scripts.native_joint4_support import task_state,tensor_sha
from scripts.dual_stream_frozen_cache import rgb_key,device_tree,cpu_tree

def validate_config_override(source,runtime,cfg):
    expected=dict(source)
    assert source['max_superpoints']==512 and cfg['max_superpoints']==999
    expected['max_superpoints']=999
    assert runtime==expected, 'Only max_superpoints 512 -> 999 is authorized'
    return dict(max_superpoints=dict(source=512,runtime=999),other_fields_exact=True,no_truncation=True)


def initialize(model,saved,cfg):
    assert saved['progress']==dict(epoch=1,update_in_epoch=0,full_updates=1043)
    assert saved['meta']['cumulative_epochs']==5 and saved['meta']['cumulative_updates']==5215
    assert saved['meta']['architecture']==cfg['architecture']=='QP_ANCHOR1_NATIVE_JOINT4_20261026_V1'
    assert saved['meta']['context_views']==cfg['context_views']==2
    assert saved['meta']['token_count']==cfg['token_count']==4096
    assert saved['meta']['config']==model.config.to_dict()==cfg['config']
    assert cfg['config']['max_superpoints']==cfg['max_superpoints']==512
    assert model.frozen_omega.source_identity==saved['meta']['omega_identity']
    assert model.sam_proposals.source_identity==saved['meta']['sam_identity']
    assert len(saved['rank_rng'])==cfg['world_size']==4 and 'optimizer' not in saved
    target=task_state(model);source=saved['model'];assert set(target)==set(source) and len(target)==760
    with torch.no_grad():
        for name,tensor in target.items():
            value=source[name]
            assert tensor.shape==value.shape and tensor.dtype==value.dtype,name
            tensor.copy_(value);assert torch.equal(tensor.cpu(),value),name
    assert all(source[name].shape==(4096,2048) for name in TOKEN_TABLES)
    assert model.core.reconstruction.opt.num_gs_tokens==4096
    return dict(status='PASSED',copied_task_tensors=760,expanded_task_tensors={},
                all_shapes_and_values_exact=True,source_updates=5215,source_sha256=cfg['source_sha256'],
                new_optimizer=True,context_views=2,token_count=4096,source_cumulative_epochs=5)

TOKEN_TABLES=('core.reconstruction.gs_tokens','core.reconstruction.post_rgb_gs_tokens')
def expanded_tensor(name,value,count):
    if name not in TOKEN_TABLES or count==1024:return value
    assert count==4096 and value.shape==(1024,2048)
    parents=torch.arange(3072)%1024
    extra=value[parents].clone()
    if name==TOKEN_TABLES[0]:
        generator=torch.Generator(device='cpu').manual_seed(20261027)
        extra.add_(torch.randn(extra.shape,generator=generator,dtype=value.dtype)*(.01*value.std().item()))
    return torch.cat((value,extra))


from scripts.resilient_esam_cache import install_caches

def optimizer_audit(adam,step):
    states=adam.optim.state
    assert states
    for state in states.values():
        assert int(state['step'])==step
        assert all(torch.isfinite(x).all() for x in state.values() if torch.is_tensor(x))
    return dict(status='PASSED',local_states=len(states),step=step,all_moments_finite=True,optimizer_saved=False)
