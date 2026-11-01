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
    assert saved['progress']==dict(epoch=4,update_in_epoch=0,full_updates=4172)
    assert saved['meta']['architecture']=='QP_ANCHOR1_NATIVE_JOINT4_20261026_V1'
    override=validate_config_override(saved['meta']['config'],model.config.to_dict(),cfg)
    assert model.frozen_omega.source_identity==saved['meta']['omega_identity']
    assert model.sam_proposals.source_identity==saved['meta']['sam_identity']
    target=task_state(model);source=saved['model'];assert set(target)==set(source) and len(target)==760
    count=cfg['token_count'];proof={}
    with torch.no_grad():
        for name,tensor in target.items():
            value=expanded_tensor(name,source[name],count)
            assert tensor.shape==value.shape and tensor.dtype==value.dtype,name
            tensor.copy_(value);assert torch.equal(tensor.cpu(),value),name
            if value.shape!=source[name].shape:
                assert torch.equal(value[:1024],source[name])
                proof[name]=dict(shape=list(value.shape),prefix_exact=True,sha256=tensor_sha(value))
    assert model.core.reconstruction.opt.num_gs_tokens==count
    return dict(status='PASSED',config_override=override,copied_task_tensors=760-len(proof),expanded_task_tensors=proof,
                all_shapes_and_values_exact=count==1024,unchanged_tensors_exact=True,old_token_prefix_exact=True,
                source_updates=4172,source_sha256=cfg['source_sha256'],new_optimizer=True,context_views=cfg['context_views'],token_count=count)

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
