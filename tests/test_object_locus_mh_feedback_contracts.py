"""CPU contracts for the fixed Object-Locus MH Feedback V1 route."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from tokengs.models.object_locus_mh_feedback import (
    MultiHeadFeedbackRegisteredObjectLayer,
    initialize_feedback_layers,
)
from tokengs.models.object_locus_panoptic_v1_controller import (
    RegisteredObjectLayer,
    geometry_bias,
)


class _ToyPanoptic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({f'L{i}': RegisteredObjectLayer() for i in (6,8,10,12)})


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.panoptic = _ToyPanoptic()


def test_rng_isolation_parameter_count_and_route_shapes():
    torch.manual_seed(91)
    random.seed(22)
    np.random.seed(33)
    model = _ToyModel()
    for layer in model.panoptic.layers.values():
        nn.init.zeros_(layer.W_inject.weight)
    torch_cpu = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    initialize_feedback_layers(model, seed=31416)
    assert torch.equal(torch_cpu, torch.get_rng_state())
    if cuda_rng is not None:
        assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    assert py_rng == random.getstate()
    now_np = np.random.get_state()
    assert np_rng[0] == now_np[0] and np.array_equal(np_rng[1], now_np[1])
    assert np_rng[2:] == now_np[2:]
    assert sum(p.numel() for p in model.parameters()) - sum(
        p.numel() for n,p in model.named_parameters() if '.feedback_' not in n
    ) == 1_048_576
    assert sum(p.numel() for p in model.parameters() if p.ndim == 2) > 0
    for layer in model.panoptic.layers.values():
        assert type(layer) is MultiHeadFeedbackRegisteredObjectLayer
        assert layer.W_inject.weight.count_nonzero() == 0
        assert torch.equal(layer.feedback_o.weight, torch.eye(256))

    torch.manual_seed(0)
    b = 1
    h = torch.randn(b,1024,1024)
    a = torch.randn(b,1024,256)
    mu = torch.randn(b,1024,3)
    q = torch.randn(b,102,256)
    c = torch.randn(b,100,3)
    s = torch.rand(b,100,3) + 0.2
    ell = torch.ones(b)
    image = torch.randn(b,2,256)
    f_anchor = torch.randn(b,1024,256)
    mask_embedder = nn.Linear(256,256,bias=False)
    layer = model.panoptic.layers['L6']
    out = layer(h,a,mu,q,c,s,ell,image,mask_embedder,f_anchor,exposure=8)
    attn = out['feedback_attention']
    assert attn.shape == (1,8,1024,103)
    assert out['route'].shape == (1,1024,103)
    assert out['anchor_mask_logits'].shape == (1,1024,102)
    assert out['anchor_membership'].shape == (1,1024,102)
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn[...,0]), atol=1e-6, rtol=1e-6)
    assert torch.allclose(out['route'], attn.mean(1), atol=0, rtol=0)
    expected_mask_logits=f_anchor @ mask_embedder(out['q']).transpose(1,2)
    assert torch.allclose(out['anchor_mask_logits'],expected_mask_logits,atol=1e-6,rtol=1e-6)
    assert torch.allclose(out['anchor_membership'],expected_mask_logits.sigmoid(),atol=1e-6,rtol=1e-6)
    assert torch.allclose(out['evidence_attention'].sum(-1), torch.ones_like(out['evidence_attention'][...,0]), atol=1e-6)
    expected_g = geometry_bias(mu,out['c'],out['s']).transpose(1,2)
    assert expected_g.shape == (1,1024,102)
    assert torch.count_nonzero(expected_g[...,100:]) == 0
    q_ln = torch.nn.functional.layer_norm(out['q'],(256,),eps=1e-5)
    Q = layer.feedback_q(torch.nn.functional.layer_norm(a,(256,),eps=1e-5)).reshape(1,1024,8,32).transpose(1,2)
    K = layer.feedback_k(q_ln).reshape(1,102,8,32).transpose(1,2)
    logits = Q @ K.transpose(-1,-2) / (32 ** 0.5) + expected_g[:,None]
    expected_attn = torch.cat((logits,torch.zeros(1,8,1024,1)),-1).softmax(-1)
    assert torch.allclose(attn,expected_attn,atol=1e-6,rtol=1e-6)
    assert torch.isfinite(out['joint_delta']).all()


def test_fixed_plan_and_optimizer_schedule_contracts():
    from scripts.object_locus_mh_feedback_runtime import checkpoint_epochs, build_model, build_optimizer
    from scripts.object_locus_mask_guided_runtime import lr_multiplier, manifest_and_plan
    manifest, plan = manifest_and_plan()
    assert len(manifest['train_all56']) == 56
    assert plan['global_updates'] == 448 and plan['exposures'] == 3584
    assert [plan['entries'][u]['epoch'] for u in (0,7,55,56,447)] == [0,1,7,8,63]
    assert all(len(entry['rank_windows']) == 8 for entry in plan['entries'])
    assert checkpoint_epochs() == {0:0,56:8,112:16,224:32,448:64}
    assert lr_multiplier(0) == 1/25
    assert lr_multiplier(24) == 1
    assert lr_multiplier(447) == 0.1


def test_fresh_model_matches_c_epoch0_and_optimizer_coverage():
    from scripts.object_locus_mh_feedback_runtime import build_model, build_optimizer
    model,_=build_model('cpu',report=False)
    assert sum(p.numel() for p in model.parameters()) == 573_480_679
    opt=build_optimizer(model)
    ids=[id(p) for group in opt.param_groups for p in group['params']]
    assert len(ids)==len(set(ids))==sum(1 for _ in model.parameters())
    added=[(n,p) for n,p in model.named_parameters() if '.feedback_' in n]
    assert len(added)==16 and sum(p.numel() for _,p in added)==1_048_576
    for name,param in added:
        group=next(group for group in opt.param_groups if any(id(param)==id(x) for x in group['params']))
        assert group['peak_lr']==1e-4 and group['weight_decay']==.05
