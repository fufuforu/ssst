"""Opt-in fixed-reconstruction understanding probes; production graph unchanged."""
from __future__ import annotations
import types
import torch
from torch import nn
import torch.nn.functional as F
from .input_types import EncoderLatent
from .object_locus_interaction_controls_v1 import install_interaction

ARMS = ('joint_full', 'u_read', 'u_no_anchor_read', 'u_fusion')


class AlignedGeometryFusion(nn.Module):
    """Shared pointwise residual on the three query-decoder feature scales.

Frozen VGGT adapter memory is view-major 37x37, matching the same two context
RGB crops. Resize within each view; never mix views or use target features.
The zero final projection preserves the exact initial visual features.
Mask features for lifting/readout remain the original frozen visual features.
"""
    def __init__(self):
        super().__init__()
        self.geo_norm = nn.LayerNorm(1024)
        self.geo_proj = nn.Linear(1024, 256)
        self.visual_norm = nn.LayerNorm(256)
        self.mix = nn.Linear(512, 256)
        self.out = nn.Linear(256, 256)
        nn.init.xavier_uniform_(self.geo_proj.weight); nn.init.zeros_(self.geo_proj.bias)
        nn.init.xavier_uniform_(self.mix.weight); nn.init.zeros_(self.mix.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, scales, memory):
        b = memory.shape[0]
        if memory.shape != (b, 2738, 1024) or len(scales) != 3:
            raise ValueError('fusion memory/view/scale contract differs')
        geo = self.geo_proj(self.geo_norm(memory.float())).reshape(b, 2, 37, 37, 256)
        geo = geo.permute(0, 1, 4, 2, 3).flatten(0, 1)
        result = []
        for visual in scales:
            if visual.ndim != 5 or visual.shape[:3] != (b, 2, 256):
                raise ValueError('query feature correspondence differs')
            resized = F.interpolate(geo, visual.shape[-2:], mode='bilinear', align_corners=False)
            resized = resized.reshape(b, 2, 256, *visual.shape[-2:]).permute(0, 1, 3, 4, 2)
            v = visual.permute(0, 1, 3, 4, 2)
            delta = self.out(F.gelu(self.mix(torch.cat((self.visual_norm(v), resized), -1))))
            result.append(visual + delta.permute(0, 1, 4, 2, 3))
        return tuple(result)


def _memory_for_fusion(model, result):
    memory, keys, values = model.vggt_memory_adapter(result.patch_layers)
    model.understanding.__dict__['_probe_geometry_memory'] = memory.detach()
    return EncoderLatent(keys=keys, values=values)


def install(model, arm):
    if arm not in ARMS or hasattr(model, 'understanding_probe_arm'):
        raise ValueError('unknown or already installed probe')
    model.understanding_probe_arm = arm
    install_interaction(model.panoptic, 'bidirectional' if arm == 'joint_full' else 'forward_only')
    if arm == 'joint_full':
        return
    # Pixel features stay fixed, while the pretrained query decoder/mask embedder
    # and native panoptic readout remain trainable.
    model.requires_grad_(False)
    model.understanding.mask2former.transformer_module.requires_grad_(True)
    model.panoptic.requires_grad_(True)
    for layer in model.panoptic.layers.values():
        layer.W_inject.requires_grad_(False)
    if arm == 'u_no_anchor_read':
        def no_feature_residual(module, args, output):
            ev, weights = output
            return torch.zeros_like(ev), weights
        for layer in model.panoptic.layers.values():
            # Preserve native attention weights/geometry updates and all GS
            # segmentation support; ablate only the anchor feature residual.
            layer.anchor_attention.requires_grad_(False)
            layer.anchor_attention.register_forward_hook(no_feature_residual)
    if arm == 'u_fusion':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(314159)
            model.understanding.geometry_fusion = AlignedGeometryFusion()
        model._latent_from_vggt = types.MethodType(_memory_for_fusion, model)
        def fuse_before_query(module, args, kwargs):
            memory = model.understanding.__dict__.pop('_probe_geometry_memory')
            kwargs = dict(kwargs)
            kwargs['multi_scale_features'] = model.understanding.geometry_fusion(kwargs['multi_scale_features'], memory)
            return args, kwargs
        model.understanding.mask2former.transformer_module.register_forward_pre_hook(fuse_before_query, with_kwargs=True)
    fixed_eval(model)


def fixed_eval(model):
    if model.understanding_probe_arm != 'joint_full':
        for module in (model.frozen_vggt, model.vggt_memory_adapter, model.anchor_decoder,
                       model.activation_head, model.understanding.encoder,
                       model.understanding.adapter, model.understanding.mask2former.pixel_decoder):
            module.eval()
