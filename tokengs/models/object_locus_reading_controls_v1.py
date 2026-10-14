"""Isolated, zero-start interventions on context-only C decoder reading."""
import torch
from torch import nn
import torch.nn.functional as F

ARMS = ('baseline', 'object_local_read', 'anchor_visual', 'reliable_depth')


def support_bias(support):
    """Previous predicted anchor masks; stuff/empty/flat masks retain global reads."""
    support = support.detach()[..., :100].transpose(1, 2).clamp(0, 1)
    peak = support.amax(-1, keepdim=True)
    normalized = support / peak.clamp_min(1e-6)
    bias = (.1 + .9 * normalized).log()
    valid = (peak > 1e-6) & ((support.amax(-1, keepdim=True) - support.amin(-1, keepdim=True)) > 1e-6)
    bias = torch.where(valid, bias, torch.zeros_like(bias))
    return torch.cat((bias, bias.new_zeros((bias.shape[0], 2, bias.shape[-1]))), 1)


def local_visual(mu, fm, c2w, intrinsics):
    """Two context projections; bilinear evidence, positive-z and image bounds."""
    w2c = torch.linalg.inv(c2w.detach())
    camera = torch.einsum('bvij,bnj->bvni', w2c[..., :3, :3], mu) + w2c[..., None, :3, 3]
    xyh = torch.einsum('bvij,bvnj->bvni', intrinsics.detach(), camera)
    xy = xyh[..., :2] / xyh[..., 2:].clamp_min(1e-6)
    valid = (camera[..., 2] > 1e-6) & (xy[..., 0] >= 0) & (xy[..., 0] < 256) & (xy[..., 1] >= 0) & (xy[..., 1] < 256)
    grid = (xy / 128 - 1).reshape(-1, mu.shape[1], 1, 2)
    sampled = F.grid_sample(fm.flatten(0, 1), grid, align_corners=False, padding_mode='zeros')
    sampled = sampled.squeeze(-1).transpose(1, 2).reshape(mu.shape[0], 2, mu.shape[1], 256)
    mass = valid.sum(1).unsqueeze(-1)
    value = (sampled * valid.unsqueeze(-1)).sum(1) / mass.clamp_min(1)
    return value, mass > 0


def depth_bias(mu, radii, points, directions, reliability, floor):
    # Along-ray distance complements, rather than replaces, the original
    # perpendicular distance to an infinite ray. Reliability is not probability.
    directions = F.normalize(directions, dim=-1, eps=1e-6)
    along = ((mu[:, :, None] - points[:, None]) * directions[:, None]).sum(-1)
    bandwidth = radii[:, :, None].clamp_min(floor)
    return (-.5 * (along / bandwidth).square()).clamp(-20, 0) * reliability[:, None]


class ReadingControl(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        # No new parameters/capacity. A matched 20-update onset avoids changing
        # the source state abruptly; the strength is locked, not validation-fit.
        self.completed_updates = 0
        self.context = None
        self.observations = {}

    def strength(self, index):
        peaks = (.5, .1, .1)
        active = {'baseline': -1, 'object_local_read': 0, 'anchor_visual': 1, 'reliable_depth': 2}[self.arm]
        return peaks[index] * min(self.completed_updates / 20, 1) if active == index else 0.

    def prepare(self, fm, c2w, intrinsics, points, confidence):
        # Context tensors only. These are passed by generate before any target
        # calibration; not persisted in state_dict or held in diagnostics.
        self.context = (fm, c2w.detach(), intrinsics.detach())
        # Same14x14 pooling and view/row/column order as patch Pluecker rays.
        pooled = F.avg_pool2d(points.permute(0, 1, 4, 2, 3).flatten(0, 1), 14, 14)
        self.points = pooled.reshape(points.shape[0], 2, 3, 1369).permute(0, 1, 3, 2).reshape(points.shape[0], -1, 3).detach()
        conf = F.avg_pool2d(confidence.flatten(0, 1), 14, 14).reshape(points.shape[0], 2, 1369).detach()
        lo, hi = torch.quantile(conf, .2, dim=-1, keepdim=True), torch.quantile(conf, .8, dim=-1, keepdim=True)
        self.reliability = ((conf - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1).flatten(1)
        if not all(torch.isfinite(x).all() for x in (self.points, self.reliability)):
            raise FloatingPointError('nonfinite context geometry control')
        self.observations = {}

    def anchor(self, a, mu):
        if self.arm != 'anchor_visual':
            return a
        value, valid = local_visual(mu, *self.context)
        value = F.layer_norm(value, (256,), eps=1e-5)
        delta = self.strength(1) * .25 * a.norm(dim=-1, keepdim=True).detach() * F.normalize(value, dim=-1, eps=1e-6) * valid
        self.observations['visual_valid_fraction'] = float(valid.float().mean().detach())
        return a + delta

    def object_bias(self, bias, previous):
        if self.arm != 'object_local_read' or previous is None:
            return bias
        return bias + self.strength(0) * support_bias(previous)

    def geometry(self, bias, mu, radii, directions, floor):
        if self.arm != 'reliable_depth':
            return bias
        extra = depth_bias(mu, radii, self.points, directions, self.reliability, floor)
        return (torch.zeros_like(extra) if bias is None else bias) + self.strength(2) * extra

    def describe(self):
        return dict(arm=self.arm, strengths=[float(self.strength(i)) for i in range(3)], **self.observations)


def install(model, arm):
    if hasattr(model.panoptic, 'reading_control'):
        raise ValueError('control already installed')
    model.panoptic.reading_control = ReadingControl(arm).to(next(model.parameters()).device)
    # Observe actual head values and projected output, not mean(R) @ raw a.
    for name, layer in model.panoptic.layers.items():
        def observe(module, inputs, output, key=name):
            ev, weights = output
            ctrl = model.panoptic.reading_control
            ctrl.observations[key] = dict(output_norm_mean=float(ev.detach().norm(dim=-1).mean()),
                per_head_entropy_mean=float((-(weights.detach().clamp_min(1e-12).log() * weights.detach()).sum(-1) / __import__('math').log(weights.shape[-1])).mean()))
        layer.anchor_attention.register_forward_hook(observe)
        def observe_values(module, inputs, output, key=name):
            value = output.detach().reshape(output.shape[0], output.shape[1], 8, 32)
            model.panoptic.reading_control.observations[key + '_V_head_norm'] = value.norm(dim=-1).mean((0, 1)).cpu().tolist()
        layer.anchor_attention.V.register_forward_hook(observe_values)
