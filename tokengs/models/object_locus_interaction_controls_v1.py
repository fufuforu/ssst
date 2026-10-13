"""Opt-in decoder interaction controls; leave segmentation readout unchanged."""
from __future__ import annotations
import math
import types
import torch
from torch import nn
import torch.nn.functional as F
from .object_locus_panoptic_v1_controller import capped_injection, geometry_bias

ARMS = ('bidirectional', 'forward_only', 'gated_feedback', 'independent_feedback')


class ForwardOnly(nn.Module):
    def forward(self, h, a, q, mu, c, s, message, projection, exposure):
        return torch.zeros_like(h)


class GatedFeedback(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_proj = nn.Linear(1024, 64)
        self.message_proj = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)
        for m in (self.h_proj, self.message_proj):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, math.log(9.0))
        self.last_gate = None

    def forward(self, h, a, q, mu, c, s, message, projection, exposure):
        gate = torch.sigmoid(self.out(F.gelu(self.h_proj(F.layer_norm(h, (1024,))) +
                                            self.message_proj(F.layer_norm(message, (256,))))))
        self.last_gate = gate.detach()
        return gate * capped_injection(h, message, projection, exposure)


class IndependentFeedback(nn.Module):
    """GS queries read object queries in a dedicated eight-head feature space.

    Preserve the original geometric prior and zero key/value null option;
    neither feedback logits nor values use the mask embedder.
    """
    def __init__(self):
        super().__init__()
        for name in ('Q', 'K', 'V', 'O'):
            m = nn.Linear(256, 256)
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            setattr(self, name, m)
        self.last_weights = None

    def forward(self, h, a, q, mu, c, s, message, projection, exposure):
        b, nt = a.shape[:2]; nq = q.shape[1]
        normalized = F.layer_norm(q, (256,), eps=1e-5)
        query = self.Q(a).reshape(b, nt, 8, 32).transpose(1, 2)
        key = self.K(normalized).reshape(b, nq, 8, 32).transpose(1, 2)
        value = self.V(normalized).reshape(b, nq, 8, 32).transpose(1, 2)
        logits = query @ key.transpose(-1, -2) / math.sqrt(32)
        logits = logits + geometry_bias(mu, c, s).transpose(1, 2)[:, None]
        logits = torch.cat((logits, logits.new_zeros((*logits.shape[:-1], 1))), -1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite independent feedback attention')
        weights = logits.softmax(-1)
        message = self.O((weights[..., :nq] @ value).transpose(1, 2).reshape(b, nt, 256))
        self.last_weights = weights.detach()
        return capped_injection(h, message, projection, exposure)


def _observe(layer, *, h, delta, route):
    module = getattr(layer, 'interaction_feedback', None)
    feedback_route = getattr(module, 'last_weights', None)
    if feedback_route is None: feedback_route = route
    ratio = delta.detach().norm(dim=-1) / h.detach().norm(dim=-1).clamp_min(1e-6)
    values = [ratio.mean(), ratio.max(),
              -(feedback_route.detach() * feedback_route.detach().clamp_min(1e-20).log()).sum(-1).mean()]
    gate = getattr(module, 'last_gate', None)
    if gate is not None: values += [gate.mean(), gate.min(), gate.max()]
    numbers = torch.stack(values).cpu().tolist()
    layer.interaction_observation = dict(zip(('delta_ratio_mean', 'delta_ratio_max',
                                            'feedback_route_entropy', 'gate_mean', 'gate_min', 'gate_max'), numbers))
    # Never retain feature graphs or detached GPU arrays between microbatches.
    if gate is not None: module.last_gate = None
    if getattr(module, 'last_weights', None) is not None: module.last_weights = None


def install_interaction(controller, arm):
    if arm not in ARMS: raise ValueError('unknown interaction arm')
    if any(hasattr(layer, 'interaction_mode') for layer in controller.layers.values()):
        raise ValueError('interaction already installed')
    # Initialization must not advance the shared training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(314159)
        for layer in controller.layers.values():
            if arm != 'bidirectional':
                cls = {'forward_only': ForwardOnly, 'gated_feedback': GatedFeedback,
                       'independent_feedback': IndependentFeedback}[arm]
                parameter = next(layer.parameters())
                layer.interaction_feedback = cls().to(device=parameter.device, dtype=parameter.dtype)
            layer.interaction_mode = arm
            layer._interaction_observe = types.MethodType(_observe, layer)


def observations(controller):
    return {name: dict(layer.interaction_observation) for name, layer in controller.layers.items()}
