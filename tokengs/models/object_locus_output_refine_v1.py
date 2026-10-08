"""Output-side 3D Gaussian evidence refinement for Object-Locus V1."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class Output3DEvidenceRefine(nn.Module):
    """One pre-LN cross-attention and FFN over every decoded Gaussian."""

    def __init__(self):
        super().__init__()
        self.ln_q = nn.LayerNorm(256, eps=1e-5)
        self.ln_g = nn.LayerNorm(256, eps=1e-5)
        self.ln_ffn = nn.LayerNorm(256, eps=1e-5)
        self.W_Q = nn.Linear(256, 256, bias=True)
        self.W_K = nn.Linear(256, 256, bias=True)
        self.W_V = nn.Linear(256, 256, bias=True)
        self.W_O = nn.Linear(256, 256, bias=True)
        self.W_1 = nn.Linear(256, 512, bias=True)
        self.W_2 = nn.Linear(512, 256, bias=True)
        self.act = nn.GELU(approximate='none')
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.W_Q, self.W_K, self.W_V, self.W_1):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in (self.W_O, self.W_2):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in (self.ln_q, self.ln_g, self.ln_ffn):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def geometry_bias(xyz, c, s):
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f'xyz must be [B,G,3], got {tuple(xyz.shape)}')
        if c.ndim != 3 or c.shape[1:] != (100, 3):
            raise ValueError(f'c must be [B,100,3], got {tuple(c.shape)}')
        if s.shape != c.shape:
            raise ValueError(f's must match c, got {tuple(s.shape)}')
        if xyz.shape[0] != c.shape[0]:
            raise ValueError('xyz/c batch mismatch')
        x, center, scale = xyz.detach(), c.detach(), s.detach()
        if not torch.isfinite(x).all() or not torch.isfinite(center).all():
            raise FloatingPointError('nonfinite detached geometry')
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('s must be finite and strictly positive')
        thing = -0.5 * (((x[:, None] - center[:, :, None]) / scale[:, :, None]) ** 2).sum(-1)
        return torch.cat((thing, thing.new_zeros((x.shape[0], 2, x.shape[1]))), dim=1)

    @staticmethod
    def split_heads(x):
        return x.reshape(x.shape[0], x.shape[1], 8, 32).transpose(1, 2)

    @staticmethod
    def merge_heads(x):
        return x.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[2], 256)

    def _attend_chunk(self, qh, kh, vh, bias):
        scores = (qh @ kh.transpose(-1, -2)) / math.sqrt(32.0)
        scores = scores + bias[:, None]
        weights = torch.softmax(scores, dim=-1)
        return weights @ vh

    def forward(self, q_base, features, xyz, c, s, *, collect_stats=False):
        if q_base.ndim != 3 or q_base.shape[1:] != (102, 256):
            raise ValueError(f'q_base must be [B,102,256], got {tuple(q_base.shape)}')
        if features.ndim != 3 or features.shape[0] != q_base.shape[0] or features.shape[-1] != 256:
            raise ValueError(f'features must be [B,G,256], got {tuple(features.shape)}')
        if xyz.shape != (*features.shape[:2], 3):
            raise ValueError('xyz must match feature batch/G dimensions')
        bias = self.geometry_bias(xyz, c, s)
        # Explicit FP32 projections/matmul/softmax. This block is intentionally
        # outside AMP/SDPA so the full 65,536-GS denominator is used.
        qh = self.split_heads(self.W_Q(self.ln_q(q_base.float())))
        gnorm = self.ln_g(features.float())
        kh = self.split_heads(self.W_K(gnorm))
        vh = self.split_heads(self.W_V(gnorm))
        outputs = []
        for start in range(0, q_base.shape[1], 8):
            stop = min(start + 8, q_base.shape[1])
            q_part, bias_part = qh[:, :, start:stop], bias[:, start:stop]
            if self.training and torch.is_grad_enabled():
                heads = checkpoint(self._attend_chunk, q_part, kh, vh, bias_part,
                                   use_reentrant=False, preserve_rng_state=True)
            else:
                heads = self._attend_chunk(q_part, kh, vh, bias_part)
            outputs.append(self.merge_heads(heads))
        attended = torch.cat(outputs, dim=1)
        q1 = q_base.float() + self.W_O(attended)
        q_refined = q1 + self.W_2(self.act(self.W_1(self.ln_ffn(q1))))
        if collect_stats:
            delta = q_refined.detach() - q_base.detach().float()
            stats = {'q_delta_rms': float(delta.square().mean().sqrt()),
                     'q_base_rms': float(q_base.detach().float().square().mean().sqrt())}
            stats['q_delta_relative_rms'] = stats['q_delta_rms'] / max(stats['q_base_rms'], 1e-12)
            return q_refined, stats
        return q_refined


def initialize_on_cpu(seed=31416):
    # Keep the caller's CPU RNG stream unchanged.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Output3DEvidenceRefine().cpu()
