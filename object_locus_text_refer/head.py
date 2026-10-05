from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TextObjectBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(256, 8, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(256)
        self.ffn = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Dropout(0.0), nn.Linear(1024, 256))
        self.norm2 = nn.LayerNorm(256)

    def forward(self, text, objects):
        x = self.norm1(text + self.attn(text, objects, objects, need_weights=False)[0])
        return self.norm2(x + self.ffn(x))


class ObjectLocusTextReferHead(nn.Module):
    """Fixed 2-block CLIP text to 100 thing slots plus null decoder."""
    def __init__(self, seed: int = 31415):
        super().__init__()
        # Constructors also consume RNG, so keep construction and final init local.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.text_projection = nn.Linear(512, 256)
            self.text_norm = nn.LayerNorm(256)
            self.blocks = nn.ModuleList([TextObjectBlock(), TextObjectBlock()])
            self.text_score = nn.Linear(256, 256)
            self.object_score = nn.Linear(256, 256)
            self.null_score = nn.Linear(256, 1)
            self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                nn.init.xavier_uniform_(module.out_proj.weight)
                nn.init.zeros_(module.out_proj.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, text_features, input_ids, attention_mask, object_features, eot_token_id=49407):
        if text_features.ndim != 3 or tuple(text_features.shape[1:]) != (77, 512):
            raise ValueError("text_features must be [B,77,512]")
        if tuple(input_ids.shape[1:]) != (77,) or tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError("input_ids and attention_mask must be [B,77]")
        if tuple(object_features.shape[1:]) != (100, 256):
            raise ValueError("object_features must be [B,100,256]")
        x = self.text_norm(self.text_projection(text_features))
        q = object_features.detach()
        for block in self.blocks:
            x = block(x, q)
        # CLIP's EOT token has the largest token ID in each unpadded sequence.
        # Locate it within attention_mask; do not read a padding position.
        eot = input_ids.eq(eot_token_id) & attention_mask.bool()
        if not eot.any(dim=1).all():
            # Supports tokenizer instances whose EOT ID differs from the default.
            ids = input_ids.masked_fill(~attention_mask.bool(), -1)
            eot_pos = ids.argmax(dim=1)
        else:
            eot_pos = eot.to(torch.int64).argmax(dim=1)
        t = x[torch.arange(x.shape[0], device=x.device), eot_pos]
        u = F.normalize(self.text_score(t), dim=-1)
        v = F.normalize(self.object_score(q), dim=-1)
        thing = torch.einsum("bd,bjd->bj", u, v) / 0.07
        scores = torch.cat([thing, self.null_score(t)], dim=-1)
        return {"scores": scores, "pi": scores.softmax(dim=-1), "text_state": t}


def build_head_optimizer(head, lr=1e-4, weight_decay=.05):
    decay=[]; nodecay=[]
    for name,parameter in head.named_parameters():
        (decay if parameter.ndim == 2 and not name.endswith('.bias') else nodecay).append(parameter)
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':weight_decay},
                                 {'params':nodecay,'weight_decay':0.0}],
                                lr=lr,betas=(.9,.95),eps=1e-8)
    if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in head.parameters()}:
        raise RuntimeError('optimizer must contain exactly the new text refer head')
    return optimizer


def soft_gaussian_membership(pi, gaussian_membership):
    if gaussian_membership.ndim != 3 or tuple(gaussian_membership.shape[1:]) != (65536, 100):
        raise ValueError("gaussian_membership must be [B,65536,100]")
    if tuple(pi.shape) != (gaussian_membership.shape[0], 101):
        raise ValueError("pi must be [B,101]")
    return torch.einsum("bj,bgj->bg", pi[:, :100], gaussian_membership.detach())


def hard_gaussian_membership(scores, gaussian_membership):
    slot = scores.argmax(dim=-1)
    batch = torch.arange(scores.shape[0], device=scores.device)
    selected = gaussian_membership.detach()[batch, :, slot.clamp_max(99)]
    return torch.where(slot[:, None] == 100, torch.zeros_like(selected), selected), slot
