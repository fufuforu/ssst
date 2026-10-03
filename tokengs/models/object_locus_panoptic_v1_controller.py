"""Fixed FP32 object evidence, geometry updates and token-only injection."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from tokengs.models.object_locus_v3_set_controller import ObjectLocusV3SetController


def geometry_bias(mu, c, s):
    thing = -0.5 * ((mu[:, None] - c[:, :, None]) / s[:, :, None]).square().sum(-1)
    return torch.cat((thing, thing.new_zeros((mu.shape[0], 2, mu.shape[1]))), 1)


def capped_injection(h, message, projection, exposure):
    sigma = (h.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    raw = sigma * torch.tanh(projection(message) / sigma)
    scale = (0.25 * h.norm(dim=-1, keepdim=True) / (raw.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1)
    return 0.1 * min(max(float(exposure), 0) / 1000, 1) * raw * scale


class EvidenceAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Q = nn.Linear(256, 256)
        self.K = nn.Linear(256, 256)
        self.V = nn.Linear(256, 256)
        self.O = nn.Linear(256, 256)

    def forward(self, q, evidence, bias=None):
        b, nq = q.shape[:2]
        ne = evidence.shape[1]
        qh = self.Q(q).reshape(b, nq, 8, 32).transpose(1, 2)
        kh = self.K(evidence).reshape(b, ne, 8, 32).transpose(1, 2)
        vh = self.V(evidence).reshape(b, ne, 8, 32).transpose(1, 2)
        logits = qh @ kh.transpose(-1, -2) / math.sqrt(32)
        if bias is not None:
            logits = logits + bias[:, None]
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite object evidence logits')
        weights = logits.softmax(-1)
        return self.O((weights @ vh).transpose(1, 2).reshape(b, nq, 256)), weights


class RegisteredObjectLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor_attention = EvidenceAttention()
        self.image_attention = EvidenceAttention()
        self.image_ln = nn.LayerNorm(256, eps=1e-5)
        self.norms = nn.ModuleList([nn.LayerNorm(256, eps=1e-5) for _ in range(4)])
        self.self_attn = nn.MultiheadAttention(256, 8, dropout=0, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))
        self.ln_geom = nn.LayerNorm(256, eps=1e-5)
        self.W_c = nn.Linear(256, 3)
        self.W_s = nn.Linear(256, 3)
        self.W_inject = nn.Linear(256, 1024, bias=False)

    def _diagnostic_observe(self, *args, **kwargs):
        pass

    update_geometry = ObjectLocusV3SetController.update_geometry

    def forward(self, h, a, mu, q, c, s, ell, image, mask_embedder, f_anchor, exposure):
        ev, R = self.anchor_attention(q, a, geometry_bias(mu, c, s))
        q1 = self.norms[0](q + ev)
        im, _ = self.image_attention(q1, self.image_ln(image))
        q2 = self.norms[1](q1 + im)
        q3 = self.norms[2](q2 + self.self_attn(q2, q2, q2, need_weights=False)[0])
        qnew = self.norms[3](q3 + self.ffn(q3))
        cnew, snew = self.update_geometry(R.mean(1), mu, qnew[:, :100], c, s, ell)
        mq = mask_embedder(qnew)
        logits = F.normalize(f_anchor, dim=-1, eps=1e-6) @ F.normalize(mq, dim=-1, eps=1e-6).transpose(1, 2) / 0.1
        logits = logits + geometry_bias(mu, cnew, snew).transpose(1, 2)
        route = torch.cat((logits, logits.new_zeros((*logits.shape[:2], 1))), -1).softmax(-1)
        message = route[..., :102] @ F.layer_norm(qnew, (256,), eps=1e-5)
        delta = capped_injection(h, message, self.W_inject, exposure)
        return dict(q=qnew, c=cnew, s=snew, route=route, joint_delta=delta,
                    joint_h_norm=h.norm(dim=-1), evidence_attention=R,
                    anchor_embedding=a, f_anchor=f_anchor, m_query=mq,
                    anchor_mask_logits=f_anchor @ mq.transpose(1, 2),
                    anchor_membership=torch.sigmoid(f_anchor @ mq.transpose(1, 2)), ell=ell)


class ObjectLocusPanopticV1Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_h = nn.LayerNorm(1024, eps=1e-5)
        self.proj_h = nn.Linear(1024, 256)
        self.proj_m = nn.Linear(4, 256)
        self.ln_x = nn.LayerNorm(256, eps=1e-5)
        self.stuff_seed = nn.Parameter(torch.empty(2, 256))
        self.stuff_ln = nn.LayerNorm(256, eps=1e-5)
        self.W_mask_a = nn.Linear(256, 256)
        self.ln_mask_a = nn.LayerNorm(256, eps=1e-5)
        self.layers = nn.ModuleDict({f'L{i}': RegisteredObjectLayer() for i in (6,8,10,12)})
        self.child_index_embedding = nn.Embedding(64, 16)
        self.child_mlp = nn.Sequential(nn.Linear(286, 256), nn.GELU(), nn.Linear(256, 256))
        self.W_res = nn.Linear(256, 256)
        self.class_ln = nn.LayerNorm(256, eps=1e-5)
        self.class_head = nn.Linear(256, 19)
        self.initialize_new()

    encode_token = ObjectLocusV3SetController.encode_token
    gaussian_child_features = ObjectLocusV3SetController.gaussian_child_features

    def initialize_new(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.normal_(self.stuff_seed, std=0.02)
        nn.init.normal_(self.child_index_embedding.weight, std=0.02)
        for layer in self.layers.values():
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.zeros_(layer.self_attn.in_proj_bias)
            for m in (layer.anchor_attention.O, layer.image_attention.O,
                      layer.self_attn.out_proj, layer.ffn[-1], layer.W_c,
                      layer.W_s, layer.W_inject):
                nn.init.zeros_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        for m in (self.child_mlp[-1], self.W_res):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def initialize_states(self, qpre, fm, a, mu, ell):
        weights = (F.normalize(qpre, dim=-1, eps=1e-6) @ F.normalize(a, dim=-1, eps=1e-6).transpose(1,2) / 0.1).softmax(-1)
        c = weights @ mu
        var = torch.einsum('bqt,bqtd->bqd', weights, (mu[:,None] - c[:,:,None]).square())
        s = (var + (0.05*ell[:,None,None]).square()).sqrt()
        s = s.clamp(min=0.05*ell[:,None,None], max=2*ell[:,None,None])
        stuff = self.stuff_ln(fm.mean((1,3,4))[:,None] + self.stuff_seed[None])
        return torch.cat((qpre,stuff),1), c, s

    def classify(self, q):
        logits = self.class_head(self.class_ln(q[:,:100]))
        p = logits.softmax(-1)
        pad = logits.new_full((*logits.shape[:-1],2), -1e4)
        return dict(thing_logits19=logits, class_logits19=logits, p_class=p,
                    conditional_class_prob=p[...,:18]/p[...,:18].sum(-1,keepdim=True).clamp_min(1e-12),
                    objectness_prob=1-p[...,18], thing_class_logits=torch.cat((pad,logits),-1))
