"""Object-Locus V2 scene states, independent anchor masks and child residuals."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


NUM_THING = 100
NUM_STUFF = 2
NUM_QUERIES = 102
VOID_INDEX = 102
NUM_REGION_CHANNELS = 103
STATE_DIM = 256
ID_DIM = 16
EVIDENCE_HEADS = 8
HEAD_DIM = 32
INIT_NEIGHBORS = 16
GROUP_TEMPERATURE = 0.1


def _stable_seed_indices(mu: torch.Tensor, h: torch.Tensor, origin: torch.Tensor,
                         ell: torch.Tensor, count: int = NUM_THING) -> torch.Tensor:
    """Joint spatial/feature farthest selection, evaluated deterministically on CPU."""
    if mu.ndim != 2 or h.ndim != 2 or mu.shape[0] != h.shape[0]:
        raise ValueError("mu/h must be [N,3]/[N,C] with the same N")
    n = mu.shape[0]
    if n < count:
        raise ValueError(f"need at least {count} candidate anchors, got {n}")
    m = mu.detach().to(device="cpu", dtype=torch.float64)
    f = h.detach().to(device="cpu", dtype=torch.float64)
    o = origin.detach().to(device="cpu", dtype=torch.float64).reshape(1, 3)
    e = ell.detach().to(device="cpu", dtype=torch.float64).reshape(()).clamp_min(0.05)
    if not torch.isfinite(m).all() or not torch.isfinite(f).all() or not torch.isfinite(o).all() or not torch.isfinite(e):
        raise RuntimeError("nonfinite input to deterministic object seed selection")
    f = F.layer_norm(f, (f.shape[-1],), eps=1e-5)
    f = F.normalize(f, dim=-1, eps=1e-6)
    x = (m - o) / e
    spatial = torch.cdist(x, x).square().clamp(max=4.0) / 4.0
    feature = ((1.0 - f @ f.T) / 2.0).clamp(0.0, 1.0)
    joint = 0.5 * spatial + 0.5 * feature
    first_d = x.square().sum(-1)
    first = int(torch.argmin(first_d).item())
    chosen = [first]
    minimum = joint[:, first].clone()
    minimum[first] = -torch.inf
    while len(chosen) < count:
        nxt = int(torch.argmax(minimum).item())  # first maximum is lowest anchor index
        chosen.append(nxt)
        minimum = torch.minimum(minimum, joint[:, nxt])
        minimum[torch.tensor(chosen, dtype=torch.long)] = -torch.inf
    return torch.tensor(chosen, dtype=torch.long)


def _stable_neighborhoods(mu: torch.Tensor, ell: torch.Tensor, seed_indices: torch.Tensor,
                          k: int = INIT_NEIGHBORS):
    """CPU float64 nearest spatial anchors; ties keep original anchor order."""
    n = int(mu.shape[0])
    if n == 0:
        raise ValueError("cannot initialize from zero anchors")
    use_k = min(k, n)
    m = mu.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(m).all():
        raise RuntimeError("nonfinite geometry in seed neighborhood selection")
    seeds = seed_indices.detach().to(device="cpu", dtype=torch.long)
    all_neighbors, sigmas = [], []
    for seed in seeds.tolist():
        d2 = (m - m[seed]).square().sum(-1)
        sorted_order = torch.argsort(d2, stable=True)
        others = sorted_order[sorted_order != seed][:use_k - 1]
        order = torch.cat((torch.tensor([seed], dtype=torch.long), others))
        all_neighbors.append(order)
        sigma = max(math.sqrt(max(float(d2[order[-1]]), 0.0)), 0.05 * float(ell))
        sigmas.append(sigma)
    return torch.stack(all_neighbors), torch.tensor(sigmas, dtype=torch.float64)


def scene_normalization(mu6: torch.Tensor):
    """Detached per-scene origin and scale from layer-6 anchor geometry."""
    if mu6.ndim != 3 or mu6.shape[1:] != (1024, 3):
        raise ValueError(f"layer-6 mu must be [B,1024,3], got {tuple(mu6.shape)}")
    origin = mu6.mean(1, keepdim=True).detach()
    ell = torch.sqrt((mu6 - origin).square().sum(-1).mean(-1)).clamp_min(0.05).detach()
    return origin, ell


class ObjectLocusV2Controller(nn.Module):
    """Shared scene states, evidence reading and independent mask heads."""

    def __init__(self, token_dim: int = 1024):
        super().__init__()
        d = STATE_DIM
        self.token_dim = int(token_dim)

        self.ln_h = nn.LayerNorm(token_dim, eps=1e-5)
        self.proj_h = nn.Linear(token_dim, d)
        self.proj_m = nn.Linear(4, d)
        self.ln_x = nn.LayerNorm(d, eps=1e-5)

        self.init_proj = nn.Linear(d, d)
        self.pose_proj = nn.Linear(6, d)
        self.ln_init = nn.LayerNorm(d, eps=1e-5)
        self.stuff_proj = nn.Linear(d, d)
        self.stuff_ln = nn.LayerNorm(d, eps=1e-5)
        self.stuff_seed = nn.Parameter(torch.empty(NUM_STUFF, d))
        self.stuff_seed._no_weight_decay = True

        self.ln_ev_q = nn.LayerNorm(d, eps=1e-5)
        self.ln_ev_a = nn.LayerNorm(d, eps=1e-5)
        self.W_Q = nn.Linear(d, d, bias=True)
        self.W_K = nn.Linear(d, d, bias=True)
        self.W_V = nn.Linear(d, d, bias=True)
        self.W_O = nn.Linear(d, d, bias=True)

        self.ln_cross = nn.LayerNorm(d, eps=1e-5)
        self.self_attn = nn.MultiheadAttention(d, EVIDENCE_HEADS, dropout=0.0,
                                               bias=True, batch_first=True)
        self.ln_self = nn.LayerNorm(d, eps=1e-5)
        self.ffn_fc1 = nn.Linear(d, 2 * d)
        self.ffn_fc2 = nn.Linear(2 * d, d)
        self.ln_ffn = nn.LayerNorm(d, eps=1e-5)

        self.ln_geom = nn.LayerNorm(d, eps=1e-5)
        self.W_c = nn.Linear(d, 3)
        self.W_s = nn.Linear(d, 3)

        self.ln_own_a = nn.LayerNorm(d, eps=1e-5)
        self.W_own_e = nn.Linear(d, ID_DIM)

        self.ln_mask_a = nn.LayerNorm(d, eps=1e-5)
        self.W_mask_a = nn.Linear(d, d)
        self.ln_mask_q = nn.LayerNorm(d, eps=1e-5)
        self.mask_q_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.mask_bias = nn.Linear(d, 1)
        self.child_index_embedding = nn.Embedding(64, 16)
        self.child_index_embedding.weight._no_weight_decay = True
        self.child_mlp = nn.Sequential(nn.Linear(d + 14 + 16, d), nn.GELU(), nn.Linear(d, d))

        self.ln_cls = nn.LayerNorm(d, eps=1e-5)
        self.thing_classifier = nn.Linear(d, 19)

        self.ln_de = nn.LayerNorm(token_dim, eps=1e-5)
        self.W_de = nn.Linear(token_dim, 64 * ID_DIM)
        self.W_off = nn.Linear(3, ID_DIM)

        self._initialize()

    def _diagnostic_observe(self, stage: str, **values):
        """Opt-in, read-only observation hook used only by failure diagnostics."""
        callback = getattr(self, "_failure_diagnostic_callback", None)
        if callback is not None:
            callback(stage, values)

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Make PyTorch MHA initialization explicit and version independent.
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        if self.self_attn.in_proj_bias is not None:
            nn.init.zeros_(self.self_attn.in_proj_bias)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        nn.init.zeros_(self.self_attn.out_proj.bias)
        nn.init.normal_(self.stuff_seed, std=0.02)
        for head in (self.W_c, self.W_s, self.W_de):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.mask_bias.weight)
        nn.init.zeros_(self.mask_bias.bias)
        nn.init.normal_(self.child_index_embedding.weight, std=0.02)
        nn.init.zeros_(self.child_mlp[-1].weight)
        nn.init.zeros_(self.child_mlp[-1].bias)

    def encode_token(self, tokens, mu, radii, ell):
        geom = torch.cat((mu / ell.reshape(-1, 1, 1),
                          torch.log((radii / ell[:, None]).clamp(1e-4, 1e4)).unsqueeze(-1)), -1)
        return self.ln_x(self.proj_h(self.ln_h(tokens)) + self.proj_m(geom))

    def initialize_states(self, a6, h6, mu6, origin, ell):
        batch, anchors = mu6.shape[:2]
        if anchors != 1024:
            raise RuntimeError(f"Object-Locus V1 requires all 1024 anchors, got {anchors}")
        seed_rows, neighborhood_rows, weight_rows = [], [], []
        c_rows, s_rows, q_rows = [], [], []
        for b in range(batch):
            seed = _stable_seed_indices(mu6[b], h6[b], origin[b], ell[b])
            neigh_cpu, sigma_cpu = _stable_neighborhoods(mu6[b], ell[b].detach().cpu(), seed)
            neigh = neigh_cpu.to(mu6.device)
            seed_device = seed.to(mu6.device)
            d2 = (mu6[b, neigh] - mu6[b, seed_device, None, :]).square().sum(-1)
            sigma = sigma_cpu.to(device=mu6.device, dtype=mu6.dtype).clamp_min(1e-12)
            weights = torch.exp(-d2.detach() / (2.0 * sigma[:, None].square()))
            denom = weights.sum(-1, keepdim=True)
            fallback = torch.zeros_like(weights)
            fallback[:, 0] = 1.0
            weights = torch.where(denom > 1e-12, weights / (denom + 1e-12), fallback)
            pooled = (a6[b, neigh] * weights.unsqueeze(-1)).sum(1)
            c0 = (mu6[b, neigh] * weights.unsqueeze(-1)).sum(1)
            diff = mu6[b, neigh] - c0[:, None, :]
            s0 = torch.sqrt((weights.unsqueeze(-1) * diff.square()).sum(1)
                            + (0.05 * ell[b]).square())
            s0 = torch.clamp(s0, 0.05 * ell[b], 2.0 * ell[b])
            pose = torch.cat(((c0 - origin[b]) / ell[b], torch.log(s0 / ell[b])), -1)
            qt = self.ln_init(self.init_proj(pooled) + self.pose_proj(pose)) if hasattr(self, "ln_init") else self.ln_init(self.init_proj(pooled) + self.pose_proj(pose))
            # Keep exact spec names for the init normalizers.
            qs = self.stuff_ln(self.stuff_proj(a6[b].mean(0)[None, :]) + self.stuff_seed)
            seed_rows.append(seed_device)
            neighborhood_rows.append(neigh)
            weight_rows.append(weights)
            c_rows.append(c0)
            s_rows.append(s0)
            q_rows.append(torch.cat((qt, qs), 0))
        return (torch.stack(q_rows), torch.stack(c_rows), torch.stack(s_rows),
                torch.stack(seed_rows), torch.stack(neighborhood_rows), torch.stack(weight_rows))

    @staticmethod
    def _geometry_bias(mu, c, s):
        # [B,T,3] and [B,Q,3] -> [B,Q,T]
        delta = (mu[:, :, None, :] - c[:, None, :, :]) / (s[:, None, :, :] + 1e-6)
        return (-0.5 * delta.square().sum(-1)).clamp(-20.0, 0.0).transpose(1, 2)

    def read_evidence(self, a, mu, q, c, s):
        b, t, _ = a.shape
        if t != 1024 or q.shape[1] != NUM_QUERIES:
            raise RuntimeError(f"expected 1024 anchors/102 queries, got {t}/{q.shape[1]}")
        qh = self.W_Q(self.ln_ev_q(q)).reshape(b, NUM_QUERIES, EVIDENCE_HEADS, HEAD_DIM).transpose(1, 2)
        kh = self.W_K(self.ln_ev_a(a)).reshape(b, t, EVIDENCE_HEADS, HEAD_DIM).transpose(1, 2)
        vh = self.W_V(self.ln_ev_a(a)).reshape(b, t, EVIDENCE_HEADS, HEAD_DIM).transpose(1, 2)
        geo = self._geometry_bias(mu, c, s)
        geo = torch.cat((geo[:, :NUM_THING], geo.new_zeros((b, NUM_STUFF, t))), dim=1)
        logits = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(HEAD_DIM) + geo[:, None]
        self._diagnostic_observe("evidence", logits=logits, R_inputs=(qh, kh, vh, geo))
        if not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite evidence logits")
        R = torch.softmax(logits, dim=-1)
        z = torch.matmul(R, vh).transpose(1, 2).contiguous().reshape(b, NUM_QUERIES, STATE_DIM)
        z = self.W_O(z)
        return R, R.mean(1), z

    def decode_queries(self, q, z):
        q1 = self.ln_cross(q + z)
        attended = self.self_attn(q1, q1, q1, need_weights=False)[0]
        q2 = self.ln_self(q1 + attended)
        q_new = self.ln_ffn(q2 + self.ffn_fc2(F.gelu(self.ffn_fc1(q2))))
        return q_new

    def update_geometry(self, R_bar, mu, q_new_thing, c, s, ell):
        weights = R_bar[:, :NUM_THING]
        c_ev = torch.einsum("bqt,btd->bqd", weights, mu)
        diff = mu[:, :, None, :] - c_ev[:, None, :, :]
        var = torch.einsum("bqt,btqd->bqd", weights, diff.square())
        s_ev = torch.sqrt(var + (0.05 * ell).reshape(-1, 1, 1).square())
        s_ev = torch.clamp(s_ev, 0.05 * ell.reshape(-1, 1, 1), 2.0 * ell.reshape(-1, 1, 1))

        residual_c = 0.25 * (c_ev - c) + 0.05 * ell[:, None, None] * torch.tanh(self.W_c(self.ln_geom(q_new_thing)))
        norm = torch.linalg.vector_norm(residual_c, dim=-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(norm), (0.25 * ell[:, None, None]) / (norm + 1e-6))
        c_new = c + scale * residual_c
        v = torch.log(s / ell[:, None, None])
        delta_v = 0.25 * torch.log(s_ev / s).clamp(-math.log(2.0), math.log(2.0)) \
            + 0.1 * torch.tanh(self.W_s(self.ln_geom(q_new_thing)))
        v_new = (v + delta_v).clamp(math.log(0.05), math.log(2.0))
        s_new = ell[:, None, None] * torch.exp(v_new)
        self._diagnostic_observe("geometry", c=c, s=s, c_ev=c_ev, s_ev=s_ev,
                                 residual_c=residual_c, residual_norm=norm,
                                 residual_scale=scale, c_new=c_new, v=v,
                                 delta_v=delta_v, v_new=v_new, s_new=s_new)
        return c_new, s_new

    def anchor_masks(self, a, q_new):
        f_anchor = self.ln_mask_a(self.W_mask_a(a))
        q_hidden = self.ln_mask_q(q_new)
        m_query = self.mask_q_mlp(q_hidden)
        b_query = self.mask_bias(q_hidden)
        logits = torch.einsum("btd,bqd->btq", f_anchor, m_query) / math.sqrt(STATE_DIM)
        logits = logits - b_query.transpose(1, 2)
        membership = torch.sigmoid(logits)
        if logits.shape != (a.shape[0], 1024, 102):
            raise RuntimeError(f"bad V2 anchor mask shape {tuple(logits.shape)}")
        return logits, membership, f_anchor, m_query, b_query

    def gaussian_child_features(self, a, f_anchor, gaussians, mu, radii):
        batch, anchors = a.shape[:2]
        if gaussians.ndim != 3 or gaussians.shape[1:] != (anchors * 64, 14):
            raise ValueError(f"expected [B,65536,14] Gaussian tensor, got {tuple(gaussians.shape)}")
        child = gaussians.reshape(batch, anchors, 64, 14)
        radius = radii.reshape(batch, anchors, 1, 1)
        delta = ((child[..., 0:3] - mu[:, :, None, :]) / (radius + 1e-6)).clamp(-5.0, 5.0)
        log_scale = torch.log(child[..., 4:7].clamp_min(1e-8) /
                              (radii[:, :, None, None] + 1e-6)).clamp(-5.0, 5.0)
        geometry = torch.cat((delta, log_scale, child[..., 7:11], child[..., 3:4],
                              child[..., 11:14]), dim=-1)
        ids = torch.arange(64, device=gaussians.device)
        emb = self.child_index_embedding(ids).reshape(1, 1, 64, 16).expand(batch, anchors, -1, -1)
        parent = a[:, :, None, :].expand(-1, -1, 64, -1)
        residual = self.child_mlp(torch.cat((parent, geometry, emb), dim=-1))
        child_feature = f_anchor[:, :, None, :] + residual
        return child_feature.reshape(batch, anchors * 64, STATE_DIM), residual

    @staticmethod
    def gaussian_membership(child_features, m_query, b_query):
        logits = torch.einsum("bnd,bqd->bnq", child_features, m_query) / math.sqrt(STATE_DIM)
        logits = logits - b_query.transpose(1, 2)
        return logits, torch.sigmoid(logits)

    def classify(self, q_new):
        logits19 = self.thing_classifier(self.ln_cls(q_new[:, :NUM_THING]))
        pad = torch.full_like(logits19[..., :1], -1e4)
        return logits19, torch.cat((pad, pad, logits19), dim=-1)

    def forward_registered_layer(self, tokens, mu, radii, ell, q, c, s, anchor_embedding=None):
        a = self.encode_token(tokens, mu, radii, ell) if anchor_embedding is None else anchor_embedding
        self._diagnostic_observe("registered_layer_input", tokens=tokens, mu=mu,
                                 radii=radii, ell=ell, q=q, c=c, s=s,
                                 anchor_embedding=a)
        evidence, evidence_mean, z = self.read_evidence(a, mu, q, c, s)
        q_new = self.decode_queries(q, z)
        c_new, s_new = self.update_geometry(evidence_mean, mu, q_new[:, :NUM_THING], c, s, ell)
        mask_logits, A, f_anchor, m_query, b_query = self.anchor_masks(a, q_new)
        logits19, logits21 = self.classify(q_new)
        self._diagnostic_observe("registered_layer_output", q_new=q_new,
                                 evidence_mean=evidence_mean, anchor_membership=A,
                                 anchor_mask_logits=mask_logits,
                                 thing_logits19=logits19, thing_logits21=logits21,
                                 c_new=c_new, s_new=s_new)
        return {"q": q_new, "c": c_new, "s": s_new, "anchor_embedding": a,
                "evidence_attention": evidence, "evidence_attention_mean": evidence_mean,
                "anchor_assignment": A, "anchor_membership": A,
                "anchor_mask_logits": mask_logits,
                "f_anchor": f_anchor, "m_query": m_query, "mask_bias": b_query,
                "thing_logits19": logits19,
                "thing_class_logits": logits21, "ell": ell,
                "ownership_e": F.normalize(self.W_own_e(self.ln_own_a(a)), dim=-1, eps=1e-6),
                "c_displacement": c_new-c, "seed_indices": None}

    def identity_features(self, final, gaussians):
        batch, anchors = final["tokens"].shape[:2]
        children = gaussians.shape[1] // anchors
        parent = F.normalize(self.W_own_e(self.ln_own_a(final["anchor_embedding"])), dim=-1, eps=1e-6)
        de = self.W_de(self.ln_de(final["tokens"])).reshape(batch, anchors, children, ID_DIM)
        xyz = gaussians[..., :3].reshape(batch, anchors, children, 3)
        off = ((xyz - final["mu"][:, :, None, :]) / 0.15).clamp(-2.0, 2.0)
        child = F.normalize(parent[:, :, None, :] + 0.25 * torch.tanh(de)
                            + 0.1 * torch.tanh(self.W_off(off)), dim=-1, eps=1e-6)
        return child.reshape(batch, anchors * children, ID_DIM)
