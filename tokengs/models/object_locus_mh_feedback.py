"""Task-local eight-head object-to-anchor feedback for the MH experiment."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn

from tokengs.models.object_locus_panoptic_v1_controller import (
    RegisteredObjectLayer,
    capped_injection,
    geometry_bias,
)


class MultiHeadFeedbackRegisteredObjectLayer(RegisteredObjectLayer):
    """Keep the registered object update intact and replace only feedback pooling."""

    def forward(self, h, a, mu, q, c, s, ell, image, mask_embedder, f_anchor, exposure):
        ev, R = self.anchor_attention(q, a, geometry_bias(mu, c, s))
        q1 = self.norms[0](q + ev)
        im, _ = self.image_attention(q1, self.image_ln(image))
        q2 = self.norms[1](q1 + im)
        q3 = self.norms[2](q2 + self.self_attn(q2, q2, q2, need_weights=False)[0])
        qnew = self.norms[3](q3 + self.ffn(q3))
        cnew, snew = self.update_geometry(R.mean(1), mu, qnew[:, :100], c, s, ell)

        mq = mask_embedder(qnew)
        mask_logits = f_anchor @ mq.transpose(1, 2)
        mask_membership = torch.sigmoid(mask_logits)

        b = a.shape[0]
        a_ln = F.layer_norm(a, (256,), eps=1e-5)
        q_ln = F.layer_norm(qnew, (256,), eps=1e-5)
        Q = self.feedback_q(a_ln).reshape(b, 1024, 8, 32).transpose(1, 2)
        K = self.feedback_k(q_ln).reshape(b, 102, 8, 32).transpose(1, 2)
        V = self.feedback_v(q_ln).reshape(b, 102, 8, 32).transpose(1, 2)

        G = geometry_bias(mu, cnew, snew).transpose(1, 2)
        S = Q @ K.transpose(-1, -2) / math.sqrt(32) + G[:, None]
        void = S.new_zeros((*S.shape[:-1], 1))
        feedback_attention = torch.cat((S, void), dim=-1).softmax(dim=-1)
        U = feedback_attention[..., :102] @ V
        U_merged = U.transpose(1, 2).reshape(b, 1024, 256)
        message = self.feedback_o(U_merged)

        delta = capped_injection(h, message, self.W_inject, exposure)
        route = feedback_attention.mean(dim=1)
        return dict(
            q=qnew, c=cnew, s=snew, route=route,
            feedback_attention=feedback_attention, joint_delta=delta,
            joint_h_norm=h.norm(dim=-1), evidence_attention=R,
            anchor_embedding=a, f_anchor=f_anchor, m_query=mq,
            anchor_mask_logits=mask_logits, anchor_membership=mask_membership,
            ell=ell,
        )


def initialize_feedback_layers(model, *, seed=31416):
    """Install four independent projections without changing caller RNG state."""
    layers = []
    for name in ("L6", "L8", "L10", "L12"):
        layer = model.panoptic.layers[name]
        if type(layer) is not RegisteredObjectLayer:
            raise RuntimeError(f"{name} must be the original RegisteredObjectLayer")
        layers.append((name, layer))

    # fork_rng restores CPU RNG and leaves Python, NumPy and CUDA RNG streams untouched.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for name, layer in layers:
            layer.__class__ = MultiHeadFeedbackRegisteredObjectLayer
            device = layer.W_inject.weight.device
            dtype = layer.W_inject.weight.dtype
            for projection_name in ("feedback_q", "feedback_k", "feedback_v", "feedback_o"):
                projection = nn.Linear(256, 256, bias=False, device="cpu", dtype=dtype)
                if projection_name == "feedback_o":
                    nn.init.eye_(projection.weight)
                else:
                    nn.init.xavier_uniform_(projection.weight, gain=1.0)
                setattr(layer, projection_name, projection.to(device=device))
    return model
