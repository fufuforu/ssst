"""Mask correspondence feedback route for the paired Object-Locus experiment."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tokengs.models.object_locus_panoptic_v1_controller import (
    RegisteredObjectLayer,
    capped_injection,
    geometry_bias,
)


class MaskGuidedRegisteredObjectLayer(RegisteredObjectLayer):
    """Original registered layer with only its feedback route replaced."""

    def forward(self, h, a, mu, q, c, s, ell, image, mask_embedder, f_anchor, exposure):
        ev, evidence_route = self.anchor_attention(q, a, geometry_bias(mu, c, s))
        q1 = self.norms[0](q + ev)
        im, _ = self.image_attention(q1, self.image_ln(image))
        q2 = self.norms[1](q1 + im)
        q3 = self.norms[2](q2 + self.self_attn(q2, q2, q2, need_weights=False)[0])
        q_new = self.norms[3](q3 + self.ffn(q3))
        c_new, s_new = self.update_geometry(evidence_route.mean(1), mu, q_new[:, :100], c, s, ell)

        m_query = mask_embedder(q_new)
        z = f_anchor @ m_query.transpose(1, 2)
        z_void = z.new_zeros((*z.shape[:2], 1))
        route = torch.cat((z, z_void), dim=-1).softmax(dim=-1)
        message = route[..., :102] @ F.layer_norm(q_new, (256,), eps=1e-5)
        delta_h = capped_injection(h, message, self.W_inject, exposure)
        return dict(
            q=q_new, c=c_new, s=s_new, route=route, joint_delta=delta_h,
            joint_h_norm=h.norm(dim=-1), evidence_attention=evidence_route,
            anchor_embedding=a, f_anchor=f_anchor, m_query=m_query,
            anchor_mask_logits=z, anchor_membership=torch.sigmoid(z), ell=ell,
        )
