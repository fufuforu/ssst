"""R3D model variant; the inherited decoder and reconstruction path are unchanged."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tokengs.models.object_locus_panoptic_v1 import LocusGSObjectLocusPanopticV1Recon
from tokengs.models.object_locus_v3_set import alpha_normalize_membership
from tokengs.models.object_locus_panoptic_v1_lift import lift_features
from tokengs.models.object_locus_output_refine_v1 import initialize_on_cpu


class LocusGSObjectLocusOutputRefineV1Recon(LocusGSObjectLocusPanopticV1Recon):
    architecture_name = 'LOCUSGS_OBJECT_LOCUS_OUTPUT_REFINE_V1'

    def _readout(self, final, gaussians, fm, read_context_decoder, render_decoder_input):
        b = gaussians.shape[0]
        feature_grid = F.interpolate(fm.flatten(0, 1), size=(256, 256), mode='bilinear',
                                     align_corners=False).reshape(b, 2, 256, 256, 256)
        evidence, gate, mass = lift_features(feature_grid, gaussians, read_context_decoder, self.gs)
        child, residual = self.panoptic.gaussian_child_features(
            final['anchor_embedding'], final['f_anchor'], gaussians, final['mu'], final['radii'])
        features = gate * evidence + (1 - gate) * child + 0.1 * torch.tanh(self.panoptic.W_res(child))
        q_base = final['q']
        xyz = gaussians[..., :3]
        q_refined = self.panoptic.output_3d_refine(q_base, features, xyz, final['c'], final['s'])
        mq = self.understanding.mask_embedder(q_refined)
        logits = features @ mq.transpose(1, 2)
        membership = logits.sigmoid()
        rendered = self.gs.render_feature_channels(gaussians, membership, render_decoder_input.cam_view,
                                                   render_decoder_input.intrinsics)
        alpha = rendered['alphas_pred']
        region = alpha_normalize_membership(rendered['images_pred'], alpha)
        cls = self.panoptic.classify(q_refined)
        # Keep the decoder query in final['q']; publish both query versions.
        final['q_refined'] = q_refined
        final.update(cls)
        semantic = region.new_zeros((b, region.shape[1], 20, *region.shape[-2:]))
        semantic[:, :, 0:2] = region[:, :, 100:102]
        semantic[:, :, 2:20] = torch.einsum('bvqhw,bqc->bvchw', region[:, :, :100], cls['p_class'][..., :18])
        semantic = semantic / (semantic.sum(2, keepdim=True) + 1e-6)
        return dict(**cls, q_base=q_base, q_refined=q_refined,
                    assignment=membership, gaussian_membership=membership, gaussian_mask_logits=logits,
                    gaussian_feature=features, child_feature_residual=residual,
                    anchor_membership=final['anchor_membership'], anchor_assignment=final['anchor_membership'],
                    anchor_mask_logits=final['anchor_mask_logits'], membership_mass=rendered['images_pred'],
                    region_mass=region, pixel_membership=region, semantic_scores=semantic,
                    pixel_void_mass=1 - alpha, alpha=alpha, lifting_mass=mass, lifting_gate=gate)


def attach_output_refiner(model):
    if not isinstance(model, LocusGSObjectLocusPanopticV1Recon):
        raise TypeError('source builder did not return the registered parent model')
    model.__class__ = LocusGSObjectLocusOutputRefineV1Recon
    model.panoptic.output_3d_refine = initialize_on_cpu(31416)
    return model
