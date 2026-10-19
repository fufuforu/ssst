"""A dual stream: original 1024 point anchors, unchanged superpoint understanding.

Reconstruction seeding follows LocusGSGFMDualStreamV2.generate (also used by
V3/V4). The dynamic A controller, decoder, normalization and readout are reused.
Reconstruction anchors have no one-to-one ownership of understanding queries.
"""
import torch
from torch import nn

from tokengs.options import config_defaults
from tokengs.models.dual_stream_initialization import seed_anchors
from tokengs.models.gfm_dual_stream_components import gfm_grid, sample_anchor_gfm
from tokengs.models.superpoint_dual_core import SuperpointDualCore


class SuperpointDualPointReconstructionCore(SuperpointDualCore):
    reconstruction_count = 1024

    def __init__(self, opt, config):
        super().__init__(opt, config)
        # Remove the replaced trainable seed path, rather than leave unused
        # reconstruction point-encoder/projection parameters in the optimizer.
        del self.initializer.reconstruction
        del self.reconstruction_seed_projection
        legacy = config_defaults['train_siu3r_object_locus_gfm_dual_stream_v4']
        self.gs_tokens = nn.Parameter(legacy.gs_token_std * torch.randn(1024, 1024))
        self.anchor_content = nn.Sequential(nn.Linear(7, 1024), nn.GELU(), nn.Linear(1024, 1024))
        self.anchor_visual_projection = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 1024))

    def reconstruction_seed(self, omega, rgb):
        mu, radius, color, source_ids = seed_anchors(
            omega['points'], omega['confidence'], rgb, self.reconstruction_count,
            return_source_indices=True)
        origin = mu.mean(1, keepdim=True)
        ell = (mu-origin).square().sum(-1).mean(-1).sqrt().clamp_min(.05)
        content = torch.cat(((mu-origin)/ell[:, None, None],
                             (radius/ell[:, None])[..., None], color), -1)
        visual = sample_anchor_gfm(gfm_grid(omega['patch_layers']), source_ids)
        tokens = (self.gs_tokens[None].expand(len(rgb), -1, -1).clone()
                  + self.anchor_content(content) + self.anchor_visual_projection(visual))
        return dict(tokens=tokens, centers=mu, radii=radius, source_indices=source_ids)

    def prepare_memory(self, omega, rgb, renderer):
        memory = super().prepare_memory(omega, rgb, renderer)
        memory['point_reconstruction_seed'] = self.reconstruction_seed(omega, rgb)
        return memory

    def initialize_streams(self, cloud, memory):
        under = self.initializer.understanding(cloud)
        rec = memory['point_reconstruction_seed']
        return rec['tokens'], rec, under

    def parent_metadata(self, tokens, n):
        # Anchor indices are not semantic-query indices: N_rec=1024, N_under
        # remains dynamic. Membership still comes from A's feature readout.
        return dict(gaussian_parent_anchor=torch.arange(tokens.shape[1], device=tokens.device).repeat_interleave(64),
                    num_reconstruction_tokens=tokens.shape[1])

    def forward(self, cloud, memory, superpoint_ids=None):
        output = super().forward(cloud, memory, superpoint_ids)
        rec = memory['point_reconstruction_seed']
        output.update(initial_anchor_mu=rec['centers'], initial_anchor_radius=rec['radii'],
                      initial_anchor_source_ids=rec['source_indices'])
        return output

    def parameter_groups(self):
        groups = super().parameter_groups()
        extra = {id(p) for p in (self.gs_tokens, *self.anchor_content.parameters(),
                                *self.anchor_visual_projection.parameters())}
        moved = [p for p in groups['understanding'] if id(p) in extra]
        groups['understanding'] = [p for p in groups['understanding'] if id(p) not in extra]
        groups['reconstruction'].extend(moved)
        return groups
