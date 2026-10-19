"""Point1024 + superpoint queries, one GFM memory and one joint decoder."""
import torch
from torch import nn
import torch.nn.functional as F

from tokengs.models.dual_stream_decoder import PointSeededAnchorDecoder
from tokengs.models.gfm_dual_stream_components import GFMReconstructionAdapter
from tokengs.models.input_types import EncoderLatent, ModelInputDecoder
from tokengs.models.canonical_recon_models import patch_plucker_rays
from tokengs.models.object_locus_posefree_geometry import pixel_rays, camera_vectors
from tokengs.models.object_locus_panoptic_v1_lift import lift_features
from tokengs.models.superpoint_dual_core import superpoint_scene_normalization
from tokengs.models.superpoint_dual_point_reconstruction import SuperpointDualPointReconstructionCore
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.unified_superpoints import lift_context_cloud


class SharedGFMMemoryAdapter(GFMReconstructionAdapter):
    """Expose the SAME projected memory for point pooling and both token roles."""
    def forward(self, layers):
        if len(layers) != 4:
            raise ValueError('Four Omega layers required')
        mixed = sum(w*norm(x.float()) for w, norm, x in
                    zip(self.layer_logits.softmax(0), self.layer_norms, layers))
        b, v, p, _ = mixed.shape
        if (v, p) != (2, 1024):
            raise ValueError('Two native 32x32 GFM grids required')
        features = self.projection(mixed.reshape(b, v*p, 2048))
        keys, values = self.kv(features).chunk(2, -1)
        keys = self.key_norm(keys.reshape(b, v*p, 16, 64).transpose(1, 2).contiguous())
        values = values.reshape(b, v*p, 16, 64).transpose(1, 2).contiguous()
        grid = features.reshape(b, v, 32, 32, 1024).permute(0, 1, 4, 2, 3).contiguous()
        return dict(latent=EncoderLatent(keys=keys, values=values), grid=grid)


class SharedFeatureMaskMemory(nn.Module):
    """A's local RGB/mask readout, with no separate GFM adapter."""
    def __init__(self, previous):
        super().__init__()
        self.local = previous.local
        self.refine = previous.refine
        self.mask_embedder = previous.mask_embedder

    def forward(self, grid, rgb):
        b, v = rgb.shape[:2]
        feature = F.interpolate(grid.flatten(0, 1), size=(128, 128), mode='bilinear', align_corners=False)
        local = self.local(F.interpolate(rgb.flatten(0, 1), size=(128, 128), mode='bilinear', align_corners=False))
        return self.refine(feature+local).reshape(b, v, 256, 128, 128)


class ConcatPointSeededDecoder(PointSeededAnchorDecoder):
    reconstruction_count = 1024

    def update_geometry(self, h, mu, rho, head_mu, head_rho):
        # Only reconstruction anchors move. Superpoint/stuff centers remain
        # fixed positional references while their content updates in all layers.
        n = self.reconstruction_count
        new_mu, new_rho = super().update_geometry(h[:, :n], mu[:, :n], rho[:, :n], head_mu, head_rho)
        return torch.cat((new_mu, mu[:, n:]), 1), torch.cat((new_rho, rho[:, n:]), 1)


class ConcatSharedCore(SuperpointDualPointReconstructionCore):
    anchor_decoder_type = ConcatPointSeededDecoder

    def __init__(self, opt, config):
        super().__init__(opt, config)
        # Same parameter layout for the common GFM adapter; expose its grid too.
        self.reconstruction_memory.__class__ = SharedGFMMemoryAdapter
        self.semantic_memory = SharedFeatureMaskMemory(self.semantic_memory)
        del self.panoptic.layers  # No second, understanding-only decoder.
        self.shared_feature_projection = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256))
        self.understanding_input_projection = nn.Linear(256, 1024)
        self.understanding_output_projection = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256))

    def prepare_memory(self, omega, rgb, renderer):
        shared = self.reconstruction_memory(omega['patch_layers'])
        grid = self.shared_feature_projection(shared['grid'].permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3).contiguous()
        rays_o, rays_d = pixel_rays(omega['c2w'], omega['k'], 256, 256)
        return dict(latent=shared['latent'], point_feature_grid=grid, fm=self.semantic_memory(grid, rgb),
                    rays=patch_plucker_rays(rays_o, rays_d, patch_size=8), renderer=renderer,
                    decoder=ModelInputDecoder(cam_view=torch.linalg.inv(omega['c2w']).transpose(-1, -2),
                                              intrinsics=camera_vectors(omega['k'])),
                    point_reconstruction_seed=self.reconstruction_seed(omega, rgb))

    def forward(self, cloud, memory, superpoint_ids=None):
        if superpoint_ids is not None:
            cloud = dict(cloud, point_superpoint_ids=superpoint_ids)
        under = self.initializer.understanding(cloud)
        rec = memory['point_reconstruction_seed']
        nrec, n = rec['tokens'].shape[1], under['query'].shape[1]
        if nrec != 1024:
            raise ValueError('Exactly 1024 reconstruction anchors required')
        fm = memory['fm']
        stuff = self.panoptic.stuff_ln(fm.mean((1, 3, 4))[:, None]+self.panoptic.stuff_seed[None])
        initial_under = torch.cat((under['query'], stuff), 1)
        joint = torch.cat((rec['tokens'], self.understanding_input_projection(initial_under)), 1)
        origin, ell = superpoint_scene_normalization(rec['centers'])
        mu = torch.cat((rec['centers'], under['centers'], origin.expand(-1, 2, -1)), 1)
        radius = torch.cat((rec['radii'], under['radii'], ell[:, None].expand(-1, 2)), 1)
        joint_states, _ = self.anchor_decoder.forward_stateful(
            joint, memory['latent'], memory['rays'], None, None, None, None, 0,
            seed_mu=mu, seed_radius=radius, seed_c=None, seed_s=None,
            write_strength=0., reconstruction_only=True)
        # Intermediate reconstruction losses see ONLY the first 1024 tokens.
        states = [dict(s, tokens=s['tokens'][:, :nrec], mu=s['mu'][:, :nrec],
                       rho=s['rho'][:, :nrec], radii=s['radii'][:, :nrec]) for s in joint_states]
        final = states[-1]
        q = self.understanding_output_projection(joint_states[-1]['tokens'][:, nrec:])
        _, readout_ell = superpoint_scene_normalization(states[5]['mu'])
        a = self.panoptic.encode_token(final['tokens'], final['mu'], final['radii'], readout_ell)
        f_anchor = self.panoptic.ln_mask_a(self.panoptic.W_mask_a(a))
        gaussians = self.activation_head(final['tokens'], final['mu'], final['radii'])
        image_size = self.activation_head.opt.img_size if hasattr(self.activation_head, 'opt') else (256, 256)
        grid = F.interpolate(fm.flatten(0, 1), size=image_size, mode='bilinear', align_corners=False)
        grid = grid.reshape(1, 2, 256, *grid.shape[-2:])
        evidence, gate, _ = lift_features(grid, gaussians, memory['decoder'], memory['renderer'])
        child, _ = self.panoptic.gaussian_child_features(a, f_anchor, gaussians, final['mu'], final['radii'])
        features = gate*evidence+(1-gate)*child+.1*self.panoptic.W_res(child).tanh()
        logits = features@self.semantic_memory.mask_embedder(q).transpose(1, 2)
        classification = self.panoptic.classify(q)
        final.update(classification)
        return dict(states=states, gaussians=gaussians, gaussian_membership=logits.sigmoid(), gaussian_mask_logits=logits,
                    thing_logits19=classification['thing_logits19'], p_class=classification['p_class'],
                    q_pre=under['query'], unified_query=q[:, :n], superpoints=under, cloud=cloud,
                    num_thing_queries=n, **self.parent_metadata(rec['tokens'], n),
                    initial_anchor_mu=rec['centers'], initial_anchor_radius=rec['radii'],
                    initial_anchor_source_ids=rec['source_indices'], joint_token_count=nrec+n+2)

    def parameter_groups(self):
        groups = dict(shared=[], reconstruction=[], understanding=[])
        for name, p in self.named_parameters():
            if name.startswith(('blocks.', 'anchor_decoder.', 'reconstruction_memory.', 'shared_feature_projection.')):
                group = 'shared'
            elif name.startswith(('gs_tokens', 'anchor_content.', 'anchor_visual_projection.', 'activation_head.')):
                group = 'reconstruction'
            else:
                group = 'understanding'
            groups[group].append(p)
        return groups


class LocusGSConcatShared(LocusGSUnifiedQueryV1):
    architecture_name = 'OMEGA_POINT1024_SUPERPOINT_CONCAT_SHARED_DECODER_GFM_V1'
    core_type = ConcatSharedCore
    extra_checkpoint_metadata = dict(concat_shared=dict(reconstruction_tokens=1024, decoder_dim=1024,
        understanding_seed_dim=256, understanding_readout_dim=256, stuff_tokens=2, decoder_depth=12,
        gfm_adapters=1, token_attention='unrestricted_joint_self_attention', understanding_geometry='fixed_seed'))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.gfm_projection

    def encode_context(self, rgb):
        if rgb.shape != (1, 2, 3, 256, 256) or not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError('One finite two-view 256 RGB scene in [0,1] required')
        omega = self.frozen_omega(rgb)
        memory = self.core.prepare_memory(omega, rgb, self.gs)
        proposals = self.sam_proposals(rgb)[0] if self.sam_proposals is not None else None
        cloud = lift_context_cloud(omega, memory['point_feature_grid'], rgb, self.config, proposals)
        return omega, cloud, memory

    def parameter_groups(self):
        return self.core.parameter_groups()
