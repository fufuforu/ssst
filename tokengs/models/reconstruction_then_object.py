"""Complete point1024 reconstruction, then Object geometry -> semantic reads.

Same modules/parameters as A point1024. All four Object layers read the final
layer-12 reconstruction features, rather than interleaved layer6/8/10/12 reads.
No reconstruction_states replay: that legacy interface detaches its inputs.
"""
import torch
import torch.nn.functional as F

from tokengs.models.superpoint_dual_point_reconstruction import SuperpointDualPointReconstructionCore
from tokengs.models.superpoint_dual_core import superpoint_scene_normalization
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1


class ReconstructionThenObjectCore(SuperpointDualPointReconstructionCore):
    def object_updates(self, states, under, memory):
        if len(states) != 12:
            raise ValueError('Object decoding requires completed 12-layer reconstruction')
        final = states[-1]
        tokens, mu, radii = final['tokens'], final['mu'], final['radii']
        fm = memory['fm']
        b = tokens.shape[0]
        image = F.interpolate(fm.flatten(0, 1), size=(32, 32), mode='bilinear', align_corners=False)
        image = image.reshape(b, 2, 256, 1024).permute(0, 1, 3, 2).reshape(b, 2048, 256)
        # Keep A's normalization definition for an otherwise exact parameter
        # comparison; geometric evidence itself always comes from layer12.
        origin, ell = superpoint_scene_normalization(states[5]['mu'])
        a = self.panoptic.encode_token(tokens, mu, radii, ell)
        f_anchor = self.panoptic.ln_mask_a(self.panoptic.W_mask_a(a))
        stuff = self.panoptic.stuff_ln(fm.mean((1, 3, 4))[:, None]+self.panoptic.stuff_seed[None])
        q = torch.cat((under['query'], stuff), 1)
        c = under['centers']; s = under['radii'][..., None].expand(-1, -1, 3)
        for layer in (6, 8, 10, 12):
            # DynamicObjectLayer already implements geometry attention ->
            # semantic-image attention -> query self attention -> FFN.
            result = self.panoptic.layers[f'L{layer}'](
                tokens, a, mu, q, c, s, ell, image, self.semantic_memory.mask_embedder, f_anchor, 0)
            q, c, s = result['q'], result['c'], result['s']
            result.update(joint_delta=result['joint_delta']*0., scene_origin=origin,
                          reconstruction_read_layer=12, reconstruction_read_count=tokens.shape[1])
            # Never overwrite original reconstruction tokens/geometry, never
            # detach geometric evidence, and never inject Object writes.
            states[layer-1].update(result)
        return states

    def decode_streams(self, tokens, rec, under, memory):
        if tokens.shape[1] != 1024:
            raise ValueError('Exactly 1024 point-initialized reconstruction tokens required')
        states, _ = self.anchor_decoder.forward_stateful(
            tokens, memory['latent'], memory['rays'], None, None, None, None, 0,
            seed_mu=rec['centers'], seed_radius=rec['radii'], seed_c=None, seed_s=None,
            write_strength=0., reconstruction_only=True)
        return self.object_updates(states, under, memory)


class LocusGSReconstructionThenObject(LocusGSUnifiedQueryV1):
    architecture_name = 'OMEGA_POINT1024_RECON_THEN_OBJECT_GEOMETRY_SEMANTIC_V1'
    core_type = ReconstructionThenObjectCore
    extra_checkpoint_metadata = dict(reconstruction_then_object=dict(reconstruction_tokens=1024,
        reconstruction_depth=12, object_depth=4, reconstruction_read_layers=[12, 12, 12, 12],
        object_update_order=['geometry_cross_attention', 'semantic_cross_attention', 'query_self_attention', 'ffn'],
        object_initialization='superpoint_features', detached_reconstruction=False, write_strength=0.))

    def encode_context(self, rgb):
        omega, cloud, _ = super().encode_context(rgb)
        return omega, cloud, self.core.prepare_memory(omega, rgb, self.gs)
