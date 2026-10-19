"""Joint-attention and one-adapter contracts, using synthetic CPU inputs."""
import unittest
import torch
from torch import nn

from tokengs.options import config_defaults
from tokengs.models.concat_shared_decoder import (
    LocusGSConcatShared, SharedGFMMemoryAdapter, ConcatPointSeededDecoder)
from tokengs.models.gfm_dual_stream_components import GFMReconstructionAdapter
from tokengs.models.enc_dec import DecoderBlock
from tokengs.models.input_types import EncoderLatent
from tests.test_superpoint_dual_point_reconstruction import build


class ConcatTests(unittest.TestCase):
    def test_one_shared_adapter_preserves_kv_and_both_output_gradients(self):
        old = GFMReconstructionAdapter()
        new = SharedGFMMemoryAdapter(); new.load_state_dict(old.state_dict())
        layers = [torch.randn(1, 2, 1024, 2048) for _ in range(4)]
        with torch.no_grad():
            expected = old(layers)
        out = new(layers)
        torch.testing.assert_close(out['latent'].keys, expected.keys, rtol=0, atol=0)
        torch.testing.assert_close(out['latent'].values, expected.values, rtol=0, atol=0)
        self.assertEqual(out['grid'].shape, (1, 2, 1024, 32, 32))
        for loss in (out['grid'].square().mean(), out['latent'].values.square().mean()):
            grad = torch.autograd.grad(loss, new.projection[0].weight, retain_graph=True)[0]
            self.assertGreater(float(grad.norm()), 0.)

    def tiny_decoder(self):
        opt = config_defaults['train_siu3r_object_locus_gfm_dual_stream_v4'].evolve(
            enc_embed_dim=16, enc_num_heads=4, num_gs_tokens=2,
            locusgs_pe_hidden_dim=16, locusgs_ray_bias_scale=0.)
        block = DecoderBlock(16, 4, 2, True, True, True)
        decoder = ConcatPointSeededDecoder(opt, [block]); decoder.reconstruction_count = 2
        return decoder, block

    def test_joint_self_attention_has_bidirectional_task_gradients_and_fixed_under_geometry(self):
        decoder, block = self.tiny_decoder()
        # Nonzero refinement proves the suffix stays fixed rather than merely
        # appearing fixed because the production geometry heads start at zero.
        for head in (decoder.refine_mu[0], decoder.refine_rho[0]):
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.constant_(module.bias, .1)
        rec = torch.randn(1, 2, 16, requires_grad=True)
        under = torch.randn(1, 3, 16, requires_grad=True)
        tokens = torch.cat((rec, under), 1)
        mu = torch.randn(1, 5, 3); radius = torch.full((1, 5), .2)
        latent = EncoderLatent(keys=torch.randn(1, 4, 7, 4), values=torch.randn(1, 4, 7, 4))
        states, _ = decoder.forward_stateful(tokens, latent, (torch.zeros(1, 7, 3), torch.ones(1, 7, 3)),
            None, None, None, None, 0, seed_mu=mu, seed_radius=radius, seed_c=None, seed_s=None,
            reconstruction_only=True)
        out = states[-1]
        torch.testing.assert_close(out['mu'][:, 2:], mu[:, 2:], rtol=0, atol=0)
        self.assertGreater(float((out['mu'][:, :2]-mu[:, :2]).abs().max()), 0.)
        rec_loss = out['tokens'][:, :2].square().mean()
        under_loss = out['tokens'][:, 2:].square().mean()
        self.assertGreater(float(torch.autograd.grad(rec_loss, under, retain_graph=True)[0].norm()), 0.)
        self.assertGreater(float(torch.autograd.grad(under_loss, rec, retain_graph=True)[0].norm()), 0.)
        self.assertGreater(float(torch.autograd.grad(rec_loss+under_loss, block.gs_cross_attn.q_proj.weight)[0].norm()), 0.)

    def test_no_parallel_decoder_or_gfm_adapter_and_complete_groups(self):
        with torch.device('meta'):
            model = build(LocusGSConcatShared)
        self.assertFalse(hasattr(model, 'gfm_projection'))
        self.assertFalse(hasattr(model.core.semantic_memory, 'projections'))
        self.assertFalse(hasattr(model.core.panoptic, 'layers'))
        self.assertEqual(sum(isinstance(m, GFMReconstructionAdapter) for m in model.modules()), 1)
        self.assertEqual(sum(isinstance(m, DecoderBlock) for m in model.modules()), 12)
        groups = model.parameter_groups()
        ids = [id(p) for ps in groups.values() for p in ps]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
        shared = {id(p) for p in groups['shared']}
        self.assertIn(id(model.core.reconstruction_memory.projection[0].weight), shared)
        self.assertIn(id(model.core.blocks[0].gs_self_attn.gs_self_attn.qkv.weight), shared)

    def test_real_1024_plus_superpoint_forward_all_twelve_layers(self):
        model = build(LocusGSConcatShared).eval()
        lengths = []
        hooks = [b.gs_self_attn.register_forward_pre_hook(lambda m, args: lengths.append(args[0].shape[1]))
                 for b in model.core.blocks]
        adapter_calls = []
        hooks.append(model.core.reconstruction_memory.register_forward_hook(lambda *args: adapter_calls.append(1)))
        with torch.no_grad():
            out = model.generate(torch.rand(1, 2, 3, 256, 256))
        for hook in hooks:
            hook.remove()
        self.assertEqual(adapter_calls, [1])
        self.assertEqual(lengths, [1028]*12)
        self.assertEqual(out['num_thing_queries'], 2)
        self.assertEqual(out['joint_token_count'], 1028)
        self.assertEqual(out['gaussians'].shape, (1, 65536, 14))
        self.assertEqual(out['gaussian_membership'].shape, (1, 65536, 4))
        self.assertEqual(out['thing_logits19'].shape, (1, 2, 19))
        self.assertEqual(len(out['states']), 12)
        for state in out['states']:
            self.assertEqual(state['tokens'].shape, (1, 1024, 1024))
        self.assertNotIn('gaussian_parent_query', out)
        self.assertTrue(torch.isfinite(out['gaussians']).all())
        self.assertTrue(torch.isfinite(out['gaussian_membership']).all())


if __name__ == '__main__':
    unittest.main()
