"""CPU contract tests for the spatially grounded shared-token joint model."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from tokengs.models import model_registry
from tokengs.models.input_types import ModelInputDecoder, split_data
from tokengs.models.enc_dec import DecoderBlock
from tokengs.models.spatial_grounded_tokens import (
    anchor_ray_bias,
    anchor_ray_distance,
    build_anchor_encoding,
    patch_rays,
)
from tokengs.models.ssst_contracts import (
    QUERY_COUNT,
    SEMANTIC_CLASS_COUNT,
    TRAIN_TARGET_RECORDS,
    query_semantic_maps,
    validate_variable_target_batch,
    validate_view_protocol,
)
from tokengs.models.ssst_diagnostics import (
    query_layer_scale_stats,
    shared_gradient_diagnostic,
    token_purity_metrics,
)
from tokengs.models.ssst_loss import (
    build_context_segments,
    class_aware_context_loss,
    spatial_regularization,
    understanding_weight,
)
from tokengs.options import config_defaults


class FakeRenderer:
    """Deterministic CPU stand-in for the gsplat renderer."""

    def __init__(self, img_size):
        self.img_size = img_size

    def render(self, gaussians, cam_view, bg_color=None, intrinsics=None):
        batch, views = cam_view.shape[:2]
        height, width = self.img_size
        num = gaussians.shape[1]
        base = gaussians[..., 0].mean(dim=1).view(batch, 1, 1, 1, 1)
        return {
            "images_pred": torch.zeros(batch, views, 3, height, width) + base,
            "alphas_pred": torch.zeros(batch, views, 1, height, width) + torch.sigmoid(base),
            "depths_pred": torch.ones(batch, views, 1, height, width),
            "means2d_pred": torch.zeros(batch, views, num, 2),
        }

    def render_feature_channels(self, gaussians, features, cam_view, intrinsics, opacity_scale=1.0):
        batch, views = cam_view.shape[:2]
        height, width = self.img_size
        channels = features.shape[-1]
        per_gaussian = features.mean(dim=1).view(batch, 1, channels, 1, 1)
        images = per_gaussian.expand(batch, views, channels, height, width).contiguous()
        alphas = torch.ones(batch, views, 1, height, width)
        return {"images_pred": images, "alphas_pred": alphas}


def tiny_options():
    return config_defaults["train_siu3r_ssst"].evolve(
        workspace="/tmp/ssst_unit",
        img_size=(32, 32),
        patch_size=8,
        dec_patch_size=8,
        enc_depth=1,
        dec_depth=2,
        enc_embed_dim=64,
        enc_num_heads=4,
        num_gs_tokens=8,
        token_dim=64,
        num_object_query_layers=1,
        lambda_lpips=0.0,
        lambda_ssim=0.0,
        lambda_visibility=0.0,
        mixed_precision="no",
        num_workers=0,
    )


def synthetic_batch(opt, batch_size=1, views=TRAIN_TARGET_RECORDS, height=32, width=32):
    channels = 3 + 6
    images = torch.rand(batch_size, views, channels, height, width)
    semantic = torch.zeros(batch_size, views, height, width, dtype=torch.long)
    instance = torch.zeros(batch_size, views, height, width, dtype=torch.long)
    semantic[:, :, : height // 2] = 0
    semantic[:, :, height // 2 :] = 3
    instance[:, :, height // 2 :] = 5
    intrinsics = torch.tensor([[[16.0, 16.0, 8.0, 8.0]] * views]).repeat(batch_size, 1, 1)
    cam_view = torch.eye(4).repeat(batch_size, views, 1, 1)
    # Non-degenerate rays so the anchor-to-ray geometry is actually exercised.
    grid = torch.arange(height * width, dtype=torch.float32).reshape(height, width)
    directions = torch.stack(
        [0.2 * torch.cos(grid / 37.0), 0.2 * torch.sin(grid / 53.0), torch.ones_like(grid)]
    )[None, None]
    return {
        "input": images,
        "images_all": images[:, :, :3],
        "images_input": images[:, : opt.num_input_views, :3],
        "images_output": images[:, opt.num_input_views :, :3],
        "masks_all": torch.ones(batch_size, views, 1, height, width),
        "masks_output": torch.ones(batch_size, views - opt.num_input_views, 1, height, width),
        "has_mask": torch.ones(batch_size, dtype=torch.bool),
        "rays_os": torch.zeros(batch_size, views, 3, height, width),
        "rays_ds": directions.repeat(batch_size, views, 1, 1, 1),
        "intrinsics_all": intrinsics,
        "intrinsics": intrinsics[:, opt.num_input_views :],
        "intrinsics_input": intrinsics[:, : opt.num_input_views],
        "cam_view_all": cam_view,
        "cam_view": cam_view[:, opt.num_input_views :],
        "cam_to_world_input": torch.eye(4).repeat(batch_size, opt.num_input_views, 1, 1),
        "semantic_label_all": semantic,
        "instance_label_all": instance,
        "frame_ids": torch.tensor([[1, 2, 3, 4]] * batch_size, dtype=torch.long),
    }


class ViewProtocolTests(unittest.TestCase):
    def test_protocol_accepts_released_view_counts(self):
        validate_view_protocol(context_views=2, target_records=4, phase="train")
        validate_view_protocol(context_views=2, target_records=6, phase="validation")

    def test_protocol_rejects_other_view_counts(self):
        with self.assertRaises(ValueError):
            validate_view_protocol(context_views=2, target_records=6, phase="train")
        with self.assertRaises(ValueError):
            validate_view_protocol(context_views=4, target_records=4, phase="train")

    def test_batch_validation_requires_context_and_novel_views(self):
        opt = tiny_options()
        batch = synthetic_batch(opt)
        validate_variable_target_batch(batch, phase="train")
        broken = dict(batch)
        broken["images_input"] = batch["images_input"][:, :1]
        with self.assertRaises(ValueError):
            validate_variable_target_batch(broken, phase="train")


class SpatialTokenTests(unittest.TestCase):
    def test_anchor_encoding_shape(self):
        anchors = torch.zeros(2, 5, 3)
        radii = torch.ones(2, 5) * 0.1
        encoding = build_anchor_encoding(anchors, radii, num_freqs=4, extent=0.3)
        self.assertEqual(tuple(encoding.shape), (2, 5, 4 + 6 * 4))

    def test_model_tokens_are_local_and_bounded(self):
        opt = tiny_options()
        model = model_registry["siu3r_joint_ssst"](opt).eval()
        model.gs = FakeRenderer(opt.img_size)
        batch = synthetic_batch(opt)
        with torch.no_grad():
            output = model(batch, skip_loss=True)
        anchors = output["anchors"]
        radii = output["radii"]
        self.assertEqual(tuple(anchors.shape), (1, opt.num_gs_tokens, 3))
        self.assertEqual(tuple(radii.shape), (1, opt.num_gs_tokens))
        self.assertTrue(torch.all(radii > 0))
        # x, y are centered on the camera axis; z is centered on the prior depth.
        self.assertLessEqual(anchors[..., 0].abs().max().item(), opt.anchor_extent + 1e-6)
        self.assertLessEqual(anchors[..., 1].abs().max().item(), opt.anchor_extent + 1e-6)
        self.assertGreaterEqual(anchors[..., 2].min().item(), opt.anchor_center_z - opt.anchor_extent - 1e-6)
        self.assertLessEqual(anchors[..., 2].max().item(), opt.anchor_center_z + opt.anchor_extent + 1e-6)
        positions = output["gaussians"][..., :3].reshape(
            1, opt.num_gs_tokens, model.gaussians_per_token, 3
        )
        offset = (positions - anchors.unsqueeze(2)).norm(dim=-1)
        self.assertLessEqual(
            offset.max().item(),
            opt.anchor_local_offset_bound * radii.max().item() + 1e-5,
        )

    def test_decoder_is_identity_to_tokengs_when_ray_bias_is_disabled(self):
        # With the geometric bias off, the zero-initialized anchor projection and
        # refinement heads make the SSST decoder numerically identical to the
        # plain TokenGS decoder at initialization.
        opt = tiny_options().evolve(anchor_ray_bias=False)
        model = model_registry["siu3r_joint_ssst"](opt).eval()
        batch = synthetic_batch(opt)
        from tokengs.models.input_types import split_data

        with torch.no_grad():
            model_input, _ = split_data(batch, opt)
            latent = model.forward_encoder(model_input.encoder)
            base = model.get_gs_tokens(batch_size=1)
            spatial = model.spatial_decoder(base, latent)
            plain = base
            for block in model.enc_dec_backbone.decoder_blocks:
                plain = block(gs_tokens=plain, keys=latent.keys, values=latent.values)
        self.assertTrue(torch.allclose(plain, spatial.tokens, atol=1e-6))
        del model_input


class LossTests(unittest.TestCase):
    def test_understanding_curriculum_is_monotonic(self):
        opt = tiny_options()
        weights = [understanding_weight(step, opt) for step in (0, 500, 1000, 2000, 4000)]
        self.assertAlmostEqual(weights[0], opt.understanding_start_weight)
        self.assertAlmostEqual(weights[-1], opt.understanding_final_weight)
        for earlier, later in zip(weights, weights[1:]):
            self.assertLessEqual(earlier, later)

    def test_context_segments_split_stuff_and_things(self):
        semantic = torch.zeros(1, 4, 16, 16, dtype=torch.long)
        instance = torch.zeros(1, 4, 16, 16, dtype=torch.long)
        semantic[:, :, 8:] = 3
        instance[:, :, 8:] = 7
        classes, masks = build_context_segments(semantic, instance, (0, 1))
        self.assertEqual(classes[0].tolist(), [0, 3])
        self.assertEqual(tuple(masks[0].shape), (2, 2, 16, 16))

    def test_understanding_loss_is_finite_and_differentiable(self):
        semantic = torch.zeros(1, 4, 16, 16, dtype=torch.long)
        instance = torch.zeros(1, 4, 16, 16, dtype=torch.long)
        semantic[:, :, 8:] = 3
        instance[:, :, 8:] = 7
        classes, masks = build_context_segments(semantic, instance, (0, 1))
        class_logits = torch.randn(1, QUERY_COUNT, SEMANTIC_CLASS_COUNT + 1, requires_grad=True)
        mask_logits = torch.randn(1, QUERY_COUNT, 2, 16, 16, requires_grad=True)
        result = class_aware_context_loss(class_logits, mask_logits, classes, masks)
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertTrue(torch.isfinite(class_logits.grad).all())
        self.assertTrue(torch.isfinite(mask_logits.grad).all())

    def test_spatial_regularization_penalizes_large_radii(self):
        anchors = torch.zeros(1, 4, 3, requires_grad=True)
        radii = torch.full((1, 4), 0.05, requires_grad=True)
        gaussians = torch.zeros(1, 8, 14)
        gaussians[..., :3] = anchors.detach().repeat_interleave(2, dim=1) + 0.01
        small = spatial_regularization(
            gaussians, anchors, radii, gaussians_per_token=2,
            compactness_weight=1e-3, radius_weight=1e-3,
            radius_soft_min=0.01, radius_soft_max=0.25,
        )
        large = spatial_regularization(
            gaussians, anchors, radii.detach() * 10, gaussians_per_token=2,
            compactness_weight=1e-3, radius_weight=1e-3,
            radius_soft_min=0.01, radius_soft_max=0.25,
        )
        self.assertLess(float(small["loss_radius"]), float(large["loss_radius"]))


class JointStepTests(unittest.TestCase):
    def test_joint_step_backward_is_finite(self):
        opt = tiny_options()
        model = model_registry["siu3r_joint_ssst"](opt).train()
        model.gs = FakeRenderer(opt.img_size)
        batch = synthetic_batch(opt)
        output, metrics = model.joint_step(batch, step=0, phase="train")
        self.assertTrue(torch.isfinite(metrics["loss"]))
        self.assertEqual(tuple(output["query_class_logits"].shape), (1, QUERY_COUNT, 21))
        self.assertEqual(
            tuple(output["query_assignment_logits"].shape), (1, QUERY_COUNT, opt.num_gs_tokens)
        )
        self.assertEqual(
            tuple(output["gaussians"].shape),
            (1, opt.num_gs_tokens * opt.dec_patch_size**2, 14),
        )
        metrics["loss"].backward()
        anchor_grad = model.spatial_decoder.anchor_pre.grad
        self.assertIsNotNone(anchor_grad)
        self.assertTrue(torch.isfinite(anchor_grad).all())
        self.assertTrue(torch.isfinite(model.object_queries.class_head.weight.grad).all())

    def test_query_mask_renderer_uses_gaussian_geometry(self):
        opt = tiny_options()
        model = model_registry["siu3r_joint_ssst"](opt).eval()
        model.gs = FakeRenderer(opt.img_size)
        gaussians = torch.rand(1, opt.num_gs_tokens * opt.dec_patch_size**2, 14)
        assignment = torch.softmax(torch.randn(1, QUERY_COUNT, opt.num_gs_tokens), dim=1)
        decoder = ModelInputDecoder(
            cam_view=torch.eye(4).repeat(1, opt.num_input_views, 1, 1),
            intrinsics=torch.tensor([[[16.0, 16.0, 8.0, 8.0]] * opt.num_input_views]),
        )
        mask_prob, mask_logits = model.render_query_masks(gaussians, assignment, decoder)
        self.assertEqual(
            tuple(mask_prob.shape), (1, QUERY_COUNT, opt.num_input_views, 32, 32)
        )
        self.assertEqual(mask_prob.shape, mask_logits.shape)


class GradientAccumulationTests(unittest.TestCase):
    """The accumulation window must span epochs without losing gradients."""

    @staticmethod
    def _events(epoch_sizes, *, num_steps, grad_accum, start_step=0):
        from scripts.run_ssst_joint import accumulate_microbatches

        epochs = [
            [f"epoch{index}-batch{batch}" for batch in range(size)]
            for index, size in enumerate(epoch_sizes)
        ]
        return list(
            accumulate_microbatches(
                lambda epoch: iter(epochs[epoch]),
                num_steps=num_steps,
                grad_accum=grad_accum,
                start_step=start_step,
            )
        )

    def test_every_optimizer_step_consumes_exactly_grad_accum_microbatches(self):
        # 5 batches per epoch, grad_accum = 3 -> the window crosses the boundary.
        events = self._events([5, 5, 5], num_steps=5, grad_accum=3)
        self.assertEqual(len(events), 15)
        windows, current = [], []
        for step, _, batch, do_step in events:
            current.append(batch)
            if do_step:
                windows.append((step, current))
                current = []
        self.assertEqual([len(window) for _, window in windows], [3, 3, 3, 3, 3])
        self.assertEqual([step for step, _ in windows], [1, 2, 3, 4, 5])
        # The second window starts in epoch 0 and finishes in epoch 1: the
        # partial gradient is carried over the epoch boundary, not dropped.
        self.assertEqual(
            windows[1][1], ["epoch0-batch3", "epoch0-batch4", "epoch1-batch0"]
        )
        self.assertEqual(current, [])

    def test_partial_window_is_not_stepped_at_the_end_of_training(self):
        events = self._events([4, 4], num_steps=2, grad_accum=3)
        self.assertEqual(len(events), 6)
        self.assertEqual(sum(1 for *_, do_step in events if do_step), 2)

    def test_window_resumes_from_an_existing_step(self):
        events = self._events([4], num_steps=7, grad_accum=2, start_step=6)
        self.assertEqual([step for step, *_, do_step in events if do_step], [7])
        self.assertEqual(len(events), 2)

    def test_empty_epoch_fails_closed(self):
        from scripts.run_ssst_joint import accumulate_microbatches

        with self.assertRaises(RuntimeError):
            list(
                accumulate_microbatches(
                    lambda epoch: iter([]), num_steps=1, grad_accum=1
                )
            )


class AnchorRayGeometryTests(unittest.TestCase):
    def test_patch_rays_match_encoder_patch_layout(self):
        opt = tiny_options()
        batch = synthetic_batch(opt, height=256, width=256)
        pooled_o, pooled_d = patch_rays(
            batch["rays_os"][:, : opt.num_input_views],
            batch["rays_ds"][:, : opt.num_input_views],
            patch_size=opt.patch_size,
        )
        patches_per_view = (256 // opt.patch_size) ** 2
        self.assertEqual(
            tuple(pooled_o.shape),
            (1, opt.num_input_views * patches_per_view, 3),
        )
        self.assertEqual(pooled_o.shape, pooled_d.shape)
        # View-major ordering: the first patch of view 1 is the pooled ray of
        # the first 8x8 block of view 1, not of view 0.
        self.assertTrue(
            torch.allclose(
                pooled_d[0, patches_per_view],
                batch["rays_ds"][0, 1, :, : opt.patch_size, : opt.patch_size].mean(dim=(-1, -2)),
            )
        )
        # The encoder itself must produce exactly that many tokens.
        model = model_registry["siu3r_joint_ssst"](opt).eval()
        from tokengs.models.input_types import split_data

        with torch.no_grad():
            model_input, _ = split_data(batch, opt)
            latent = model.forward_encoder(model_input.encoder)
        self.assertEqual(latent.keys.shape[-2], pooled_o.shape[1])

    def test_distance_matches_plucker_formulation(self):
        torch.manual_seed(0)
        anchors = torch.randn(2, 6, 3)
        rays_o = torch.randn(2, 9, 3)
        rays_d = torch.randn(2, 9, 3)
        distance = anchor_ray_distance(anchors, rays_o, rays_d)
        self.assertEqual(tuple(distance.shape), (2, 6, 9))
        self.assertTrue(torch.all(distance >= 0))
        # Plucker form: m = o x d, D = || mu x d - m || / ||d||
        plucker = torch.cross(rays_o, rays_d, dim=-1)
        numerator = torch.cross(
            anchors.unsqueeze(2).expand(-1, -1, rays_o.shape[1], -1),
            rays_d.unsqueeze(1).expand(-1, anchors.shape[1], -1, -1),
            dim=-1,
        ) - plucker.unsqueeze(1)
        expected = numerator.norm(dim=-1) / rays_d.norm(dim=-1).unsqueeze(1)
        self.assertTrue(torch.allclose(distance, expected, atol=1e-5))

    def test_distance_is_zero_on_the_ray_and_grows_off_axis(self):
        rays_o = torch.zeros(1, 1, 3)
        rays_d = torch.tensor([[[0.0, 0.0, 1.0]]])
        on_ray = torch.tensor([[[0.0, 0.0, 0.5]]])
        off_ray = torch.tensor([[[0.3, 0.0, 0.5]]])
        self.assertLess(float(anchor_ray_distance(on_ray, rays_o, rays_d)), 1e-6)
        self.assertAlmostEqual(
            float(anchor_ray_distance(off_ray, rays_o, rays_d)), 0.3, places=5
        )

    def test_bias_is_finite_non_positive_and_radius_sharpened(self):
        opt = tiny_options()
        # Anchor slightly off a forward ray; the second ray points sideways, so
        # the same anchor is far from it.
        anchors = torch.tensor([[[0.05, 0.0, 0.5]]])
        rays_o = torch.zeros(1, 2, 3)
        rays_d = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
        distance = anchor_ray_distance(anchors, rays_o, rays_d)
        self.assertAlmostEqual(float(distance[0, 0, 0]), 0.05, places=5)
        self.assertGreater(float(distance[0, 0, 1]), 0.4)
        radii = torch.full((1, 1), 0.05)
        bias = anchor_ray_bias(distance, radii, sigma0=1.0)
        self.assertEqual(tuple(bias.shape), (1, 1, 1, 2))
        self.assertTrue(torch.isfinite(bias).all())
        self.assertTrue(torch.all(bias <= 0))
        near, far = bias[0, 0, 0, 0], bias[0, 0, 0, 1]
        self.assertGreater(float(near), float(far))
        # Same geometry, smaller radius => sharper (more negative) bias.
        small = anchor_ray_bias(distance, radii, sigma0=1.0)
        large = anchor_ray_bias(distance, radii * 4.0, sigma0=1.0)
        self.assertLess(float(small[0, 0, 0, 1]), float(large[0, 0, 0, 1]))

    def test_default_sigma_saturates_beyond_the_clamp(self):
        # Documents the default regime: radius 0.05 and sigma0 0.1 give an
        # effective width of 0.005, so a 0.1 offset already clamps to -20.
        opt = tiny_options()
        far_away = anchor_ray_bias(
            torch.full((1, 1, 1), 0.1),
            torch.full((1, 1), 0.05),
            sigma0=opt.anchor_ray_sigma0,
            clamp_min=opt.anchor_ray_bias_clamp,
        )
        self.assertAlmostEqual(float(far_away), opt.anchor_ray_bias_clamp)
        near = anchor_ray_bias(
            torch.full((1, 1, 1), 0.001),
            torch.full((1, 1), 0.05),
            sigma0=opt.anchor_ray_sigma0,
            clamp_min=opt.anchor_ray_bias_clamp,
        )
        self.assertAlmostEqual(float(near), -0.02, places=5)

    def test_geometry_bias_receives_gradients(self):
        anchors = torch.zeros(1, 1, 3, requires_grad=True)
        radii = torch.full((1, 1), 0.05, requires_grad=True)
        # 0.01 offset stays inside the unclamped region at the default sigma0,
        # so the anchor and radius must receive geometry gradients.
        offset_anchor = anchors + torch.tensor([[[0.0, 0.01, 0.0]]])
        distance = anchor_ray_distance(
            offset_anchor,
            torch.zeros(1, 2, 3),
            torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
        )
        bias = anchor_ray_bias(
            distance,
            radii,
            sigma0=0.1,
        )
        self.assertGreater(float(bias.min()), -20.0)
        bias.sum().backward()
        self.assertIsNotNone(anchors.grad)
        self.assertGreater(float(anchors.grad.abs().sum()), 0.0)
        self.assertGreater(float(radii.grad.abs().sum()), 0.0)

    def test_decoder_block_without_bias_is_unchanged(self):
        torch.manual_seed(0)
        block = DecoderBlock(
            dim=16,
            num_heads=4,
            mlp_ratio=2.0,
            qkv_bias=True,
            ffn_bias=True,
            qk_norm=True,
            init_values=None,
        )
        tokens = torch.randn(2, 3, 16)
        keys = torch.randn(2, 4, 5, 4)  # [B, H, N, Dh]
        values = torch.randn(2, 4, 5, 4)
        without_bias = block(tokens, keys, values)
        zero_bias = block(tokens, keys, values, attn_bias=torch.zeros(2, 1, 3, 5))
        self.assertTrue(torch.allclose(without_bias, zero_bias, atol=1e-6))
        mixed_bias = torch.zeros(2, 1, 3, 5)
        mixed_bias[..., :2] = -5.0  # constant offsets cancel, so use a shaped bias
        with_bias = block(tokens, keys, values, attn_bias=mixed_bias)
        self.assertFalse(torch.allclose(without_bias, with_bias, atol=1e-6))

    def test_ssst_decoder_uses_ray_bias_and_stays_finite(self):
        opt = tiny_options()
        model = model_registry["siu3r_joint_ssst"](opt).eval()
        model.gs = FakeRenderer(opt.img_size)
        batch = synthetic_batch(opt, height=256, width=256)
        from tokengs.models.input_types import split_data

        with torch.no_grad():
            model_input, _ = split_data(batch, opt)
            latent = model.forward_encoder(model_input.encoder)
            pooled = patch_rays(
                model_input.encoder.rays_os, model_input.encoder.rays_ds, patch_size=opt.patch_size
            )
            spatial = model.spatial_decoder(
                model.get_gs_tokens(batch_size=1), latent, patch_rays=pooled
            )
        self.assertTrue(torch.isfinite(spatial.tokens).all())
        self.assertIn("ray_bias_mean", spatial.stats)
        self.assertLess(float(spatial.stats["ray_bias_mean"]), 0.0)

    def test_ray_bias_fails_closed_with_latent_bottleneck(self):
        opt = tiny_options().evolve(use_latent_bottleneck=True)
        model = model_registry["siu3r_joint_ssst"](opt).eval()
        batch = synthetic_batch(opt, height=256, width=256)
        with self.assertRaises(ValueError):
            model(batch, skip_loss=True)


class SpatialConditioningTests(unittest.TestCase):
    def test_query_head_consumes_anchors_and_radii(self):
        opt = tiny_options()
        model = model_registry["siu3r_joint_ssst"](opt).eval()
        tokens = torch.randn(1, opt.num_gs_tokens, opt.enc_embed_dim)
        anchors = torch.zeros(1, opt.num_gs_tokens, 3)
        radii = torch.full((1, opt.num_gs_tokens), 0.05)
        with torch.no_grad():
            base = model.object_queries(tokens, anchors, radii)
            moved = model.object_queries(tokens, anchors + 0.2, radii)
            resized = model.object_queries(tokens, anchors, radii * 3.0)
        self.assertGreater(
            float((base.assignment_logits - moved.assignment_logits).abs().max()), 0.0
        )
        self.assertGreater(
            float((base.assignment_logits - resized.assignment_logits).abs().max()), 0.0
        )
        self.assertGreater(
            float((base.query_features - moved.query_features).abs().max()), 0.0
        )

    def test_query_semantic_maps_suppress_no_object_queries(self):
        class_logits = torch.full((1, 2, SEMANTIC_CLASS_COUNT + 1), -5.0)
        class_logits[0, 0, SEMANTIC_CLASS_COUNT] = 5.0  # query 0 = no-object
        class_logits[0, 1, 0] = 5.0  # query 1 = class 0
        mask_prob = torch.ones(1, 2, 1, 2, 2)
        semantic_prob, per_query = query_semantic_maps(class_logits, mask_prob)
        self.assertLess(float(per_query[0, 0, 0]), 1e-3)
        self.assertGreater(float(per_query[0, 1, 0]), 0.99)
        self.assertLess(float(semantic_prob[0, 0].sum()), mask_prob[0, 1, 0].sum() + 1e-3)


class ConstantChannelRenderer:
    """Renders explicit per-channel constant maps (test double for token mass)."""

    def __init__(self, maps, gaussians_per_token):
        self.maps = maps  # [N_token, H, W]
        self.gaussians_per_token = gaussians_per_token

    def render_feature_channels(self, gaussians, features, cam_view, intrinsics, opacity_scale=1.0):
        batch, views = cam_view.shape[:2]
        channels = features.shape[-1]
        selected = []
        for channel in range(channels):
            token = int(features[0, :, channel].argmax()) // self.gaussians_per_token
            selected.append(self.maps[token])
        stacked = torch.stack(selected).to(torch.float32)
        height, width = stacked.shape[-2:]
        images = stacked[None, None].expand(batch, views, -1, -1, -1)
        return {
            "images_pred": images.contiguous(),
            "alphas_pred": torch.ones(batch, views, 1, height, width),
        }


class TokenPurityDiagnosticTests(unittest.TestCase):
    def test_token_mass_and_purity_follows_rendered_contribution(self):
        opt = tiny_options()
        model = model_registry["siu3r_joint_ssst"](opt).eval()
        height = width = 8
        tokens = 4
        gaussians_per_token = 2
        model.gaussians_per_token = gaussians_per_token
        gaussians = torch.zeros(1, tokens * gaussians_per_token, 14)
        maps = torch.zeros(tokens, height, width)
        maps[0, :, : width // 2] = 1.0  # token 0 -> instance A only
        maps[1, :, width // 2 :] = 1.0  # token 1 -> instance B only
        maps[2, :, :] = 0.5  # token 2 -> split evenly
        model.gs = ConstantChannelRenderer(maps, gaussians_per_token)
        semantic = torch.full((1, 1, height, width), 3, dtype=torch.long)
        instance = torch.zeros(1, 1, height, width, dtype=torch.long)
        instance[..., : width // 2] = 7
        instance[..., width // 2 :] = 9
        decoder = ModelInputDecoder(
            cam_view=torch.eye(4).repeat(1, 1, 1, 1),
            intrinsics=torch.tensor([[[16.0, 16.0, 4.0, 4.0]]]),
        )
        metrics = token_purity_metrics(
            model, gaussians, decoder, semantic, instance, chunk_size=2, min_mass=1.0
        )
        self.assertAlmostEqual(float(metrics["token_gt_purity_mean"]), (1 + 1 + 0.5) / 3, places=4)
        self.assertAlmostEqual(float(metrics["token_gt_purity_median"]), 1.0, places=4)
        self.assertAlmostEqual(
            float(metrics["token_gt_purity_gt_08_ratio"]), 2 / 3, places=4
        )
        self.assertAlmostEqual(
            float(metrics["token_gt_instance_entropy_mean"]), float(torch.log(torch.tensor(2.0))) / 3,
            places=4,
        )
        self.assertAlmostEqual(float(metrics["token_gt_valid_token_ratio"]), 0.75, places=4)
        self.assertAlmostEqual(float(metrics["token_gt_background_mass_ratio"]), 0.0, places=4)
        self.assertEqual(int(metrics["token_gt_thing_instance_count"]), 2)


class GradientPathTests(unittest.TestCase):
    GROUNDING = (
        "spatial_decoder.anchor_pre",
        "spatial_decoder.radius_pre",
        "spatial_decoder.refine_heads.0.weight",
        "spatial_decoder.anchor_pe_proj.weight",
        "spatial_decoder.ray_bias_raw",
    )

    def _setup(self):
        opt = tiny_options()
        model = model_registry["siu3r_joint_ssst"](opt).train()
        model.gs = FakeRenderer(opt.img_size)
        return model, synthetic_batch(opt), opt

    def test_understanding_only_gradient_reaches_spatial_grounding(self):
        model, batch, _ = self._setup()
        _, metrics = model.joint_step(batch, step=0, phase="train")
        named = dict(model.named_parameters())
        parameters = [named[name] for name in self.GROUNDING]
        grads = torch.autograd.grad(
            metrics["loss_understanding"], parameters, retain_graph=True, allow_unused=True
        )
        for name, gradient in zip(self.GROUNDING, grads):
            self.assertIsNotNone(gradient, f"no understanding gradient for {name}")
            self.assertGreater(float(gradient.abs().sum()), 0.0, f"zero grad for {name}")

    def test_reconstruction_only_gradient_reaches_spatial_grounding(self):
        model, batch, _ = self._setup()
        _, metrics = model.joint_step(batch, step=0, phase="train")
        named = dict(model.named_parameters())
        parameters = [named[name] for name in self.GROUNDING]
        grads = torch.autograd.grad(
            metrics["loss_recon"], parameters, retain_graph=False, allow_unused=True
        )
        for name, gradient in zip(self.GROUNDING, grads):
            self.assertIsNotNone(gradient, f"no reconstruction gradient for {name}")
        self.assertGreater(float(grads[0].abs().sum()), 0.0)

    def test_shared_gradient_diagnostic_reports_reachability(self):
        model, batch, _ = self._setup()
        report = shared_gradient_diagnostic(model, batch, step=0, phase="train")
        self.assertGreater(report["grad_recon_norm"], 0.0)
        self.assertGreater(report["grad_understanding_norm"], 0.0)
        self.assertTrue(report["grad_cosine_defined"])
        self.assertGreaterEqual(report["grad_recon_understanding_cosine"], -1.0)
        self.assertLessEqual(report["grad_recon_understanding_cosine"], 1.0)
        reach = report["grad_spatial_grounding"]
        for name in ("spatial_decoder.anchor_pre", "spatial_decoder.radius_pre",
                     "spatial_decoder.refine_heads.0.weight"):
            self.assertTrue(reach[name]["received_understanding_grad"], name)
            self.assertGreater(reach[name]["understanding_grad_norm"], 0.0)


class WarmStartTests(unittest.TestCase):
    def _legacy_head_state(self, model):
        head = model.activation_head
        rows = head.deconv.weight.shape[0]
        channels = head.output_dims
        weight = torch.arange(rows, dtype=torch.float32).reshape(rows, 1).repeat(1, head.deconv.weight.shape[1])
        bias = torch.arange(rows, dtype=torch.float32)
        return weight, bias, channels, rows

    def test_partial_head_load_skips_legacy_absolute_xyz_channels(self):
        opt = tiny_options()
        model = model_registry["siu3r_joint_ssst"](opt)
        weight, bias, channels, rows = self._legacy_head_state(model)
        before_weight = model.activation_head.deconv.weight.detach().clone()
        before_bias = model.activation_head.deconv.bias.detach().clone()
        checkpoint = {
            "enc_dec_backbone.encoder_norm.weight": model.state_dict()[
                "enc_dec_backbone.encoder_norm.weight"
            ].clone(),
            "activation_head.deconv.weight": weight,
            "activation_head.deconv.bias": bias,
        }
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(checkpoint, path)
            report = model.init_from_checkpoint(str(path), log=messages.append)
        after_weight = model.activation_head.deconv.weight.detach()
        after_bias = model.activation_head.deconv.bias.detach()
        row_channel = torch.arange(rows) % channels
        position_rows = row_channel < 3
        # channels 0:3 keep the fresh initialization ...
        self.assertTrue(
            torch.equal(after_weight[position_rows], before_weight[position_rows])
        )
        self.assertTrue(torch.equal(after_bias[position_rows], before_bias[position_rows]))
        # ... while channels 3:14 take the checkpoint values.
        self.assertTrue(
            torch.equal(after_weight[~position_rows], weight[~position_rows])
        )
        self.assertTrue(torch.equal(after_bias[~position_rows], bias[~position_rows]))
        self.assertIn("activation_head.deconv.weight", report["partially_loaded"])
        self.assertIn("activation_head.deconv.bias", report["partially_loaded"])
        self.assertEqual(len(report["skipped"]), 2)
        self.assertTrue(
            any("skipped legacy absolute-XYZ channels" in message for message in messages)
        )
        self.assertIn(
            "enc_dec_backbone.encoder_norm.weight", report["loaded"]
        )


class TrainerCliTests(unittest.TestCase):
    """The trainer CLI must never feed `None` into numeric comparisons."""

    MINIMAL = [
        "--workspace", "/tmp/ssst_cli_test",
        "--num-steps", "1",
    ]

    @staticmethod
    def _options(argv):
        from scripts.run_ssst_joint import build_options, parse_args

        return parse_args(argv), build_options(parse_args(argv))

    def test_gradient_diagnostic_freq_defaults_to_the_option_value(self):
        args, opt = self._options(self.MINIMAL)
        # argparse may leave the flag unset ...
        self.assertIsNone(args.gradient_diagnostic_freq)
        # ... but the effective value must come from Options, never None.
        self.assertEqual(int(opt.gradient_diagnostic_freq), 200)
        self.assertGreater(int(opt.gradient_diagnostic_freq), 0)
        # This is exactly the comparison the training loop performs.
        self.assertTrue(int(opt.gradient_diagnostic_freq) > 0)

    def test_gradient_diagnostic_freq_can_be_disabled_and_overridden(self):
        _, opt = self._options(self.MINIMAL + ["--gradient-diagnostic-freq", "0"])
        self.assertEqual(int(opt.gradient_diagnostic_freq), 0)
        _, opt = self._options(self.MINIMAL + ["--gradient-diagnostic-freq", "25"])
        self.assertEqual(int(opt.gradient_diagnostic_freq), 25)

    def test_ray_bias_flags_are_accepted_and_default_unchanged(self):
        _, default_opt = self._options(self.MINIMAL)
        self.assertEqual(default_opt.anchor_ray_sigma0, 0.1)
        self.assertEqual(default_opt.anchor_ray_bias_clamp, -20.0)
        self.assertEqual(default_opt.anchor_ray_bias_init, -6.0)
        _, opt = self._options(
            self.MINIMAL
            + [
                "--anchor-ray-sigma0", "1.0",
                "--anchor-ray-bias-clamp", "-15",
                "--anchor-ray-bias-init", "-8",
            ]
        )
        self.assertEqual(opt.anchor_ray_sigma0, 1.0)
        self.assertEqual(opt.anchor_ray_bias_clamp, -15.0)
        self.assertEqual(opt.anchor_ray_bias_init, -8.0)

    def test_parsed_options_build_a_valid_model_configuration(self):
        _, opt = self._options(self.MINIMAL)
        small = tiny_options()
        tiny = opt.evolve(
            **{
                field: getattr(small, field)
                for field in (
                    "img_size",
                    "patch_size",
                    "dec_patch_size",
                    "enc_depth",
                    "dec_depth",
                    "enc_embed_dim",
                    "enc_num_heads",
                    "num_gs_tokens",
                    "token_dim",
                    "num_object_query_layers",
                )
            }
        )
        model = model_registry[tiny.model_type](tiny)
        self.assertEqual(
            float(model.spatial_decoder.ray_bias_sigma0), tiny.anchor_ray_sigma0
        )
        self.assertTrue(model.spatial_decoder.use_ray_bias)


class GradientDiagnosticReportingTests(unittest.TestCase):
    def test_ddp_audit_key_is_scene_plus_frame_ids(self):
        from scripts.run_ssst_joint import assert_distinct_records

        # Same frame numbering in two different scenes is legitimate.
        keys = assert_distinct_records(
            [
                {"rank": 0, "scene_name": "scene0000_00", "frame_ids": [10, 20, 11, 15]},
                {"rank": 1, "scene_name": "scene0001_00", "frame_ids": [10, 20, 11, 15]},
            ]
        )
        self.assertEqual(
            keys,
            [
                ("scene0000_00", (10, 20, 11, 15)),
                ("scene0001_00", (10, 20, 11, 15)),
            ],
        )
        # The same scene with the same frames is a real duplicate.
        with self.assertRaises(RuntimeError):
            assert_distinct_records(
                [
                    {"rank": 0, "scene_name": "scene0000_00", "frame_ids": [10, 20, 11, 15]},
                    {"rank": 1, "scene_name": "scene0000_00", "frame_ids": [10, 20, 11, 15]},
                ]
            )

    def test_reports_raw_and_effective_understanding_gradients(self):
        opt = tiny_options().evolve(understanding_final_weight=1.0)
        model = model_registry["siu3r_joint_ssst"](opt).train()
        model.gs = FakeRenderer(opt.img_size)
        batch = synthetic_batch(opt)
        report = shared_gradient_diagnostic(model, batch, step=0, phase="train")
        raw = report["grad_understanding_raw_norm"]
        effective = report["grad_understanding_effective_norm"]
        self.assertGreater(raw, 0.0)
        self.assertAlmostEqual(
            effective, report["lambda_understanding"] * raw, places=6
        )
        self.assertAlmostEqual(
            report["grad_understanding_norm"], raw, places=6
        )
        self.assertAlmostEqual(
            report["grad_understanding_to_recon_ratio"],
            effective / report["grad_recon_norm"],
            places=6,
        )
        self.assertGreater(report["grad_understanding_to_recon_ratio"], 0.0)
        # The curriculum weight is the step-0 value and is not modified here.
        self.assertAlmostEqual(report["lambda_understanding"], opt.understanding_start_weight, places=6)


class AssignmentTemperatureTests(unittest.TestCase):
    """softplus(raw) must equal the configured initial temperature."""

    def _head(self, init):
        opt = tiny_options().evolve(assignment_temperature_init=init)
        return model_registry["siu3r_joint_ssst"](opt).object_queries

    def _first_forward_temperature(self, head, opt):
        tokens = torch.randn(1, opt.num_gs_tokens, opt.enc_embed_dim)
        anchors = torch.zeros(1, opt.num_gs_tokens, 3)
        radii = torch.full((1, opt.num_gs_tokens), 0.05)
        with torch.no_grad():
            output = head(tokens, anchors, radii)
        return float(output.stats["assignment_temperature"])

    def test_initial_temperature_matches_configuration(self):
        from tokengs.models.unified_object_queries import inverse_softplus

        for init in (1.0, 5.0, 10.0):
            with self.subTest(init=init):
                opt = tiny_options().evolve(assignment_temperature_init=init)
                head = self._head(init)
                self.assertAlmostEqual(
                    float(F.softplus(head.raw_temperature)), init, places=4
                )
                self.assertAlmostEqual(
                    inverse_softplus(init), float(head.raw_temperature), places=4
                )
                self.assertAlmostEqual(
                    self._first_forward_temperature(head, opt), init, places=3
                )

    def test_default_temperature_is_not_the_old_log_parameterisation(self):
        opt = tiny_options().evolve(assignment_temperature_init=5.0)
        head = self._head(5.0)
        observed = self._first_forward_temperature(head, opt)
        self.assertAlmostEqual(observed, 5.0, places=3)
        # The previous implementation produced softplus(log(5)) = log(6).
        self.assertGreater(abs(observed - math.log(6.0)), 1.0)

    def test_legacy_log_temperature_key_still_loads(self):
        opt = tiny_options().evolve(assignment_temperature_init=5.0)
        model = model_registry["siu3r_joint_ssst"](opt)
        state = model.state_dict()
        legacy = state.pop("object_queries.raw_temperature").clone()
        state["object_queries.log_temperature"] = legacy
        result = model.load_state_dict(state, strict=True)
        self.assertEqual(list(result.missing_keys), [])
        self.assertEqual(list(result.unexpected_keys), [])
        self.assertTrue(torch.equal(model.object_queries.raw_temperature.data, legacy))


class QueryGroupingDiagnosticTests(unittest.TestCase):
    def test_pairwise_cosine_excludes_the_diagonal(self):
        from tokengs.models.ssst_diagnostics import query_pairwise_cosine

        # Two identical directions and one orthogonal one: the six off-diagonal
        # pairs are 1, 0, 1, 0, 0, 0 -> mean 1/3.
        queries = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
        stats = query_pairwise_cosine(queries)
        self.assertAlmostEqual(float(stats["mean"]), 1.0 / 3.0, places=5)
        identical = torch.tensor([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]])
        self.assertAlmostEqual(float(query_pairwise_cosine(identical)["mean"]), 1.0, places=5)
        self.assertTrue(torch.isfinite(stats["p95"]))

    def test_pairwise_mask_similarity_excludes_the_diagonal_and_is_deterministic(self):
        from tokengs.models.ssst_diagnostics import query_mask_pairwise_similarity

        masks = torch.zeros(1, 3, 1, 4, 4)
        masks[0, 0] = 1.0  # query 0 and 1 identical, query 2 disjoint
        masks[0, 1] = 1.0
        first = query_mask_pairwise_similarity(masks, sample_count=16)
        second = query_mask_pairwise_similarity(masks, sample_count=16)
        # Off-diagonal pairs: (0,1)=1, (0,2)=0, (1,0)=1, (1,2)=0, (2,0)=0, (2,1)=0
        for variant in ("cosine", "dice"):
            self.assertAlmostEqual(float(first[variant]["mean"]), 1.0 / 3.0, places=5)
            self.assertAlmostEqual(
                float(first[variant]["p95"]), float(second[variant]["p95"]), places=6
            )
            self.assertTrue(torch.isfinite(first[variant]["mean"]))
            self.assertTrue(torch.isfinite(first[variant]["p95"]))

    def test_mask_cosine_is_shape_only_while_dice_scales_with_magnitude(self):
        from tokengs.models.ssst_diagnostics import query_mask_pairwise_similarity

        pattern = torch.zeros(1, 2, 1, 4, 4)
        pattern[0, 0, 0, :2, :] = 1.0  # two queries with the same shape ...
        pattern[0, 1, 0, :2, :] = 1.0
        tiny = query_mask_pairwise_similarity(pattern * 0.01, sample_count=16)
        large = query_mask_pairwise_similarity(pattern, sample_count=16)
        # ... cosine is exactly 1 for identical shapes at any magnitude ...
        self.assertAlmostEqual(float(tiny["cosine"]["mean"]), 1.0, places=5)
        self.assertAlmostEqual(float(large["cosine"]["mean"]), 1.0, places=5)
        # ... while raw soft Dice scales with the mask magnitude.
        self.assertAlmostEqual(float(large["dice"]["mean"]), 1.0, places=5)
        self.assertLess(float(tiny["dice"]["mean"]), 0.02)

    def test_layer_scale_stats_report_every_query_block(self):
        opt = tiny_options()
        model = model_registry["siu3r_joint_ssst"](opt)
        stats = query_layer_scale_stats(model.object_queries.blocks)
        for layer in range(int(opt.num_object_query_layers)):
            for name in ("cross", "self", "mlp"):
                key = f"query_layer{layer}_{name}_scale_mean"
                self.assertIn(key, stats)
                self.assertAlmostEqual(
                    float(stats[key]), opt.query_block_init_values, places=6
                )

    def test_query_diagnostics_are_detached_and_do_not_change_predictions(self):
        opt = tiny_options()
        model = model_registry["siu3r_joint_ssst"](opt).eval()
        model.gs = FakeRenderer(opt.img_size)
        batch = synthetic_batch(opt)
        tokens = torch.randn(1, opt.num_gs_tokens, opt.enc_embed_dim)
        anchors = torch.zeros(1, opt.num_gs_tokens, 3)
        radii = torch.full((1, opt.num_gs_tokens), 0.05)
        with torch.no_grad():
            first = model.object_queries(tokens, anchors, radii)
            second = model.object_queries(tokens, anchors, radii)
            output_a = model(batch, skip_loss=True)
            output_b = model(batch, skip_loss=True)
        for name in (
            "query_scene_update_norm_mean",
            "query_scene_update_norm_p95",
            "query_pairwise_cosine_mean",
            "query_pairwise_cosine_p95",
        ):
            self.assertTrue(torch.isfinite(first.stats[name]).all(), name)
            self.assertFalse(first.stats[name].requires_grad, name)
        self.assertTrue(torch.equal(first.class_logits, second.class_logits))
        self.assertTrue(torch.equal(first.assignment_logits, second.assignment_logits))
        self.assertTrue(
            torch.equal(output_a["query_class_logits"], output_b["query_class_logits"])
        )
        self.assertTrue(
            torch.equal(output_a["query_mask_prob"], output_b["query_mask_prob"])
        )
        for name in (
            "query_mask_pairwise_cosine_mean",
            "query_mask_pairwise_cosine_p95",
            "query_mask_pairwise_dice_mean",
            "query_mask_pairwise_dice_p95",
        ):
            self.assertIn(name, output_a["query_stats"])
            self.assertTrue(torch.isfinite(output_a["query_stats"][name]).all(), name)
            self.assertFalse(output_a["query_stats"][name].requires_grad, name)
        for name in (
            "assignment_entropy",
            "assignment_temperature",
            "no_object_ratio",
            "token_max_assignment_mean",
            "query_usage_mean",
        ):
            self.assertIn(name, output_a["query_stats"], name)

    def test_grouping_metrics_appear_in_joint_step_metrics(self):
        opt = tiny_options()
        model = model_registry["siu3r_joint_ssst"](opt).train()
        model.gs = FakeRenderer(opt.img_size)
        _, metrics = model.joint_step(synthetic_batch(opt), step=0, phase="train")
        for name in (
            "spatial/query_scene_update_norm_mean",
            "spatial/query_pairwise_cosine_mean",
            "spatial/query_mask_pairwise_cosine_mean",
            "spatial/query_mask_pairwise_dice_mean",
            "spatial/assignment_temperature",
            "spatial/assignment_entropy",
        ):
            self.assertIn(name, metrics, name)
            self.assertTrue(torch.isfinite(metrics[name]).all(), name)


class ReconstructionOnlyTests(unittest.TestCase):
    """Experiment 1: the understanding branch must never run."""

    def _setup(self):
        opt = tiny_options().evolve(reconstruction_only=True)
        model = model_registry["siu3r_joint_ssst"](opt).train()
        model.gs = FakeRenderer(opt.img_size)
        frozen = model.freeze_object_queries()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )
        optimizer_ids = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        return opt, model, frozen, optimizer, optimizer_ids

    def test_reconstruction_only_step_never_calls_the_query_branch(self):
        import tokengs.models.ssst_loss as ssst_loss
        from tokengs.models.siu3r_joint_ssst import SIU3RJointSSST
        from tokengs.models.unified_object_queries import UnifiedObjectQueryHead

        opt, model, _, _, _ = self._setup()
        originals = (
            UnifiedObjectQueryHead.forward,
            SIU3RJointSSST.forward_joint,
            SIU3RJointSSST.render_query_masks,
            ssst_loss.build_context_segments,
            ssst_loss.hungarian_match,
            ssst_loss.class_aware_context_loss,
        )

        def guard(name):
            def _guard(*args, **kwargs):
                raise AssertionError(f"reconstruction-only violated: {name} called")

            return _guard

        try:
            UnifiedObjectQueryHead.forward = guard("query forward")
            SIU3RJointSSST.forward_joint = guard("forward_joint")
            SIU3RJointSSST.render_query_masks = guard("render_query_masks")
            ssst_loss.build_context_segments = guard("build_context_segments")
            ssst_loss.hungarian_match = guard("hungarian_match")
            ssst_loss.class_aware_context_loss = guard("class_aware_context_loss")
            _, metrics = model.joint_step(synthetic_batch(opt), step=0, phase="train")
            metrics["loss"].backward()
        finally:
            (
                UnifiedObjectQueryHead.forward,
                SIU3RJointSSST.forward_joint,
                SIU3RJointSSST.render_query_masks,
                ssst_loss.build_context_segments,
                ssst_loss.hungarian_match,
                ssst_loss.class_aware_context_loss,
            ) = originals

        # L = L_recon + L_spatial, nothing else.
        expected = metrics["loss_recon"] + metrics["loss_spatial"]
        self.assertAlmostEqual(float(metrics["loss"]), float(expected), places=5)
        self.assertNotIn("loss_understanding", metrics)
        self.assertNotIn("lambda_understanding", metrics)
        self.assertFalse(any("understanding" in key for key in metrics))

    def test_query_parameters_are_frozen_and_have_no_gradients(self):
        opt, model, frozen, optimizer, optimizer_ids = self._setup()
        self.assertTrue(frozen)
        query_parameters = dict(model.object_queries.named_parameters())
        for name, parameter in query_parameters.items():
            self.assertFalse(parameter.requires_grad, name)
            self.assertNotIn(id(parameter), optimizer_ids, name)
        _, metrics = model.joint_step(synthetic_batch(opt), step=0, phase="train")
        metrics["loss"].backward()
        for name, parameter in query_parameters.items():
            self.assertIsNone(parameter.grad, name)
        # The reconstruction path does receive gradients.
        self.assertIsNotNone(model.spatial_decoder.anchor_pre.grad)
        self.assertIsNotNone(model.activation_head.deconv.weight.grad)

    def test_joint_forward_is_refused_in_reconstruction_only_mode(self):
        opt = tiny_options().evolve(reconstruction_only=True)
        model = model_registry["siu3r_joint_ssst"](opt).eval()
        model.gs = FakeRenderer(opt.img_size)
        batch = synthetic_batch(opt)
        model_input, _ = split_data(batch, opt)
        with self.assertRaises(RuntimeError):
            model.forward_joint(model_input)

    def test_reconstruction_checkpoint_initializes_joint_model_fresh_queries(self):
        opt, model, _, _, _ = self._setup()
        _, metrics = model.joint_step(synthetic_batch(opt), step=0, phase="train")
        metrics["loss"].backward()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(model.state_dict(), path)
            joint_opt = tiny_options().evolve(reconstruction_only=False)
            torch.manual_seed(7)
            joint = model_registry["siu3r_joint_ssst"](joint_opt)
            query_before = {
                name: parameter.detach().clone()
                for name, parameter in joint.object_queries.named_parameters()
            }
            head_before = joint.activation_head.deconv.weight.detach().clone()
            report = joint.init_from_reconstruction_checkpoint(str(path), log=lambda m: None)
        excluded = [
            entry["key"]
            for entry in report["skipped"]
            if entry.get("reason") == "excluded by policy"
        ]
        self.assertTrue(excluded)
        self.assertTrue(all(key.startswith("object_queries.") for key in excluded))
        for name, parameter in joint.object_queries.named_parameters():
            self.assertTrue(torch.equal(query_before[name], parameter.detach()), name)
        # Channels 3:14 of the Gaussian head are loaded; channels 0:3 (local
        # offset) intentionally keep the target model's own initialization.
        rows = model.activation_head.deconv.weight.shape[0]
        channel = torch.arange(rows) % int(model.activation_head.output_dims)
        loaded_rows = channel >= 3
        self.assertTrue(
            torch.equal(
                joint.activation_head.deconv.weight.detach()[loaded_rows],
                model.activation_head.deconv.weight.detach()[loaded_rows],
            )
        )
        self.assertTrue(
            torch.equal(
                joint.activation_head.deconv.weight.detach()[~loaded_rows],
                head_before[~loaded_rows],
            )
        )
        self.assertTrue(
            torch.equal(
                joint.spatial_decoder.anchor_pre.detach(),
                model.spatial_decoder.anchor_pre.detach(),
            )
        )
        self.assertTrue(
            torch.equal(
                joint.enc_dec_backbone.encoder_norm.weight.detach(),
                model.enc_dec_backbone.encoder_norm.weight.detach(),
            )
        )


if __name__ == "__main__":
    unittest.main()
