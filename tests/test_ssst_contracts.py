"""CPU contract tests for the spatially grounded shared-token joint model."""

from __future__ import annotations

import unittest

import torch

from tokengs.models import model_registry
from tokengs.models.input_types import ModelInputDecoder
from tokengs.models.spatial_grounded_tokens import build_anchor_encoding
from tokengs.models.ssst_contracts import (
    QUERY_COUNT,
    SEMANTIC_CLASS_COUNT,
    TRAIN_TARGET_RECORDS,
    validate_variable_target_batch,
    validate_view_protocol,
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
    return {
        "input": images,
        "images_all": images[:, :, :3],
        "images_input": images[:, : opt.num_input_views, :3],
        "images_output": images[:, opt.num_input_views :, :3],
        "masks_all": torch.ones(batch_size, views, 1, height, width),
        "masks_output": torch.ones(batch_size, views - opt.num_input_views, 1, height, width),
        "has_mask": torch.ones(batch_size, dtype=torch.bool),
        "rays_os": torch.zeros(batch_size, views, 3, height, width),
        "rays_ds": torch.zeros(batch_size, views, 3, height, width),
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

    def test_decoder_is_identity_to_tokengs_at_initialization(self):
        opt = tiny_options()
        model = model_registry["siu3r_joint_ssst"](opt).eval()
        batch = synthetic_batch(opt)
        from tokengs.models.input_types import ModelInput, split_data

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


if __name__ == "__main__":
    unittest.main()
