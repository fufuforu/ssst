"""Pinned upstream LocusGS with a ScanNet input/render/loss adapter only."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import sys
import torch
from torch import nn

VENDOR = Path(__file__).resolve().parents[2] / 'third_party/locusgs_official'
# Use the real upstream namespace; never alias it to tokengs.
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

def official_options():
    from locusgs.options import Options
    return Options().evolve(
        cross_attn_variant='geometric_positional', use_anchor_radius=True,
        anchor_radius_refinement=True, anchor_radius_affects_bias=True,
        num_input_views=2, num_views=4, batch_size=1, num_workers=0,
        random_reflect=False, mixed_precision='no', lr=1e-4,
        pct_start_steps=2000, seed=42,
    )

def convert_input(value):
    from locusgs.models.input_types import ModelInput, ModelInputEncoder, ModelInputDecoder
    e, d = value.encoder, value.decoder
    if e.images_rgb.shape[1] != 2 or e.plucker.shape[1] != 2:
        raise ValueError('Only the two context views may enter the official encoder')
    return ModelInput(
        ModelInputEncoder(
            images_rgb=e.images_rgb, plucker=e.plucker,
            rays_os=e.rays_os, rays_ds=e.rays_ds,
            intrinsics_input=e.intrinsics_input, cam_to_world_input=e.cam_to_world_input,
            time_embedding_input=None, time_embedding_target=None,
            images_rgb_unnormalized=e.images_rgb_unnormalized,
        ),
        ModelInputDecoder(time_embedding_target=None, cam_view=d.cam_view, intrinsics=d.intrinsics),
    )

class OfficialLocusGSRecon(nn.Module):
    def __init__(self, opt):
        super().__init__()
        from locusgs.models.locusgs import LocusGS
        from tokengs.rendering.gs import GaussianRenderer
        self.opt = opt
        self.official = LocusGS(official_options())
        self.gs = GaussianRenderer(opt)
        self.img_size = (256, 256)
        self.full_official_config = asdict(self.official.opt)

    def decode(self, model_input):
        value = convert_input(model_input)
        latent = self.official.forward_encoder(value.encoder)
        g, layers, anchors, radii = self.official.forward_decoder(
            latent, value.decoder, return_intermediate_gaussians=True)
        if g.shape != (value.encoder.images_rgb.shape[0], 65536, 14):
            raise RuntimeError(f'Unexpected official Gaussian shape {g.shape}')
        if set(layers) != {6, 12} or set(anchors) != {6, 12}:
            raise RuntimeError('Official decoder must return L6/L12 raw anchors and Gaussians')
        return dict(gaussians=g, gaussians_by_layer=layers, anchors_by_layer=anchors, radii_by_layer=radii)

    def render(self, gaussians, decoder):
        return self.gs.render(gaussians, decoder.cam_view,
            bg_color=gaussians.new_full((3,), .5), intrinsics=decoder.intrinsics)

    def forward_reconstruction_only(self, model_input, render_decoder_input=None):
        output = self.decode(model_input)
        output['render'] = self.render(output['gaussians'], render_decoder_input or model_input.decoder)
        return output

    def step_loss(self, batch, step=0, phase='train'):
        from tokengs.models.input_types import split_data, ModelInput, ModelInputDecoder
        from tokengs.models.canonical_recon import ssim_loss, visibility_loss_from_points, supervised_layer_weights
        value, _ = split_data(batch, self.opt)
        decoder = ModelInputDecoder(cam_view=batch['cam_view_all'], intrinsics=batch['intrinsics_all'])
        output = self.decode(ModelInput(value.encoder, decoder))
        metrics = {}
        total = batch['images_all'].new_zeros(())
        for layer, weight in zip((6, 12), supervised_layer_weights((6, 12))):
            g = output['gaussians_by_layer'][layer]
            render = self.render(g, decoder)
            for name in ('images_pred', 'alphas_pred', 'depths_pred'):
                if not torch.isfinite(render[name]).all():
                    raise FloatingPointError(f'Nonfinite layer {layer} render: {name}')
            if not torch.isfinite(g).all():
                raise FloatingPointError(f'Nonfinite layer {layer} Gaussians')
            pred, gt = render['images_pred'], batch['images_all']
            rgb = (pred - gt).square().mean()
            ssim = ssim_loss(pred.reshape(-1, 3, 256, 256), gt.reshape(-1, 3, 256, 256))
            gvis = visibility_loss_from_points(g[..., :3], decoder.cam_view, decoder.intrinsics,
                self.img_size, clamp_max=1., znear=.025)
            avis = visibility_loss_from_points(output['anchors_by_layer'][layer], decoder.cam_view,
                decoder.intrinsics, self.img_size, clamp_max=1., znear=.025)
            loss = rgb + .2 * ssim + gvis + .1 * avis
            total = total + weight * loss
            for name, v in dict(loss=loss, loss_rgb=rgb, loss_ssim=ssim,
                loss_gaussian_visibility=gvis, loss_anchor_visibility=avis).items():
                metrics[f'{name}_layer{layer}'] = v
            if layer == 12:
                output['render'] = render
                metrics['psnr'] = -10 * rgb.clamp_min(1e-12).log10()
                metrics['alpha_coverage'] = (render['alphas_pred'] > .5).float().mean()
        metrics['loss'] = total
        return output, metrics
