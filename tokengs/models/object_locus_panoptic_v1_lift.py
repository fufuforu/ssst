"""Occlusion-aware renderer adjoint: first-order gradients to features only."""
import torch


class RendererTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, gaussians, cam_view, intrinsics, renderer):
        g, cam, intr = gaussians.detach(), cam_view.detach(), intrinsics.detach()
        ctx.save_for_backward(g, cam, intr)
        ctx.renderer = renderer
        chunks = []
        with torch.enable_grad():
            for f in features.detach().split(32, dim=2):
                dummy = torch.zeros((*g.shape[:2], f.shape[2]), device=g.device, dtype=torch.float32, requires_grad=True)
                rendered = renderer.render_feature_channels(g, dummy, cam, intr)['images_pred']
                chunks.append(torch.autograd.grad(rendered, dummy, grad_outputs=f.contiguous(), create_graph=False)[0])
        return torch.cat(chunks, -1)

    @staticmethod
    def backward(ctx, grad_output):
        g, cam, intr = ctx.saved_tensors
        chunks = []
        with torch.no_grad():
            for grad in grad_output.split(32, dim=-1):
                chunks.append(ctx.renderer.render_feature_channels(g, grad.contiguous(), cam, intr)['images_pred'])
        return torch.cat(chunks,2), None, None, None, None


def lift_features(features, gaussians, decoder, renderer):
    lifted = RendererTranspose.apply(features, gaussians, decoder.cam_view, decoder.intrinsics, renderer)
    ones = torch.ones_like(features[:,:,:1])
    mass = RendererTranspose.apply(ones, gaussians, decoder.cam_view, decoder.intrinsics, renderer)
    evidence = lifted / mass.clamp_min(1e-6)
    evidence = torch.where(mass > 0, evidence, torch.zeros_like(evidence))
    gate = mass / (mass + 1.0)
    return evidence, gate, mass
