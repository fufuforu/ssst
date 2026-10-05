"""Non-invasive adapter for existing frozen Object-Locus models/renderers."""
from __future__ import annotations

import torch
from pathlib import Path


def freeze_visual_model(model):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    model.requires_grad_(False)
    return model


def load_full1201_frozen(device="cpu", checkpoint_path="/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_full1201_8gpu/checkpoint_epoch_06.pt"):
    """Build the registered Full1201 architecture and strictly load epoch 6."""
    from scripts import object_locus_panoptic_v1_runtime as runtime
    model, opt = runtime.build_model(device, report=False)
    blob = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if int(blob.get("epoch", -1)) != 6:
        raise RuntimeError(f"expected Full1201 epoch 6, got {blob.get('epoch')}")
    expected_sha = "b2624d57ea9ad5a73fd6a8375bafbe4e4c263c9c"
    if blob.get("git_sha") != expected_sha:
        raise RuntimeError(f"checkpoint git SHA mismatch: {blob.get('git_sha')}")
    model.load_state_dict(blob["model"], strict=True)
    del blob
    freeze_visual_model(model)
    return model, opt


def extract_thing_states(output):
    """Read existing L12 outputs without changing the model's forward contract."""
    prediction = output.get("prediction", output)
    states = output.get("states", prediction.get("states"))
    if states is None or not states or int(states[-1].get("layer", -1)) != 12:
        raise RuntimeError("Full1201 output is missing final L12 state")
    q = states[-1]["q"][:, :100].detach()
    P = prediction["gaussian_membership"][:, :, :100].detach()
    return validate_visual_outputs(q, P)


class OutputCapture:
    """Temporary forward hook capture; always remove via close/context manager."""
    def __init__(self, module, transform=lambda x: x):
        self.value = None
        self._hook = module.register_forward_hook(lambda _m, _i, out: setattr(self, "value", transform(out)))
    def close(self): self._hook.remove()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def validate_visual_outputs(q, membership):
    # Full1201 final Object-Locus state carries 100 thing + 2 stuff states.
    if q.ndim != 3 or q.shape[1:] not in ((100, 256), (102, 256)):
        raise ValueError(f"expected q [B,100/102,256], got {tuple(q.shape)}")
    if membership.ndim != 3 or membership.shape[1:] not in ((65536, 100), (65536, 102)):
        raise ValueError(f"expected P [B,65536,100/102], got {tuple(membership.shape)}")
    return q[:, :100].detach(), membership[:, :, :100].detach()


def render_refer_membership(renderer, membership, cameras):
    """Reuse Gaussian feature renderer and V3-Set alpha/coverage normalization."""
    from tokengs.models.object_locus_v3_set import alpha_normalize_membership
    if membership.ndim != 2 or membership.shape[1] != 65536:
        raise ValueError("membership must be [B,65536]")
    gaussians = cameras["gaussians"]
    decoder = cameras["decoder"]
    rendered = renderer.render_feature_channels(gaussians, membership.unsqueeze(-1),
                                                  decoder.cam_view, decoder.intrinsics)
    alpha = rendered["alphas_pred"]
    probability = alpha_normalize_membership(rendered["images_pred"], alpha)
    return probability[:, :, 0], alpha
