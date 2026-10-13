"""Parameter-free feedback amplitude and opt-in visual-feature freezing.

Use the original capped message/projection, routes, loss and optimizer kernels.
No new trainable parameters and no changes to either encoder architecture.
"""
from torch import nn
from .object_locus_panoptic_v1_controller import capped_injection

VISUAL_PREFIXES = ('understanding.encoder.', 'understanding.adapter.',
                   'understanding.mask2former.pixel_decoder.')
ARMS = ('joint_full', 'visual_fixed', 'feedback_small', 'visual_fixed_feedback_small')


class ScaledFeedback(nn.Module):
    def __init__(self, peak):
        super().__init__()
        if peak not in (0., .025, .05, .1):
            raise ValueError('unsupported feedback peak')
        self.peak = float(peak)

    def forward(self, h, a, q, mu, c, s, message, projection, exposure):
        delta = capped_injection(h, message, projection, exposure)
        # Exact native operation for the reference, avoiding gratuitous changes.
        return delta if self.peak == .1 else delta * (self.peak / .1)


def set_feedback_peak(model, peak):
    if any(hasattr(layer, 'interaction_feedback') for layer in model.panoptic.layers.values()):
        raise ValueError('feedback already overridden')
    for layer in model.panoptic.layers.values():
        layer.interaction_feedback = ScaledFeedback(peak)
    model.feedback_peak = float(peak)


def install(model, arm, small_peak):
    if arm not in ARMS or small_peak not in (.025, .05):
        raise ValueError('unsupported factorial arm/peak')
    model.feedback_factorial_arm = arm
    peak = small_peak if 'feedback_small' in arm else .1
    set_feedback_peak(model, peak)
    if arm.startswith('visual_fixed'):
        for name, parameter in model.named_parameters():
            if name.startswith(VISUAL_PREFIXES):
                parameter.requires_grad_(False)
        fixed_eval(model)


def fixed_eval(model):
    if getattr(model, 'feedback_factorial_arm', '').startswith('visual_fixed'):
        for module in (model.understanding.encoder, model.understanding.adapter,
                       model.understanding.mask2former.pixel_decoder):
            module.eval()
