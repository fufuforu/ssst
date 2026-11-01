"""Matched no-grad comparisons with a bounded, measured roundoff allowance.

Frozen parameter hashes and geometry/alpha remain exact. Semantic floating
outputs are checked against repeat-forward variability, with an absolute cap.
"""
import torch
KEYS=('gaussian_membership','p_class','unified_query')

@torch.no_grad()
def capture(model,rgb):
    pred=model.generate(rgb)
    return {k:pred[k].detach().cpu().clone() for k in ('gaussians',)+KEYS}


def verify(reference,repeat,after):
    if not torch.equal(reference['gaussians'][...,:11],repeat['gaussians'][...,:11]):raise ValueError('C control geometry/alpha not deterministic')
    if not torch.equal(reference['gaussians'][...,:11],after['gaussians'][...,:11]):raise ValueError('C geometry/alpha changed')
    if torch.equal(reference['gaussians'][...,11:],after['gaussians'][...,11:]):raise ValueError('C RGB output did not change')
    rows={};passed=True
    for k in KEYS:
        before=reference[k];control=repeat[k];current=after[k]
        if before.shape!=current.shape or before.dtype!=current.dtype:raise ValueError('C semantic shape/dtype changed '+k)
        if not all(torch.isfinite(x).all() for x in (before,control,current)):raise ValueError('C nonfinite protected output '+k)
        noise=float((control-before).abs().max());delta=float((current-before).abs().max())
        # Both repeat and update comparisons use the same inference grad mode.
        # A small fixed floor covers atomic floating accumulation; the hard cap
        # prevents a noisy control from allowing a meaningful semantic change.
        tolerance=min(1e-5,max(1e-6,8*noise))
        ok=noise<=1e-5 and delta<=tolerance
        rows[k]=dict(repeat_max_abs=noise,update_max_abs=delta,allowed_abs=tolerance,bitwise_equal=torch.equal(before,current),passed=ok)
        passed=passed and ok
    return dict(status='PASSED' if passed else 'FAILED',geometry_alpha_bitwise_equal=True,rgb_changed=True,
        semantic_outputs=rows,comparison_mode='same no_grad before/repeat/after',absolute_cap=1e-5)
