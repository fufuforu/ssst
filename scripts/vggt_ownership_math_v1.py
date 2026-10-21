"""Observation-only ownership math. No model imports or trainable modules."""
import math
import torch
import torch.nn.functional as F


def spatial(x):
    """[anchors, queries] -> query distributions over the same parent anchors."""
    x = x.float().clamp_min(0)
    return x / x.sum(0, keepdim=True).clamp_min(1e-12)


def js_divergence(x, y):
    x, y = spatial(x), spatial(y)
    m = (x + y) / 2
    kl = lambda p: (p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(0)
    return (kl(x) + kl(y)) / (2 * math.log(2))


def dice(x, y, weight):
    w = weight[:, None]
    return (2 * (w * x * y).sum(0)) / (w * (x.square() + y.square())).sum(0).clamp_min(1e-12)


def parent_pool(values, mass, anchors):
    assert values.shape[0] == mass.shape[0] == anchors * 64
    v = values.reshape(anchors, 64, -1)
    w = mass.reshape(anchors, 64, 1).clamp_min(0)
    return (v * w).sum(1) / w.sum(1).clamp_min(1e-12), w.sum(1)[:, 0]


def feedback_attribution(h, q, route, projection, exposure, peak, actual):
    """Exact additive attribution under the *observed* common nonlinear gate.

    delta[a,d] = gate[a,d] sum_q route[a,q] projected_q[q,d].
    Attribution is not a causal query-removal experiment: the gate is held at
    its observed value. Norms are unsigned; signed alignment exposes cancellation.
    """
    assert projection.bias is None
    values = projection(F.layer_norm(q, (q.shape[-1],), eps=1e-5))
    z = route[..., :q.shape[-2]] @ values
    sigma = (h.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    u = z / sigma
    raw = sigma * u.tanh()
    cap = (.25 * h.norm(dim=-1, keepdim=True) / (raw.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1)
    ratio = torch.where(u.abs() > 1e-5, u.tanh() / torch.where(u == 0, torch.ones_like(u), u), 1 - u.square() / 3)
    gate = peak * min(max(float(exposure), 0) / 1000, 1) * cap * ratio
    reconstructed = gate * z
    error = (reconstructed - actual).abs().max()
    assert float(error) <= 1e-6 + 2e-5 * float(actual.abs().max()), float(error)
    # Avoid [anchor,query,1024] materialization while preserving exact norms.
    weights = route[..., :q.shape[-2]]
    norms = weights * (gate.square() @ values.square().transpose(-1, -2)).clamp_min(0).sqrt()
    signed = weights * ((gate * actual) @ values.transpose(-1, -2))
    signed = signed / actual.square().sum(-1, keepdim=True).clamp_min(1e-12)
    return norms, signed, reconstructed, float(error)


def ownership_metrics(read, write, contribution, anchor, child, weight, permutations):
    """No expectation that global-context Read equals localized Write/Child."""
    r, w, c, am, gm = read[:, :100], write[:, :100], contribution[:, :100], anchor[:, :100], child[:, :100]
    result = dict(read_child_js=js_divergence(r, gm), write_child_js=js_divergence(w, gm),
                  contribution_child_js=js_divergence(c, gm), anchor_child_dice=dice(am, gm, weight),
                  read_child_support=(spatial(r) * gm).sum(0),
                  write_child_support=(spatial(w) * gm).sum(0),
                  contribution_child_support=(spatial(c) * gm).sum(0))
    nulls = {key: [] for key in result}
    for p in permutations:
        nulls['read_child_js'].append(js_divergence(r, gm[:, p]))
        nulls['write_child_js'].append(js_divergence(w, gm[:, p]))
        nulls['contribution_child_js'].append(js_divergence(c, gm[:, p]))
        nulls['anchor_child_dice'].append(dice(am, gm[:, p], weight))
        nulls['read_child_support'].append((spatial(r) * gm[:, p]).sum(0))
        nulls['write_child_support'].append((spatial(w) * gm[:, p]).sum(0))
        nulls['contribution_child_support'].append((spatial(c) * gm[:, p]).sum(0))
    for key, values in nulls.items():
        result[key + '_query_shuffle'] = torch.stack(values).mean(0)
    # Independent parent permutation preserves per-query mass and query identity.
    g = torch.Generator().manual_seed(1831)
    p = torch.randperm(len(gm), generator=g).to(gm.device)
    result['write_child_js_anchor_shuffle'] = js_divergence(w, gm[p])
    result['anchor_child_dice_anchor_shuffle'] = dice(am, gm[p], weight)
    return result
