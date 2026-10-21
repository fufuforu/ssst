"""Equal-capacity frozen-feature ridge probes; no change to production losses."""
import torch
import torch.nn.functional as F

WIDTH = 256
RIDGE = .001


def design(features):
    assert features.shape[-1] == WIDTH and torch.isfinite(features).all()
    x = F.layer_norm(features.detach().float(), (WIDTH,))
    return torch.cat((x, torch.ones_like(x[..., :1])), -1)


def targets(batch):
    labels = batch['class_target'].long()
    assert labels.ndim == 2 and labels.min() >= 0 and labels.max() < 19
    wc = torch.where(labels == 18, .1, 1.)
    wc = wc / wc.sum(-1, keepdim=True)
    wq = batch['quality_valid'].float()
    wq = wq / wq.sum(-1, keepdim=True).clamp_min(1)
    return wc, F.one_hot(labels, 19).float(), wq, batch['quality_target'].float()[..., None]


def new_stats(device):
    d = WIDTH + 1
    return dict(class_gram=torch.zeros(d, d, dtype=torch.float64, device=device),
        class_rhs=torch.zeros(d, 19, dtype=torch.float64, device=device),
        quality_gram=torch.zeros(d, d, dtype=torch.float64, device=device),
        quality_rhs=torch.zeros(d, 1, dtype=torch.float64, device=device), windows=0, quality_windows=0)


def accumulate(stats, features, batch):
    x = design(features).flatten(0, 1)
    wc, yc, wq, yq = targets(batch)
    for name, weights, y in [('class', wc, yc), ('quality', wq, yq)]:
        w = weights.flatten()[:, None]
        xw = x * w.sqrt()
        # FP32 products, FP64 accumulation/solve. Avoid a very slow3090 FP64
        # matrix product without hiding numerical stability of the final solve.
        stats[name + '_gram'] += (xw.T @ xw).double()
        stats[name + '_rhs'] += (x.T @ (w * y.flatten(0, 1))).double()
    stats['windows'] += len(features)
    stats['quality_windows'] += int(batch['quality_valid'].any(-1).sum())


def solve(stats):
    answers = {}
    for name in ('class', 'quality'):
        gram = stats[name + '_gram']
        regularizer = torch.eye(len(gram), dtype=torch.float64, device=gram.device) * (RIDGE * stats['windows'])
        regularizer[-1, -1] = 0  # unpenalized bias
        if name == 'quality' and stats['quality_windows'] == 0:
            answers[name] = torch.zeros_like(stats[name + '_rhs']).float()
            continue
        answers[name] = torch.linalg.solve(gram + regularizer, stats[name + '_rhs']).float()
        assert torch.isfinite(answers[name]).all()
    return answers


def pool_image(features, masks):
    """Same predicted context support, view-major alignment, no GT pooling."""
    b, v, d, h, w = features.shape
    assert v == 2 and masks.shape[:3] == (b, 2, 100)
    weight = F.interpolate(masks.flatten(0, 1), size=(h, w), mode='area')
    weight = weight.reshape(b, 2, 100, h, w).permute(0, 2, 1, 3, 4).reshape(b, 100, -1)
    feature = features.permute(0, 1, 3, 4, 2).reshape(b, -1, d)
    return (weight @ feature.float()) / weight.sum(-1, keepdim=True).clamp_min(1e-6)
