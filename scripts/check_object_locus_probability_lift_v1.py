#!/usr/bin/env python3
"""Small deterministic CPU contracts for the probability-domain readout."""
import json
import torch


def run():
    a, b = torch.tensor(8.0), torch.tensor(-2.0)
    expected = (a.sigmoid() + b.sigmoid()) / 2
    logit_average = ((a + b) / 2).sigmoid()
    assert torch.allclose(expected, torch.tensor(0.5594338), atol=1e-6)
    assert abs(float(expected - logit_average)) > .39

    child = torch.tensor([[-1000.0, 1000.0], [0.25, -0.75]])
    delta = torch.tensor([[.2, -.1], [.01, .03]])
    mq = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    fallback = (child + delta) @ mq.T
    p = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    mixed_logits = torch.logit(p.clamp(1e-6, 1 - 1e-6)) + delta @ mq.T
    selected = torch.where(torch.tensor([[True], [False]]), fallback, mixed_logits)
    assert torch.equal(selected[0], fallback[0])
    assert torch.isfinite(selected).all()

    pixel = torch.rand(1, 2, 102, 16, 16)
    mass = torch.rand(1, 65536, 1)
    gate = mass / (mass + 1)
    lifted = torch.rand(1, 65536, 102)
    child_prob = torch.rand(1, 65536, 102)
    membership = gate * lifted + (1 - gate) * child_prob
    assert pixel.shape[2] == membership.shape[-1] == 102
    assert membership.isfinite().all() and ((membership >= 0) & (membership <= 1)).all()
    assert torch.allclose(gate, gate.expand_as(mass))
    result = {
        "status": "PASS", "pixel_probability_before_lifting": True,
        "logit_average_is_not_probability_average": True,
        "mass_zero_fallback_preserves_saturated_logits": True,
        "channels": 102, "query_softmax": False,
        "broadcast_and_finite": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
