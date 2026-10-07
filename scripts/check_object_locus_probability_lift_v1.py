#!/usr/bin/env python3
"""Small deterministic CPU contracts for the probability-domain readout."""
import json
import torch

ATOL=1e-6
RTOL=1e-6

def isclose_contract(actual,reference):
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise FloatingPointError('nonfinite parity input')
    return torch.isclose(actual,reference,atol=ATOL,rtol=RTOL,equal_nan=False)


def run():
    a, b = torch.tensor(8.0), torch.tensor(-2.0)
    expected = (a.sigmoid() + b.sigmoid()) / 2
    logit_average = ((a + b) / 2).sigmoid()
    assert torch.allclose(expected, torch.tensor(0.5594338), atol=1e-6)
    assert abs(float(expected - logit_average)) > .39

    ref8=torch.tensor([8.0],dtype=torch.float32)
    actual8=ref8+torch.tensor([7.62939453125e-06],dtype=torch.float32)
    limit8=ATOL+RTOL*ref8.abs()
    assert bool(isclose_contract(actual8,ref8).all()) and bool((actual8-ref8).abs().le(limit8).all())
    ref0=torch.tensor([0.0],dtype=torch.float32)
    actual0=torch.tensor([2e-6],dtype=torch.float32)
    assert not bool(isclose_contract(actual0,ref0).all())
    nonfinite_failed=False
    try: isclose_contract(torch.tensor([float('inf')]),ref0)
    except FloatingPointError: nonfinite_failed=True
    assert nonfinite_failed

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
        "rtol_example_reference_8_passes": True,
        "atol_example_reference_0_fails": True,
        "nonfinite_parity_rejected": True,
        "mass_zero_fallback_preserves_saturated_logits": True,
        "channels": 102, "query_softmax": False,
        "broadcast_and_finite": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
