"""Focused gradient checks for the V2.1 mask-conditioned readout."""
import torch

from tokengs.models.object_locus_v2_1_controller import ObjectLocusV2_1Controller
from tokengs.models.object_locus_v2_1_loss import _category_objectness_losses


def test_category_pool_and_objectness_have_finite_nonzero_gradients():
    torch.manual_seed(17)
    controller = ObjectLocusV2_1Controller(token_dim=32)
    q = torch.randn(1, 102, 256, requires_grad=True)
    features = torch.randn(1, 100, 256, requires_grad=True)
    readout = controller.classify(q, features)
    readout["category_logits18"].retain_grad()
    pairs = [(torch.tensor([1, 7]), torch.tensor([0, 1]))]
    targets = {"gt_classes": [torch.tensor([2, 5])]}
    cat, obj, *_ = _category_objectness_losses(readout, pairs, targets)
    (cat + obj).backward()
    for tensor in (features.grad, q.grad, controller.cls_fuse.weight.grad,
                   controller.category_head.weight.grad,
                   controller.objectness_head.weight.grad):
        assert tensor is not None
        assert torch.isfinite(tensor).all()
        assert torch.count_nonzero(tensor) > 0


def test_unmatched_category_logits_have_no_category_ce_gradient():
    torch.manual_seed(18)
    controller = ObjectLocusV2_1Controller(token_dim=32)
    q = torch.randn(1, 102, 256)
    features = torch.randn(1, 100, 256)
    readout = controller.classify(q, features)
    readout["category_logits18"].retain_grad()
    pairs = [(torch.tensor([3]), torch.tensor([0]))]
    targets = {"gt_classes": [torch.tensor([4])]}
    category, *_ = _category_objectness_losses(readout, pairs, targets)
    category.backward()
    grad = readout["category_logits18"].grad
    assert grad is not None
    assert torch.count_nonzero(grad[0, 3]) > 0
    unmatched = torch.ones(100, dtype=torch.bool)
    unmatched[3] = False
    assert torch.count_nonzero(grad[0, unmatched]) == 0


def test_conditional_category_softmax_does_not_depend_on_objectness_logit():
    controller = ObjectLocusV2_1Controller(token_dim=32)
    q = torch.randn(1, 102, 256)
    features = torch.randn(1, 100, 256)
    out = controller.classify(q, features)
    out["conditional_class_prob"].sum().backward()
    assert controller.objectness_head.weight.grad is None


def test_anchor_and_gaussian_pooling_preserve_feature_membership_gradients():
    torch.manual_seed(19)
    controller = ObjectLocusV2_1Controller(token_dim=32)
    f = torch.randn(1, 24, 256, requires_grad=True)
    anchor_logits = torch.randn(1, 24, 102, requires_grad=True)
    am = torch.sigmoid(anchor_logits)
    ua, _ = controller.pool_anchor_features(f, am)
    ua.sum().backward(retain_graph=True)
    assert f.grad is not None and torch.isfinite(f.grad).all()
    assert torch.count_nonzero(f.grad) > 0
    assert anchor_logits.grad is not None and torch.isfinite(anchor_logits.grad).all()
    assert torch.count_nonzero(anchor_logits.grad) > 0

    gf = torch.randn(1, 32, 256, requires_grad=True)
    gaussian_logits = torch.randn(1, 32, 102, requires_grad=True)
    gm = torch.sigmoid(gaussian_logits)
    gs = torch.rand(1, 32, 14, requires_grad=True)
    fallback = torch.zeros(1, 100, 256)
    ug, _ = controller.pool_gaussian_features(gf, gm, gs, fallback)
    ug.sum().backward()
    assert gf.grad is not None and torch.isfinite(gf.grad).all()
    assert torch.count_nonzero(gf.grad) > 0
    assert gaussian_logits.grad is not None and torch.isfinite(gaussian_logits.grad).all()
    assert torch.count_nonzero(gaussian_logits.grad) > 0
    # Gaussian geometry/opacity is detached only from the pooling-weight path.
    assert gs.grad is None
