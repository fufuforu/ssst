"""CPU contracts for beta scaling/freezing; real GPU proofs are separate."""
import json
from pathlib import Path
import sys
import torch
from torch import nn

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from tokengs.models.object_locus_feedback_factorial_v1 import ScaledFeedback,install,VISUAL_PREFIXES
from tokengs.models.object_locus_panoptic_v1_controller import capped_injection
from scripts.train_vggt_understanding_probes_v1 import expected_counts

torch.set_num_threads(4);torch.manual_seed(314159)
h=torch.randn(1,5,1024,requires_grad=True);message=torch.randn(1,5,256,requires_grad=True)
projection=nn.Linear(256,1024,bias=False)
for exposure in (0,8,152,1000,4000):
    reference=capped_injection(h,message,projection,exposure)
    args=(h,None,None,None,None,None,message,projection,exposure)
    assert torch.equal(ScaledFeedback(.1)(*args),reference)
    for peak in (0.,.025,.05):
        result=ScaledFeedback(peak)(*args)
        assert torch.equal(result,reference*(peak/.1))
        assert torch.isfinite(result).all()
        if peak==0:assert torch.equal(h+result,h)
        gr=torch.autograd.grad(reference.sum(),message,retain_graph=True)[0]
        gs=torch.autograd.grad(result.sum(),message,retain_graph=True)[0]
        assert torch.allclose(gs,gr*(peak/.1),atol=1e-6,rtol=1e-5)

class Probe(nn.Module):
    def __init__(self):
        super().__init__();self.understanding=nn.Module()
        self.understanding.encoder=nn.Linear(2,2);self.understanding.adapter=nn.Linear(2,2)
        self.understanding.mask2former=nn.Module();self.understanding.mask2former.pixel_decoder=nn.Linear(2,2)
        self.understanding.mask2former.transformer_module=nn.Linear(2,2)
        self.panoptic=nn.Module();self.panoptic.layers=nn.ModuleDict({'L6':nn.Linear(2,2)})
        self.anchor_decoder=nn.Linear(2,2);self.vggt_memory_adapter=nn.Linear(2,2)
for arm in ('joint_full','visual_fixed','feedback_small','visual_fixed_feedback_small'):
    model=Probe();before={n:p.detach().clone() for n,p in model.named_parameters()};rng=torch.get_rng_state()
    install(model,arm,.025)
    assert torch.equal(torch.get_rng_state(),rng)
    assert set(before)==set(dict(model.named_parameters()))
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
    assert all(p.requires_grad==(not(arm.startswith('visual_fixed') and n.startswith(VISUAL_PREFIXES)))
               for n,p in model.named_parameters())
counts=expected_counts(500);assert counts.sum()==4000 and counts.sum(0).max()==1
print(json.dumps(dict(status='CPU_FEEDBACK_FACTORIAL_CONTRACTS_PASSED',native_beta_point1_exact=True,
    scaled_forward_and_gradient=True,no_added_parameters_or_rng_changes=True,visual_only_freeze=True,
    matched_unique4000_sampling=True,not_a_real_model_or_gpu_proof=True),indent=2))
