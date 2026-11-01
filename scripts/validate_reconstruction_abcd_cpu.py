"""Full tensor/layout migration on CPU plus small mathematical contracts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import gc,json,hashlib,os
import torch
from torch import nn
from scripts.reconstruction_abcd_policy import *
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from scripts.train_dual_stream_v1 import MANIFEST,HELDOUT,digest
from tokengs.rendering.sh1 import sh1_rgb
from tokengs.models.reconstruction_abcd import activate_geometry,activate_appearance

BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
SOURCES={'ABC':BASE/'v4init_isolated_extend8_recovery_20261023_v2_jsonfix/Baseline/eval/source_checkpoint.pt','D':BASE/'v4init_isolated_noca_v4recipe_20261022_v1/train/checkpoint_untrained_initial.pt'}
class Frozen(nn.Module):
    def __init__(self,identity):super().__init__();self.source_identity=identity

def main():
    torch.set_num_threads(4);results={}
    for arm in 'ABCD':
        path=SOURCES['D' if arm=='D' else 'ABC'];s=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if arm in 'AD':
            sha=digest(path)
            if sha!=(ZERO_SHA if arm=='D' else B8_SHA):raise ValueError('Wrong source SHA')
        assert s['meta']['manifest_sha256']==digest(MANIFEST) and s['meta']['heldout_sha256']==digest(HELDOUT)
        torch.manual_seed(42)
        model=build_structural_model('reconstruction_'+arm.lower(),config_defaults['train_siu3r_unified_query_v1'],UnifiedQueryConfig(**s['meta']['config']),omega=Frozen(s['meta']['omega_identity']),sam=Frozen(s['meta']['sam_identity']),renderer=object())
        proof=initialize(model,s,arm);opt,ap=optimizer(model,s,arm)
        if arm!='D':
            assert len(ap['copied_adam_names'])==604
            assert (len(task_state(model))==632) if arm in 'AB' else len(task_state(model))>632
        else:assert not opt.state
        covered=[id(p) for ps in model.parameter_groups().values() for p in ps]
        assert len(covered)==len(set(covered)) and set(covered)=={id(p) for p in model.parameters() if p.requires_grad}
        # Real RGB/ray-memory contract, no target inputs accepted.
        if arm in 'CD':
            with torch.no_grad():
                rgb=torch.rand(1,2,3,256,256)
                keys,values=model.core.appearance.memory(rgb,dict(c2w=torch.eye(4).expand(1,2,4,4),k=torch.tensor([[256.,0,128.],[0,256.,128.],[0,0,1.]]).expand(1,2,3,3)))
                assert keys.shape==values.shape==(1,16,512,64)
                if arm=='C':
                    delta=model.core.appearance(torch.randn(1,1024,1024),(keys,values))
                    assert torch.count_nonzero(delta)==0
        results[arm]=dict(status='PASSED',model_tensors=len(task_state(model)),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),task_parameters=sum(p.numel() for p in model.parameters()),copied_keys=len(proof['copied_keys']),new_keys=len(proof['new_keys']),expanded_keys=proof['expanded_keys'],adam=ap)
        print(arm,'PASSED',results[arm]['trainable_parameters'],flush=True)
        del model,opt,s,proof,ap;gc.collect()
    # SH1 must depend on view and differentiate all degree-one coefficients.
    sh=torch.zeros(1,4,3,requires_grad=True);sh.data[:,3,0]=.1
    x=sh1_rgb(sh,torch.tensor([[1.,0,0]]));y=sh1_rgb(sh,torch.tensor([[-1.,0,0]]))
    assert x[0,0]!=y[0,0];x.sum().backward();assert sh.grad[:,3].abs().sum()>0
    position,scale,rotation=activate_geometry(torch.zeros(1,64,10))
    assert torch.equal(position[...,2],torch.ones(1,64)) and torch.all(scale>0)
    color,alpha=activate_appearance(torch.zeros(1,64,13));assert color.shape==(1,64,12) and torch.allclose(alpha,torch.full_like(alpha,torch.sigmoid(torch.tensor(-2.))))
    out=Path(sys.argv[1]);out.write_text(json.dumps(dict(status='PASSED',variants=results,sh1_view_dependence_and_gradient=True,geometry_z_offset_once=True,gpu_validation='PENDING_SLURM_GATE'),indent=2)+'\n')

if __name__=='__main__':main()
