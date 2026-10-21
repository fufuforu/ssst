"""CPU state-layout/zero-migration audit; does not run or submit GPU work."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,hashlib,json
from types import SimpleNamespace
import torch
from torch import nn
from tokengs.options import config_defaults
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.reconstruction_fps_superpoint_core import FPSSuperpointDualCore
from scripts.recon_fps_superpoint_initial_state import BANK_KEY,initialize_fps_superpoint_from_legacy_zero

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
source=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_untrained_initial.pt')
sha='3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744';assert digest(source)==sha
saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
cfg=UnifiedQueryConfig(**saved['meta']['config']);torch.manual_seed(42)
model=nn.Module();model.architecture_name='OMEGA_3D_FPS1024_SUPERPOINT_RECON_V1';model.config=cfg
model.frozen_omega=SimpleNamespace(source_identity=saved['meta']['omega_identity'])
model.sam_proposals=SimpleNamespace(source_identity=saved['meta']['sam_identity'])
model.gfm_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256))
model.core=FPSSuperpointDualCore(config_defaults['train_siu3r_unified_query_v1'],cfg)
q_before=model.core.reconstruction_fps.learned_embedding.detach().clone()
evidence=initialize_fps_superpoint_from_legacy_zero(saved,model,
    manifest_sha256=digest(Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json')),
    heldout_sha256=digest(Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json')),world=8)
state=model.state_dict()
for key in evidence['copied_keys']:assert torch.equal(state[key],saved['model'][key]),key
assert torch.equal(state[BANK_KEY],q_before)
parent=Path('/space/mawb/ssst_iouhead_code_20261020_v1')
for name in ['dynamic_panoptic_controller.py','object_locus_panoptic_v1_controller.py','dual_stream_decoder.py','unified_query_loss.py','unified_query_v1.py','dual_superpoint_initialization.py','unified_superpoints.py']:
    assert (ROOT/'tokengs/models'/name).read_bytes()==(parent/'tokengs/models'/name).read_bytes(),name
result=dict(status='CPU_FPS_SUPERPOINT_ZERO_AUDIT_PASSED',source_sha256=sha,baseline_tensors_copied=len(evidence['copied_keys']),
    baseline_tensors_bitwise_equal=True,understanding_code_and_initial_tensors_preserved=True,
    only_new_parameter=BANK_KEY,new_parameter_shape=list(q_before.shape),new_parameter_count=q_before.numel(),
    total_trainable_parameters=sum(p.numel() for p in model.parameters()),frozen_backbones_used_identity_stubs=True,
    gpu_verified=False,jobs_submitted=False)
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
