"""Fresh serial-query trainability run on the existing fixed 16-scene cohort."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import json
import torch
from torch import nn
from scripts.preflight_structural_pair import IdentityAsset
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
RUN=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt07_serial_query_isolated_short_node18_4gpu')

def main():
    torch.set_num_threads(4)
    if (RUN/'submission.json').exists() or (RUN/'STARTED.json').exists():raise ValueError('Refusing duplicate serial experiment')
    source=RUN.parent/'attempt01_esam_many_short_node18_8gpu'
    previous=json.loads((source/'preflight.json').read_text())
    for name in ['omega_identity','sam_identity']:
        assert digest(Path(previous[name]['path']))==previous[name]['sha256']
    windows,chosen,validation=cohorts();assert chosen==previous['short_train_indices']
    opt=config_defaults['train_siu3r_unified_query_v1'];cfg=UnifiedQueryConfig()
    with torch.device('meta'):
        model=build_structural_model('serial_query',opt,cfg,omega=IdentityAsset(previous['omega_identity']),sam=IdentityAsset(previous['sam_identity']),renderer=nn.Identity())
    groups=model.parameter_groups();ids=[id(p) for ps in groups.values() for p in ps]
    assert not groups['shared'] and len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
    for w in [windows[chosen[0]],validation[0]]:
        batch=build_batch(opt,w,'cpu');assert batch['images_input'].shape==(1,2,3,256,256) and torch.isfinite(batch['images_all']).all()
    report=dict(status='PASSED',architecture=model.architecture_name,config=cfg.to_dict(),reconstruction_depth=12,semantic_depth=4,
        parameter_counts={k:sum(p.numel() for p in ps) for k,ps in groups.items()},trainable_tensors=len(ids),source_checkpoint=None,optimizer_state_count=0,
        training_indices=chosen,training_windows=[windows[i] for i in chosen],heldout_probe_windows=validation[:4],manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),
        omega_identity=previous['omega_identity'],sam_identity=previous['sam_identity'],assets_hashed=True,world_size=4,global_batch=8,short_updates=256,
        understanding_weight=.05,lr=1e-4,lr_warmup=20,understanding_warmup=20,gradient_isolation=True,independent_task_gradient_clip=1.,
        detach_boundaries=['Q_rec seed','Gaussian geometry in child local coordinates','Gaussian geometry/opacity in understanding rasterization'],
        mask_readout='original Unified child-feature/query similarity, no direct-mask field or extra semantic CE',actual_cuda_validation=False,auto_full_training=False)
    RUN.mkdir(parents=True,exist_ok=True);(RUN/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['training_windows','heldout_probe_windows','omega_identity','sam_identity']},indent=2))

if __name__=='__main__':main()
