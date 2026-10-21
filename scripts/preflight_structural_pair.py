"""Prepare independent A/B trainability runs on the same fixed training cohort."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,json
import torch
from torch import nn
from tokengs.options import config_defaults
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.superpoint_structural_variants import build_structural_model
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST,HELDOUT


class IdentityAsset(nn.Module):
    def __init__(self,identity):super().__init__();self.source_identity=identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(4)
    reference=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt01_esam_many_short_node18_8gpu')
    previous=json.loads((reference/'preflight.json').read_text())
    if json.loads((reference/'RESULT_AUDIT.json').read_text())['progress']['full_updates']!=256:raise ValueError('Original unified validation missing')
    for name in ('omega_identity','sam_identity'):
        if digest(Path(previous[name]['path']))!=previous[name]['sha256']:raise ValueError('Frozen asset changed')
    windows,chosen,heldout=cohorts();assert chosen==previous['short_train_indices']
    options=config_defaults['train_siu3r_unified_query_v1'];config=UnifiedQueryConfig();models={}
    for kind in ('dual_superpoint','unified_direct'):
        with torch.device('meta'):
            model=build_structural_model(kind,options,config,omega=IdentityAsset(previous['omega_identity']),sam=IdentityAsset(previous['sam_identity']),renderer=nn.Identity())
        groups=model.parameter_groups();ids=[id(p) for group in groups.values() for p in group]
        if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:raise ValueError('Trainable role coverage differs')
        models[kind]=dict(architecture=model.architecture_name,parameter_counts={k:sum(p.numel() for p in group) for k,group in groups.items()},
            trainable_tensors=len([p for p in model.parameters() if p.requires_grad]),source_checkpoint=None,optimizer_state_count=0)
    for window in (windows[chosen[0]],heldout[0]):
        batch=build_batch(options,window,'cpu');assert batch['images_input'].shape==(1,2,3,256,256)
        assert torch.isfinite(batch['images_all']).all()
    report=dict(status='PASSED',config=config.to_dict(),models=models,training_windows=[windows[i] for i in chosen],training_indices=chosen,
        heldout_probe_windows=heldout[:4],manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),
        omega_identity=previous['omega_identity'],sam_identity=previous['sam_identity'],assets_hashed=True,
        world_size=4,global_batch=8,short_updates=256,understanding_weight=.05,lr=1e-4,lr_warmup=20,understanding_warmup=20,
        direct_semantic_ce_coefficient_inside_understanding=.1,panoptic='derived from semantic/stuff and instance outputs; no fourth independent prediction head',
        actual_cuda_validation=False,original_unified_short_job=66414)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('training_windows','heldout_probe_windows','omega_identity','sam_identity')},indent=2))


if __name__=='__main__':main()
