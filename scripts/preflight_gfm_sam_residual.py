"""Validate residual initialization against the exact common zero-step source."""
import copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.gfm_geometric_transition import template
from tokengs.models.geometric_sam_initialization import ResidualGeometricSAMQueryInitializer
from scripts.gfm_sam_residual_policy import configure,validate_initial,restore_shared_initial
from scripts.train_dual_stream_v1 import optimizer_groups,MANIFEST,HELDOUT,digest
from scripts.isolated_gradient_routing import parameter_groups
RUN=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt16_learned_base_sam_residual_isolated_node13_8gpu')
SOURCE=RUN.parent/'attempt11_geometric_sam_direct_joint_node13_8gpu/checkpoint_initial.pt'


def main():
    blob=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True)
    target=copy.deepcopy(blob['meta']);configure(target['config']);target['recipe']=target['config']['recipe']
    validate_initial(blob['meta'],target,blob)
    if target['manifest_sha256']!=digest(MANIFEST) or target['heldout_sha256']!=digest(HELDOUT):raise ValueError('Data changed')
    model=template(4)
    with torch.device('meta'):model.query_initializer=ResidualGeometricSAMQueryInitializer()
    no_decay={n for n,p in model.named_parameters() if getattr(p,'_no_weight_decay',False)}
    model=model.to_empty(device='cpu')
    for n,p in model.named_parameters():
        if n in no_decay:p._no_weight_decay=True
    optimizer=torch.optim.AdamW(optimizer_groups(model),betas=(.9,.95),eps=1e-8)
    transfer=restore_shared_initial(model,optimizer,blob,optimizer_groups)
    state=model.state_dict()
    if not all(torch.equal(state[k],v) for k,v in blob['model'].items()):raise ValueError('Shared values not exactly restored')
    if abs(float(model.query_initializer.residual_gate_logit.sigmoid())-.1)>1e-7:raise ValueError('Gate initialization differs')
    rec,under=parameter_groups(model)
    report=dict(status='PASSED',source_sha256=digest(SOURCE),source_training_updates=0,source_optimizer_states=0,
        all_shared_initial_weights_and_rank_rng_reused=True,shared_values_checked_on_real_cpu_tensors=True,transfer=transfer,world_size=8,global_batch=8,
        reconstruction_parameters=sum(p.numel() for p in rec),understanding_parameters=sum(p.numel() for p in under),
        config=target['config'])
    RUN.mkdir(parents=True,exist_ok=True);(RUN/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    (RUN/'untrained_initial.sha256').write_text(report['source_sha256']+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='config'},indent=2))


if __name__=='__main__':main()
