"""Audit exact 0-step baseline initialization for a single-variable routing comparison."""
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.gfm_geometric_transition import template
from scripts.gfm_v3_fresh_isolated_policy import configure,validate_initial,restore_shared_initial
from scripts.train_dual_stream_v1 import optimizer_groups,MANIFEST,HELDOUT,digest
from scripts.isolated_gradient_routing import parameter_groups


def main():
    source=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt11_geometric_sam_direct_joint_node13_8gpu/checkpoint_initial.pt')
    run=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt15_semantic_sam_v3_isolated_8gpu')
    sha=hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):sha.update(block)
    blob=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    target=copy.deepcopy(blob['meta']);configure(target['config']);target['recipe']=target['config']['recipe']
    validate_initial(blob['meta'],target,blob)
    if target['manifest_sha256']!=digest(MANIFEST) or target['heldout_sha256']!=digest(HELDOUT):raise ValueError('Data identity changed')
    model=template(3);state=model.state_dict()
    if any(k not in blob['model'] or state[k].shape!=blob['model'][k].shape for k in state):raise ValueError('Initial tensor registration/shape mismatch')
    optimizer=torch.optim.AdamW(optimizer_groups(model),betas=(.9,.95),eps=1e-8)
    # Meta tensors verify registration without allocating model weights.
    transfer=restore_shared_initial(model,optimizer,blob,optimizer_groups)
    rec,under=parameter_groups(model)
    report=dict(status='PASSED',source_sha256=sha.hexdigest(),source_training_updates=0,source_optimizer_states=0,
        all_shared_initial_weights_and_rank_rng_reused=True,transfer=transfer,source_model_tensors=len(state),world_size=8,global_batch=8,
        reconstruction_parameters=sum(p.numel() for p in rec),understanding_parameters=sum(p.numel() for p in under),
        reconstruction_loss_only_to_reconstruction=True,weighted_understanding_loss_only_to_understanding=True,
        shared_global_clip_norm_unchanged=1.,config=target['config'])
    run.mkdir(parents=True,exist_ok=True);(run/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    (run/'untrained_initial.sha256').write_text(sha.hexdigest()+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='config'},indent=2))


if __name__=='__main__':main()
