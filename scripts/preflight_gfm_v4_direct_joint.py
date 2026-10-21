"""Audit a fresh V4 model, official assets and cohorts without loading trained branches."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.gfm_geometric_transition import template
from scripts.gfm_v4_direct_joint_policy import configure_direct_joint
from scripts.train_dual_stream_v1 import cohorts,optimizer_groups,MANIFEST,HELDOUT,digest
from scripts.dual_stream_training_policy import CONFIG


def main():
    run=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt11_geometric_sam_direct_joint_node13_8gpu')
    assets=[('/datasets2/mengxl/vggt_omega/vggt_omega_1b_512.pt','c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934'),
            ('/space/mawb/ssst_dual_stream_assets/sam_vit_b_01ec64.pth','ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912')]
    verified=[]
    for path,expected in assets:
        sha=hashlib.sha256()
        with open(path,'rb') as stream:
            for block in iter(lambda:stream.read(16*1024*1024),b''):sha.update(block)
        if sha.hexdigest()!=expected:raise ValueError('Official asset identity differs: '+path)
        verified.append(dict(path=path,sha256=expected))
    model=template(4);optimizer=torch.optim.AdamW(optimizer_groups(model))
    if optimizer.state:raise ValueError('Fresh optimizer unexpectedly contains states')
    if not all(p.requires_grad for n,p in model.named_parameters() if not n.startswith(('frozen_omega.','sam_proposals.'))):
        raise ValueError('Fresh trainable branches contain frozen parameters')
    windows,_,validation=cohorts();config=dict(CONFIG);configure_direct_joint(config)
    report=dict(status='PASSED',all_trainable_branches_fresh=True,source_checkpoint=None,
        official_assets=verified,optimizer_state_count=0,adaptation_updates=0,short_training_updates=0,
        new_joint_updates=4172,global_batch=8,world_size=8,train_windows=len(windows),heldout_scenes=len(validation),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),config=config)
    run.mkdir(parents=True,exist_ok=True);(run/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('official_assets','config')},indent=2))


if __name__=='__main__':main()
