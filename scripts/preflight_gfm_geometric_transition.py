"""Read-only audit of the real adaptation checkpoint before V4 submission."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.gfm_geometric_transition import SOURCE_DIGEST,RECIPE,template,validate_source,plan_transfer
from scripts.dual_stream_training_policy import configure_frozen_reconstruction
from scripts.train_dual_stream_v1 import optimizer_groups


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    sha=hashlib.sha256()
    with args.source.open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b''):sha.update(chunk)
    if sha.hexdigest()!=SOURCE_DIGEST:raise ValueError('Wrong adaptation2 checksum')
    blob=torch.load(args.source,map_location='cpu',weights_only=False,mmap=True)
    target=copy.deepcopy(blob['meta']);configure_frozen_reconstruction(target['config'])
    target['config'].update(recipe=RECIPE,query_pooling='geometric_sam_v4')
    target['config']['required_gradient_families'].append('geometric_query_pooling')
    target.update(recipe=RECIPE,world_size=8,accumulation=1)
    validate_source(blob['meta'],target,blob['progress'])
    report,_=plan_transfer(template(4),template(3),blob,optimizer_groups)
    if report['new_parameters']!=29505:raise ValueError('Unexpected geometric pooling size')
    if len(blob['rank_rng'])!=4:raise ValueError('Expected four source rank RNG states')
    rec=hashlib.sha256()
    for name,value in blob['model'].items():
        if name.startswith(('understanding.','query_initializer.','panoptic.','frozen_omega.','sam_proposals.')):continue
        rec.update(name.encode());rec.update(value.contiguous().numpy().tobytes())
    paired=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt09_frozen_recon_node11_8gpu/FROZEN_RECONSTRUCTION_INITIAL.json')
    if rec.hexdigest()!=json.loads(paired.read_text())['state_sha256']:raise ValueError('Source reconstruction differs from V3 frozen control')
    report.update(status='PASSED',source_sha256=sha.hexdigest(),source_reconstruction_sha256=rec.hexdigest(),
                  identical_reconstruction_to_v3_frozen_control=True,source_adaptation_updates=2086,
                  replayed_adaptation_updates=0,new_understanding_updates=4172,source_rank_rng_count=4,
                  global_batch=8,world_size=8,config=target['config'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('config','new_tensor_names')},indent=2))


if __name__=='__main__':main()
