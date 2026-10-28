"""Validate geometry-pilot checkpoint, then use unchanged official export."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,json,os
import torch
import torch.distributed as dist
from scripts.membership_geometry_core import *

def main():
    from scripts.train_unified_query_v1 import restore_state
    from scripts.dual_stream_official_export import export
    from tokengs.models.superpoint_structural_variants import build_structural_model
    from tokengs.models.unified_superpoints import UnifiedQueryConfig
    from tokengs.options import config_defaults
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--arm',choices=['G1','G2'],required=True);a=p.parse_args()
    run=a.run_root;rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('Four export ranks required')
    torch.cuda.set_device(local);dist.init_process_group('nccl');device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    training=run/a.arm/'train';receipt=json.loads((training/'TRAINING_COMPLETE.json').read_text());path=training/'checkpoint_latest.pt'
    if receipt['status']!='PASSED' or sha(path)!=receipt['checkpoint_sha256']:raise ValueError('Training receipt differs')
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True);m=saved['meta'];cfg=json.loads((run/'configuration.json').read_text())
    if m.get('pilot_policy')!=POLICY or m['pilot_arm']!=a.arm or m['pilot_source_sha256']!=SOURCE_SHA or m['code_sha']!=cfg['code_sha'] or saved['progress']!=dict(epoch=5,update_in_epoch=0,full_updates=5215):raise ValueError('Pilot checkpoint identity differs')
    renderer=verify_teacher_renderer(cfg)
    write(run/a.arm/f'EXPORT_RUNTIME_rank{rank:02d}.json',renderer)
    opt=config_defaults['train_siu3r_unified_query_v1'];model=build_structural_model('dual_v4init_isolated_noca',opt,UnifiedQueryConfig(**m['config'])).to(device)
    if model.frozen_omega.source_identity!=m['omega_identity']:raise ValueError('Omega source differs')
    restore_state(model,saved['model']);model.requires_grad_(False)
    if sha(cfg['official_cohort'])!=cfg['official_cohort_sha256']:raise ValueError('Official cohort changed')
    windows=json.loads(Path(cfg['official_cohort']).read_text())['records'];root=run/a.arm/'evaluation'
    if len(windows)!=1860 or len({w['scene'] for w in windows})!=312:raise ValueError('Official coverage differs')
    export(model,opt,windows,root,rank,world,device,expected_windows=1860)
    if rank==0:
        baseline=json.loads((G0/'export_complete.json').read_text());current=json.loads((root/'export_complete.json').read_text())
        for old,new in zip(baseline['records'],current['records']):
            if any(old[k]!=new[k] for k in ('name','source_index','frame_ids','input_rgb_sha256')):raise ValueError('G0 input pairing differs')
        old_gt=(G0/'gt_reference_manifest.jsonl').read_text();new_gt=(root/'gt_reference_manifest.jsonl').read_text()
        if old_gt!=new_gt:raise ValueError('G0 GT changed')
        write(root/'PAIRING_AUDIT.json',dict(status='PASSED',windows=1860,scenes=312,input_rgb_equal=True,gt_manifest_byte_equal=True,source_sha256=SOURCE_SHA))
        write(root/'checkpoint_identity.json',dict(meta=m,checkpoint_sha256=receipt['checkpoint_sha256'],progress=saved['progress']))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
