"""Fixed-weight final heldout128 export, or isolated one-window protocol smoke."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import argparse
import json
import os
from datetime import timedelta
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
import torch
import torch.distributed as dist
from scripts.train_dual_stream_v1 import cohorts,digest,MANIFEST,HELDOUT,restore_trainable_state,write
from scripts.dual_stream_frozen_cache import install_frozen_caches
from scripts.dual_stream_official_export import export


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=args.checkpoint or args.run_root/'checkpoint_latest.pt'
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    meta=blob['meta'];progress=blob['progress']
    if meta['manifest_sha256']!=digest(MANIFEST) or meta['heldout_sha256']!=digest(HELDOUT):raise ValueError('Evaluation data identity differs')
    if not args.smoke:
        training=json.loads((args.run_root/'TRAINING_COMPLETE.json').read_text())
        if training['status']!='FULL_FOUR_EPOCHS_COMPLETE' or progress['epoch']!=4 or progress['full_updates']!=4172:raise ValueError('Final joint4 checkpoint incomplete')
        if meta['config'].get('adaptation_epochs',0) and progress.get('adapt_epoch')!=2:raise ValueError('Adaptation incomplete')
    from tokengs.options import config_defaults
    from tokengs.models.object_locus_gfm_dual_stream_v3 import LocusGSGFMDualStreamV3
    opt=config_defaults['train_siu3r_object_locus_gfm_dual_stream_v3']
    model=LocusGSGFMDualStreamV3(opt,write_strength=0.).to(torch.device('cuda',local))
    restore_trainable_state(model,blob['model'])
    cache=install_frozen_caches(model,Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt01/frozen_cache'))
    if cache['identity']!=meta['frozen_cache_identity']:raise ValueError('Frozen asset identity changed')
    del blob
    _,_,windows=cohorts()
    root=args.run_root/('protocol_smoke' if args.smoke else 'evaluation')
    if args.smoke:windows=windows[:1]
    export(model,opt,windows,root,rank,world,torch.device('cuda',local),expected_windows=len(windows))
    if rank==0:write(root/'checkpoint_identity.json',dict(checkpoint=str(checkpoint.resolve()),source_meta=meta,
        short_updates=progress['short_updates'],full_updates=progress['full_updates'],adaptation_updates=progress.get('adapt_updates',0),parameter_updates=0,smoke_only=args.smoke))
    dist.destroy_process_group()

if __name__=='__main__':main()
