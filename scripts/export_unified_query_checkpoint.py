"""Fixed-weight unified checkpoint export through the existing official protocol."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,json,os
import torch
import torch.distributed as dist
from tokengs.options import config_defaults
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from scripts.train_unified_query_v1 import restore_state
from scripts.train_dual_stream_v1 import cohorts,digest,MANIFEST,HELDOUT
from scripts.dual_stream_official_export import export


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true',help='One-window engineering check, never full128 results')
    args=parser.parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Existing exporter uses collectives even for one rank.
    if not dist.is_initialized():
        if world==1:
            import tempfile
            rendezvous=Path(tempfile.mkdtemp(prefix='unified-export-'))/'rendezvous'
            dist.init_process_group('nccl',rank=0,world_size=1,init_method='file://'+str(rendezvous))
        else:dist.init_process_group('nccl')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True);meta=saved['meta']
    if meta['architecture']!=LocusGSUnifiedQueryV1.architecture_name or meta['manifest_sha256']!=digest(MANIFEST) or meta['heldout_sha256']!=digest(HELDOUT):raise ValueError('Checkpoint/data identity differs')
    if not args.smoke and saved['progress']['epoch']!=meta['epochs']:raise ValueError('Incomplete training; only --smoke export allowed')
    opt=config_defaults['train_siu3r_unified_query_v1'];model=LocusGSUnifiedQueryV1(opt,config=UnifiedQueryConfig(**meta['config'])).to(device)
    if model.frozen_omega.source_identity!=meta['omega_identity']:raise ValueError('Frozen source identity differs')
    if (model.sam_proposals.source_identity if model.sam_proposals is not None else None)!=meta['sam_identity']:raise ValueError('SAM source identity differs')
    restore_state(model,saved['model']);_,_,windows=cohorts()
    if args.smoke:windows=windows[:1]
    export(model,opt,windows,args.output,rank,world,device,expected_windows=len(windows))
    if rank==0:(args.output/'checkpoint_identity.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint.resolve()),meta=meta,parameter_updates=0,smoke_only=args.smoke),indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':main()
