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
from scripts.dual_stream_training_policy import epoch_order
from scripts.dual_stream_official_export import export


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true',help='One-window engineering check, never full128 results')
    parser.add_argument('--cohort',type=Path,help='Optional cohort json overriding the fixed 128 validation list')
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
    from tokengs.models.superpoint_structural_variants import MODELS
    constructors={model.architecture_name:model for model in MODELS.values()}
    if meta['architecture'] not in constructors or meta['manifest_sha256']!=digest(MANIFEST) or meta['heldout_sha256']!=digest(HELDOUT):raise ValueError('Checkpoint/data identity differs')
    if meta.get('short_validation') and not args.smoke:raise ValueError('Short trainability run is not full-data training; use --smoke only')
    if not args.smoke and saved['progress']['epoch']!=meta['epochs']:raise ValueError('Incomplete training; only --smoke export allowed')
    training,_,_=cohorts()
    if not args.smoke and saved['progress']['full_updates']!=meta['epochs']*(len(epoch_order(len(training),0))//8):raise ValueError('Training update count differs')
    constructor_kwargs={}
    if meta['architecture']=='OMEGA_3D_SUPERPOINT_RECON_QP_V1':
        if 'recon_qp_slots' not in meta or meta['recon_qp_slots']<1:raise ValueError('Missing reconstruction Q bank identity')
        constructor_kwargs['recon_qp_slots']=meta['recon_qp_slots']
    opt=config_defaults['train_siu3r_unified_query_v1'];model=constructors[meta['architecture']](opt,config=UnifiedQueryConfig(**meta['config']),**constructor_kwargs).to(device)
    if model.frozen_omega.source_identity!=meta['omega_identity']:raise ValueError('Frozen source identity differs')
    if (model.sam_proposals.source_identity if model.sam_proposals is not None else None)!=meta['sam_identity']:raise ValueError('SAM source identity differs')
    restore_state(model,saved['model']);_,_,windows=cohorts()
    if args.cohort is not None:
        windows=json.loads(args.cohort.read_text())['records']
    if args.smoke:windows=windows[:1]
    export(model,opt,windows,args.output,rank,world,device,expected_windows=len(windows))
    if rank==0:
        record=json.loads((args.output/'export_complete.json').read_text())
        record['panoptic_assembly']='Unified Query: context-only class-aware Mask NMS IoU>'+str(model.config.mask_nms_iou)+' then existing candidate thresholds/competition/area filter'
        (args.output/'export_complete.json').write_text(json.dumps(record,indent=2)+'\n')
        (args.output/'checkpoint_identity.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint.resolve()),meta=meta,parameter_updates=0,smoke_only=args.smoke),indent=2)+'\n')
    dist.destroy_process_group()


if __name__=='__main__':main()
