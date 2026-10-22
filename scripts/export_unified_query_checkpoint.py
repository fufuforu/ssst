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
    training_identity={}
    if meta['architecture'] in ('OMEGA_POINT1024_SOURCE256_FMQUERY_ISOLATED_RESIDUAL_ADAPTER_V2','OMEGA_POINT1024_FMQUERY_ISOLATED_NO_ANCHOR_CA_GEOMETRIC_ROUTING_V1','OMEGA_POINT1024_FMQUERY_ISOLATED_ANCHOR_CA_NO_FM_CA_V1','OMEGA_POINT1024_FMQUERY_ISOLATED_NO_ANCHOR_CA_NO_FM_CA_GEOMETRIC_ROUTING_V1'):
        from scripts.training_windows_2ctx4novel import load_training_windows,DEFAULT_MANIFEST
        training,training_identity=load_training_windows(meta.get('training_window_manifest',DEFAULT_MANIFEST),training,digest(MANIFEST))
        if any(meta.get(k)!=v for k,v in training_identity.items()):raise ValueError('2+4 training manifest/protocol identity differs')
    if not args.smoke and saved['progress']['full_updates']!=meta['epochs']*(len(epoch_order(len(training),0))//8):raise ValueError('Training update count differs')
    if meta['architecture']=='OMEGA_3D_FPS1024_SUPERPOINT_RECON_V1':
        if meta.get('reconstruction_anchor_count')!=1024 or meta.get('reconstruction_initialization')!='fps_xyz_plus_learned_embedding_plus_superpoint':raise ValueError('FPS initialization identity differs')
    if meta['architecture'] in ('OMEGA_POINT1024_SOURCE256_FMQUERY_ISOLATED_RESIDUAL_ADAPTER_V2','OMEGA_POINT1024_FMQUERY_ISOLATED_NO_ANCHOR_CA_GEOMETRIC_ROUTING_V1','OMEGA_POINT1024_FMQUERY_ISOLATED_ANCHOR_CA_NO_FM_CA_V1','OMEGA_POINT1024_FMQUERY_ISOLATED_NO_ANCHOR_CA_NO_FM_CA_GEOMETRIC_ROUTING_V1'):
        if meta.get('reconstruction_anchor_count')!=1024 or meta.get('seed_feature_source')!='exact_source_pixel_shared_projection_256' or meta.get('understanding_seed_feature_source')!='semantic_memory_fm_256' or meta.get('gradient_isolation') is not True or meta.get('adapter_hidden')!=64 or meta.get('adapter_read_layers')!=[6,8,10,12] or meta.get('supervised_layers')!=[6,12] or meta.get('layer_weights')!=[1/3,2/3] or meta.get('epochs')!=4 or meta.get('global_batch')!=8:raise ValueError('Isolated point1024 checkpoint identity differs')
    if meta['architecture']=='OMEGA_POINT1024_FMQUERY_ISOLATED_NO_ANCHOR_CA_GEOMETRIC_ROUTING_V1':
        if meta.get('object_anchor_attention')!='none' or meta.get('geometry_routing')!='normalized_geometric_distance_softmax_v1':raise ValueError('No-CA routing identity differs')
    from scripts.point1024_isoadapter_initial_state import NOFM_ARCHS,C_ARCH
    if meta['architecture'] in NOFM_ARCHS:
        if meta.get('object_fm_attention')!='none' or meta.get('object_anchor_attention')!=('learned_cross_attention' if meta['architecture']==C_ARCH else 'none') or meta.get('geometry_routing')!=('learned_attention_weights' if meta['architecture']==C_ARCH else 'normalized_geometric_distance_softmax_v1'):raise ValueError('C/D attention identity differs')
    from scripts.olda4_ablation_initial_state import MODEL_KEYS,recipe_identity
    if meta['architecture'] in MODEL_KEYS:
        if any(meta.get(k)!=v for k,v in recipe_identity(MODEL_KEYS[meta['architecture']]).items()):raise ValueError('Old A4 ablation recipe identity differs')
    from scripts.v4init_isolated_recipe import ARCH as V4ISO_ARCH,recipe_identity as v4iso_recipe
    from scripts.v4init_isolated_recipe import MODEL_KEYS as V4ISO_MODELS,VIEWS_ARCH
    if meta['architecture'] in V4ISO_MODELS:
        expected=v4iso_recipe(V4ISO_MODELS[meta['architecture']])
        if meta.get('continuation_policy'):
            from scripts.v4init_extend8_policy import POLICY,START,END
            if meta['continuation_policy']!=POLICY or meta['epochs']!=8 or saved['progress']!=dict(epoch=8,update_in_epoch=0,full_updates=END) or meta['continuation_start_updates']!=START:raise ValueError('Incomplete/invalid V4 continuation export')
            expected['geometry_refinement_lr']=3e-6
        if any(meta.get(k)!=v for k,v in expected.items()) or meta['learning_rate']!=(3e-5 if meta.get('continuation_policy') else 1e-4):raise ValueError('V4 isolated recipe differs')
    if meta['architecture']==VIEWS_ARCH:
        from scripts.training_windows_2ctx4novel import load_training_windows,DEFAULT_MANIFEST
        training,training_identity=load_training_windows(meta.get('training_window_manifest',DEFAULT_MANIFEST),training,digest(MANIFEST))
        if any(meta.get(k)!=v for k,v in training_identity.items()):raise ValueError('V4 six-view manifest identity differs')
    constructor_kwargs={}
    if meta['architecture']=='OMEGA_3D_SUPERPOINT_RECON_QP_V1':
        if 'recon_qp_slots' not in meta or meta['recon_qp_slots']<1:raise ValueError('Missing reconstruction Q bank identity')
        constructor_kwargs['recon_qp_slots']=meta['recon_qp_slots']
    opt=config_defaults['train_siu3r_unified_query_v1']
    if training_identity:opt=opt.evolve(num_views=6)
    model=constructors[meta['architecture']](opt,config=UnifiedQueryConfig(**meta['config']),**constructor_kwargs).to(device)
    if meta['architecture']=='OMEGA_3D_FPS1024_SUPERPOINT_RECON_V1':
        from tokengs.models.canonical_recon import supervised_layer_weights
        layers=tuple(meta['supervised_layers'])
        if not layers or layers[-1]!=12 or tuple(sorted(set(layers)))!=layers or any(x<1 or x>12 for x in layers):raise ValueError('Invalid saved supervision layers')
        weights=supervised_layer_weights(layers)
        if list(weights)!=meta['layer_weights']:raise ValueError('Saved supervision weights differ')
        model.supervised_layers=layers;model.layer_weights=weights
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
