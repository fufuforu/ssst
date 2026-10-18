"""Fixed-checkpoint prototype/HDBSCAN comparison; never trains or submits jobs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,os,random
import numpy as np
import torch
import torch.distributed as dist
from scripts.runtime_bootstrap import prepare_runtime
from scripts.train_dual_stream_v1 import digest,cohorts,build_batch,MANIFEST,HELDOUT
from scripts.train_unified_query_v1 import restore_state
from scripts.unified_query_validation import parameter_digest
from scripts.fixed_instance_diagnostic_math import evaluate_masks
from tokengs.models.direct_instance_attributes import LocusGSDirectInstanceAttributes
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets
from tokengs.options import config_defaults


def evaluate_readout_outputs(outputs,targets):
    """Identical per-image, all-GT scoring; invoked only after inference."""
    valid=targets['valid_pixels'][0].cpu().numpy();gt=targets['gt_pixel_masks'][0].cpu().numpy()
    metrics={}
    for name,out in outputs.items():
        metrics[name]=[evaluate_masks(out['instance_masks'][0][v].numpy(),gt[:,v],valid[v]) for v in range(2)]
        if name=='prototype':
            metrics['prototype_raw']=[evaluate_masks(out['raw_membership_masks'][0,v].numpy(),gt[:,v],valid[v]) for v in range(2)]
            metrics['prototype_nms']=[evaluate_masks(out['raw_membership_masks'][0,v,out['instance_eligible']].numpy(),gt[:,v],valid[v]) for v in range(2)]
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--readout',choices=('prototype','cluster','both'),default='both')
    parser.add_argument('--evaluate',action='store_true',help='Post-prediction class-agnostic metrics on all valid GT')
    a=parser.parse_args();root=a.output
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world not in (1,2,4,8):raise ValueError('Use 1/2/4/8 inference workers for eight windows')
    prepare_runtime(ROOT);torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl')
    refusal=[bool((root/'STARTED.json').exists() or (root/'COMPLETE.json').exists()) if rank==0 else None]
    if world>1:dist.broadcast_object_list(refusal,src=0)
    if refusal[0]:raise ValueError('Refusing overwrite; use a new output directory')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if digest(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('Checkpoint hash differs')
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True);meta=saved['meta']
    if meta['architecture']!=LocusGSDirectInstanceAttributes.architecture_name:raise ValueError('Requires direct-instance-attribute architecture')
    if meta['manifest_sha256']!=digest(MANIFEST) or meta['heldout_sha256']!=digest(HELDOUT):raise ValueError('Data differs')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    opt=config_defaults['train_siu3r_unified_query_v1']
    model=LocusGSDirectInstanceAttributes(opt,config=UnifiedQueryConfig(**meta['config'])).to(device).eval()
    if any(meta.get(k)!=v for k,v in model.extra_checkpoint_metadata.items()):raise ValueError('Checkpoint recipe differs')
    if meta['omega_identity']!=model.frozen_omega.source_identity or meta['sam_identity']!=(model.sam_proposals.source_identity if model.sam_proposals is not None else None):raise ValueError('Frozen assets differ')
    restore_state(model,saved['model']);before=parameter_digest(model);versions={n:p._version for n,p in model.named_parameters()}
    del saved
    training,chosen,heldout=cohorts();training=meta.get('training_windows') or [training[i] for i in chosen]
    windows=training[:4]+heldout[:4];root.mkdir(parents=True,exist_ok=True)
    if rank==0:(root/'STARTED.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint),sha256=a.checkpoint_sha256,
        readout=a.readout,windows=windows,parameter_updates=0,optimizer_created=False,official128=False),indent=2)+'\n')
    if world>1:dist.barrier()
    rows=[]
    with torch.no_grad():
        for index,window in enumerate(windows):
            if index%world!=rank:continue
            random.seed(42+index);np.random.seed(42+index);torch.manual_seed(42+index);torch.cuda.manual_seed_all(42+index)
            batch=build_batch(opt,window,device);p=model.generate(batch['images_input'])
            cached_probe=a.checkpoint.parent/'probes'/'final'/f'{index:02d}.pt'
            if cached_probe.exists():
                cached=torch.load(cached_probe,map_location='cpu',weights_only=False)
                if cached['scene']!=window['scene'] or not torch.equal(batch['images_input'].cpu(),cached['context_rgb']):raise ValueError('Evaluation differs from training probe input')
            if not all(torch.isfinite(p[k]).all() for k in ('gaussians','gaussian_instance_attributes','query_instance_prototypes','rendered_instance_features','region_mass')):raise ValueError('Nonfinite prediction')
            methods=('prototype','cluster') if a.readout=='both' else (a.readout,)
            outputs={name:model.infer_instances(p,method=name) for name in methods}
            metrics={}
            if a.evaluate:
                metrics=evaluate_readout_outputs(outputs,build_context_instance_targets(batch))
            row=dict(index=index,scope='train' if index<4 else 'heldout',scene=window['scene'],context=window['context'],metrics=metrics,
                readouts=list(outputs),cluster_reports=outputs.get('cluster',{}).get('cluster_reports'))
            (root/f'{index:02d}.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');rows.append(row)
            torch.save(dict(scene=window['scene'],context_rgb=batch['images_input'].cpu(),outputs=outputs,
                rendered_instance_features=p['rendered_instance_features'].cpu(),alpha=p['alpha'].cpu(),
                gaussian_instance_attributes=p['gaussian_instance_attributes'].cpu(),query_instance_prototypes=p['query_instance_prototypes'].cpu()),root/f'{index:02d}.pt')
    after=parameter_digest(model)
    if before!=after or any(p._version!=versions[n] or p.grad is not None for n,p in model.named_parameters()):raise ValueError('Fixed-weight contract failed')
    witness=dict(rank=rank,before=before,after=after,parameter_updates=0,no_parameter_gradients=True,all_parameter_versions_unchanged=True)
    (root/f'FIXED_WEIGHTS_rank{rank:02d}.json').write_text(json.dumps(witness,indent=2)+'\n')
    all_rows=[None]*world;all_checks=[None]*world
    if world>1:dist.all_gather_object(all_rows,rows);dist.all_gather_object(all_checks,witness)
    else:all_rows=[rows];all_checks=[witness]
    if rank==0:
        if len({r['before'] for r in all_checks})!=1:raise ValueError('Workers loaded different weights')
        flattened=sorted([r for group in all_rows for r in group],key=lambda r:r['index'])
        if [r['index'] for r in flattened]!=list(range(len(windows))):raise ValueError('Missing windows')
        summary=[]
        for scope in ('train','heldout'):
            chosen_rows=[r for r in flattened if r['scope']==scope]
            for method in chosen_rows[0]['metrics'] if a.evaluate else ():
                selected=[m for r in chosen_rows for m in r['metrics'][method]]
                ng=sum(r['gt_count'] for r in selected);npred=sum(r['pred_count'] for r in selected);tp=sum(r['tp50'] for r in selected)
                summary.append(dict(scope=scope,method=method,gt_count=ng,pred_count=npred,tp50=tp,
                    one_to_one_gt_iou=sum(sum(r['one_to_one_gt_iou_values']) for r in selected)/ng if ng else None,
                    precision50=tp/npred if npred else 0.,recall50=tp/ng if ng else None))
        report=dict(status='DIRECT_INSTANCE_READOUT_COMPLETE',rows=flattened,summary=summary,fixed_weight_checks=all_checks,
            parameter_updates=0,optimizer_created=False,official128=False,
            limits=['Four train/four heldout windows, per-image all-GT class-agnostic masks; not official mAP/PQ.',
                'Prototype raw/NMS/assembled are separate; cluster has no class assignment or cross-view identity.'])
        (root/'SUMMARY.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (root/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],windows=len(windows),parameter_updates=0))+'\n')
        print(json.dumps(summary,indent=2))
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
