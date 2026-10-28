"""Frozen baseline + external MLP, direct IoU ranking, identical segmentation."""
from pathlib import Path
import json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import torch
import torch.distributed as dist
from scripts.frozen_iou_head_core import FrozenTokenIoUHead,SOURCE_SHA,TARGET,context_iou_targets,make_reranked_pair,digest,write_json
from scripts.paired_e0_checks import verify_gt_files,audit_pair,capture_native

REFERENCE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/fixed_query_oracle_diagnostics_20261023_v4_paired/E0/evaluation')


def export(model,opt,windows,root,rank,world,device,*,expected_windows):
    from scripts.train_dual_stream_v1 import build_batch
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import generate_and_render,_official_export_view
    from scripts.export_object_locus_v3_set_official import write_official_pair
    root=Path(root);run=root.parent
    config=json.loads((run/'configuration.json').read_text())
    if digest(config['official_cohort'])!=config['official_cohort_sha256']:raise ValueError('Official cohort identity differs')
    fit=json.loads((run/'head/FIT_COMPLETE.json').read_text());head_path=run/'head/head_best.pt'
    if fit['status']!='HEAD_ONLY_TRAINING_COMPLETE' or digest(head_path)!=fit['selected_head_sha256']:
        raise ValueError('Head fit/identity not verified')
    saved=torch.load(head_path,map_location='cpu',weights_only=False)
    if saved['source_sha256']!=SOURCE_SHA or saved['target_definition']!=TARGET:raise ValueError('IoU head targets/source differ')
    if saved['cache_manifest_sha256']!=digest(run/'cache/export_complete.json'):raise ValueError('Head feature-cache binding differs')
    head=FrozenTokenIoUHead().to(device);head.load_state_dict(saved['head']);head.requires_grad_(False);head.eval()
    model.requires_grad_(False);model.eval();versions={n:p._version for n,p in model.named_parameters()}
    if len(windows)!=1860 or expected_windows!=1860 or len({w['scene'] for w in windows})!=312:raise ValueError('Wrong official evaluation cohort')
    refs={r['name']:r for r in json.loads((REFERENCE/'export_complete.json').read_text())['records']}
    records=[]
    with torch.no_grad():
        for i in range(rank,len(windows),world):
            window=windows[i];batch=build_batch(opt,window,device)
            result=generate_and_render(model,batch['images_input'],batch['images_all'])
            out=_official_export_view(result['render']);q=result['generated']['unified_query'][0]
            predicted=head(q)
            if not torch.isfinite(predicted).all() or any(not torch.isfinite(out[k]).all() for k in ('region_mass','p_class','alpha','images_pred','depths_pred')):
                raise ValueError('Nonfinite head/baseline inference')
            name=window['scene']+'_context'+'_'.join(map(str,window['context']))
            import hashlib
            rgbsha=hashlib.sha256(batch['images_all'].float().cpu().contiguous().numpy().tobytes()).hexdigest()
            if rgbsha!=refs[name]['input_rgb_sha256']:raise ValueError('Official input identity differs')
            native=capture_native(out)
            write_official_pair(out,batch,window,root/'Baseline/evaluation/all',target_frames='all')
            pair=root/'Baseline/evaluation/all'/name
            for sub in ('rgb','rgb_gt','depth','depth_gt'):(pair/sub).symlink_to((REFERENCE/'all'/name/sub).resolve(),target_is_directory=True)
            verify_gt_files(pair,REFERENCE/'all'/name)
            history=audit_pair(pair,REFERENCE/'all'/name,native['packed'],batch['frame_ids'][0].cpu().tolist(),window['scene'])
            reranked=root/'IoU_MLP/evaluation/all'/name
            make_reranked_pair(pair,reranked,predicted)
            # Labels below are diagnostic outputs only, after predictions are
            # committed. They never select a head, score, candidate, or mask.
            truth,keep=context_iou_targets(out,batch['semantic_label_all'][0,:2],batch['instance_label_all'][0,:2])
            write_json(root/'quality_audit'/(name+'.json'),dict(name=name,query_ids=(keep.nonzero().flatten()+1).cpu().tolist(),
                       predicted=predicted[keep].cpu().tolist(),context_gt_iou=truth[keep].cpu().tolist(),historical_baseline_audit=history))
            records.append(dict(name=name,source_index=window['source_index'],input_rgb_sha256=rgbsha,ranking_only_verified=True,
                                masks_hardlinked=True,head_sha256=fit['selected_head_sha256']))
            write_json(root/f'progress_rank{rank:02d}.json',dict(completed=len(records),total=len(windows[rank::world]),backbone_updates=0))
            del batch,result,out,q,predicted,native,truth,keep
    if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in model.named_parameters()):
        raise ValueError('Backbone changed during learned reranking evaluation')
    gathered=[None]*world;dist.all_gather_object(gathered,records)
    if rank==0:
        records=sorted([r for g in gathered for r in g],key=lambda r:r['source_index'])
        if len(records)!=1860 or len({r['name'] for r in records})!=1860:raise ValueError('Incomplete ranking evaluation')
        write_json(root/'export_complete.json',dict(status='FROZEN_IOU_RANKING_EXPORTED',records=records,windows=1860,scenes=312,
                   backbone_updates=0,parameter_updates=0,head_sha256=fit['selected_head_sha256'],prediction_score='predicted_iou_direct',
                   feature_key='unified_query_L12_updated',source_sha256=SOURCE_SHA,masks_classes_candidates_unchanged=True))
    dist.barrier()


if __name__=='__main__':
    from scripts import export_unified_query_checkpoint as loader
    loader.export=export;loader.main()
