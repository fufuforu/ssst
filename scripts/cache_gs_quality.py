"""Rebuild all four arms' pairing from a single frozen forward per window."""
from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gs_quality_runtime import prepare_pinned_runtime
RUNTIME=prepare_pinned_runtime(ROOT)
import torch
import torch.distributed as dist
from scripts.gs_quality_core import *
from scripts.membership_geometry_core import tensor_sha
from scripts.gs_quality_cache_identity import validate_records

def signature(pred):
    keys=('anchor_point_indices','unified_query','p_class','instance_eligible','region_mass','alpha')
    return {k:tensor_sha(pred[k]) for k in keys}

def export(model,opt,windows,root,rank,world,device,*,expected_windows):
    from scripts.train_dual_stream_v1 import build_batch,cohorts
    from scripts.frozen_iou_head_core import context_iou_targets
    from scripts.export_object_locus_v3_set_official import assemble_panoptic
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _official_export_view
    root=Path(root);run=root.parent;cfg=json.loads((run/'configuration.json').read_text())
    train,_,_=cohorts();official=json.loads(Path(cfg['official_cohort']).read_text())['records']
    if windows!=train+official or expected_windows!=10197 or world!=8 or os.uname().nodename not in cfg['allowed_nodes']:raise ValueError('Cache cohort/world/node differs')
    if cfg['renderer_sha256']!=RUNTIME['sha256'] or digest(run/'cache_cohort.json')!=cfg['cache_cohort_sha256']:raise ValueError('Runtime/cohort identity differs')
    write_json(root/f'RUNTIME_rank{rank:02d}.json',dict(RUNTIME,world_size=world,node=os.uname().nodename,gpu=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device))))
    model.requires_grad_(False);model.eval();versions={n:p._version for n,p in model.named_parameters()}
    chunks=[];records=[];current=[];chunk_index=0
    def flush():
        nonlocal current,chunk_index
        if not current:return
        path=root/f'rank{rank:02d}_chunk{chunk_index:03d}.pt'
        atomic_save(path,dict(records=current,source_sha256=SOURCE_SHA,features=FEATURE_NAMES,policy=POLICY))
        chunks.append(dict(file=path.name,sha256=digest(path),windows=len(current)));current=[];chunk_index+=1
    with torch.no_grad():
        for i in range(rank,len(windows),world):
            window=windows[i];batch=build_batch(opt,window,device);pred=model.generate(batch['images_input'])
            # Inputs and signatures are frozen before accessing GT.
            token=pred['unified_query'][0];geo=geometry_features(pred);sig=signature(pred)
            pclass=pred['p_class'][0]
            keep=pred['instance_eligible']&(pclass.argmax(-1)!=18)&(pclass[:,:18].max(-1).values>=.05)
            ids=(keep.nonzero().flatten()+1).cpu().tolist()
            truth,keep_gt=context_iou_targets(pred,batch['semantic_label_all'][0,:2],batch['instance_label_all'][0,:2])
            if not torch.equal(keep,keep_gt):raise ValueError('Same-forward supervision candidate mismatch')
            _,ins,_=assemble_panoptic(pred)
            diag=view_gt_diagnostics(ins[:2],batch['semantic_label_all'][0,:2],batch['instance_label_all'][0,:2],len(token))[keep]
            name=window['scene']+'_context'+'_'.join(map(str,window['context']))
            row=dict(window_index=i,part='training' if i<8337 else 'official',name=name,scene=window['scene'],query_ids=ids,raw_queries=len(token),
                token=token[keep].detach().float().cpu(),geo=geo[keep].detach().float().cpu(),target=truth[keep].cpu(),view_gt_diagnostics=diag.cpu(),
                p_class=pclass[keep].cpu(),signature=sig,input_rgb_sha256=tensor_sha(batch['images_all']))
            if i>=8337:
                # Persist exact selected float32 masks from this same generated
                # state. Export never regenerates Queries or attaches old IDs.
                calib=model.calibrate_targets(batch['images_all'],pred)
                rendered=model.render_generated_at(pred,torch.linalg.inv(calib['c2w']).transpose(-1,-2),calib['intrinsics'])
                out=_official_export_view(rendered)
                sem_all,ins_all,_=assemble_panoptic(out)
                sem_context,ins_context,_=assemble_panoptic(pred)
                if not torch.equal(ins_all[:2],ins_context) or not torch.equal(sem_all[:2],sem_context):raise ValueError('Same-state Context render changed masks')
                selected=torch.cat((keep.nonzero().flatten(),torch.tensor([len(token),len(token)+1],device=device)))
                packed=dict(selected_channels=selected.cpu(),region_mass=out['region_mass'][:,:,selected].float().cpu(),
                    p_class=out['p_class'].float().cpu(),instance_eligible=out['instance_eligible'].cpu(),alpha=out['alpha'].float().cpu(),
                    raw_semantic=out['semantic_scores'].argmax(2).byte().cpu(),images_pred=rendered['images_pred'].float().cpu(),depths_pred=rendered['depths_pred'].float().cpu(),
                    signature=sig,query_ids=ids,source_sha256=SOURCE_SHA)
                forward_file=f'official/{i:05d}.pt';atomic_save(root/forward_file,packed)
                row['official_forward_file']=forward_file;row['official_forward_sha256']=digest(root/forward_file)
                del rendered,out,packed,calib,sem_all,ins_all,sem_context,ins_context
            if len(ids)!=len(row['token']) or len(set(ids))!=len(ids) or not torch.isfinite(row['target']).all():raise ValueError('Invalid pairing')
            if len(records)==0:
                write_json(root/f'FIRST_SIGNATURE_rank{rank:02d}.json',dict(window=window,signature=sig,current_ids=ids,
                    source_sha256=SOURCE_SHA,historical_pairing_used=False,raw_queries=len(token),renderer=RUNTIME))
            current.append(row);records.append({k:row[k] for k in ('window_index','part','scene','name','query_ids')})
            if len(current)==100:flush()
            del batch,pred,token,geo,row,truth,diag
        flush()
    if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('Frozen model modified')
    write_json(root/f'FROZEN_rank{rank:02d}.json',dict(status='PASSED',versions_unchanged=True,all_grads_none=True,parameter_updates=0))
    gathered=[None]*world;dist.all_gather_object(gathered,dict(records=records,chunks=chunks))
    if rank==0:
        records=sorted([r for g in gathered for r in g['records']],key=lambda r:r['window_index'])
        identity=validate_records(records,windows)
        write_json(root/'export_complete.json',dict(status='GS_QUALITY_CACHE_COMPLETE',source_sha256=SOURCE_SHA,features=FEATURE_NAMES,records=records,
            chunks=[c for g in gathered for c in g['chunks']],training_windows=8337,official_windows=1860,parameter_updates=0,same_forward_pairing=True,renderer=RUNTIME,cohort_identity=identity))
    dist.barrier()

if __name__=='__main__':
    from scripts import export_unified_query_checkpoint as loader
    loader.export=export;loader.main()
