"""Fixed-weight V4 instance tracing on heldout128; four independent GPU shards."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import torch
import torch.nn.functional as F
from scripts.train_dual_stream_v1 import cohorts,build_batch,restore_trainable_state,digest,MANIFEST,HELDOUT,write
from scripts.dual_stream_frozen_cache import install_frozen_caches
from scripts.eval_object_locus_frozen_vggt_posefree_v1 import generate_and_render
from scripts.export_object_locus_v3_set_official import assemble_panoptic
from scripts.gfm_instance_diagnostic_math import InitializerTrace,iou_matrix,stage_masks,repeat_calibrated_witness
from scripts.vggt_ownership_math_v1 import parent_pool,js_divergence
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets,final_hungarian
from tokengs.models.object_locus_panoptic_v1 import alpha_normalize_membership
BASE=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1')
ARMS={'isolated':BASE/'attempt12_geometric_sam_isolated_node13_8gpu','weight005':BASE/'attempt13_geometric_sam_weight005_node18_8gpu'}


def matrix_record(matrix):
    return matrix.detach().float().cpu().tolist()


def render_membership(model,g,result,membership):
    camera=result['camera_calibration'];cam=torch.linalg.inv(camera['c2w'][:,:2]).transpose(-1,-2)
    rr=model.gs.render_feature_channels(g['gaussians'],membership,cam,camera['intrinsics'][:,:2])
    regions=alpha_normalize_membership(rr['images_pred'],rr['alphas_pred'])
    return ((regions[0,:,:100]>=.5)&(rr['alphas_pred'][0,:2]>.05)).permute(1,0,2,3)


def golden_agreement(render,batch,window,run):
    sem,ins,_=assemble_panoptic(render);name=window['scene']+'_context'+'_'.join(map(str,window['context']))
    packed=torch.where(sem<=19,sem+1,torch.zeros_like(sem))*1000+ins
    differences=[]
    for vi,frame in enumerate(batch['frame_ids'][0].tolist()):
        path=run/'evaluation/all'/name/'target_seg_pred'/f"{window['scene']}_pred{frame}.png"
        rgb=np.asarray(Image.open(path),dtype=np.int64);previous=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2]
        differences.append(float((packed[vi].cpu().numpy()!=previous).mean()))
    return differences


def figure(path,batch,gt,stages,selected,matrices):
    if not len(gt):return
    best=matrices['raw'].max(0).values;final=matrices['final'].max(0).values
    k=int((best-final).argmax());q=int(matrices['raw'][:,k].argmax())
    p=int(matrices['selected_sam'][:,k].argmax()) if len(selected) else None
    rows=[]
    for vi in range(2):
        rgb=(batch['images_input'][0,vi].permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8)
        masks=[gt[k,vi],selected[p,vi] if p is not None else torch.zeros_like(gt[k,vi]),stages['raw'][vi,q],stages['eligible'][vi,q],stages['final'][vi,q]]
        panels=[]
        for title,mask in zip(['GT','selected SAM support','raw rendered query','after eligibility','final query'],masks):
            arr=rgb.copy();m=mask.cpu().numpy();arr[m]=(.45*arr[m]+.55*np.array([255,50,30])).astype(np.uint8)
            im=Image.fromarray(arr);ImageDraw.Draw(im).text((4,4),title,fill='white',stroke_width=1,stroke_fill='black');panels.append(np.asarray(im))
        rows.append(np.concatenate(panels,1))
    Image.fromarray(np.concatenate(rows,0)).save(path)


def observe(model,opt,trace,window,run,destination,device,witness):
    batch=build_batch(opt,window,device)
    # Measure unobserved native repeat variability before testing observer effects.
    def witness_values(output,clone=True):
        values={k:output['generated'][k] for k in ('gaussians','gaussian_membership','p_class')}
        values['region_mass']=output['render']['region_mass']
        sem,ins,_=assemble_panoptic(output['render']);values['packed']=sem*1000+ins
        return {k:v.clone() for k,v in values.items()} if clone else values
    baselines=[];witness_details=None
    if witness:
        trace.close()
        for _ in range(2):
            old=generate_and_render(model,batch['images_input'],batch['images_all'])
            baselines.append(witness_values(old));del old
        trace.__init__(model.query_initializer)
    result=generate_and_render(model,batch['images_input'],batch['images_all']);g=result['generated'];render=result['render']
    if baselines:
        witness_details=repeat_calibrated_witness(*baselines,witness_values(result,False))
        write(destination/f"{window['source_index']}_witness.json",witness_details)
        if not witness_details['passed']:raise RuntimeError('Observer differs beyond bounded native repeat variability; see witness receipt')
        del baselines
    targets,pairs=final_hungarian(g,batch);gt=targets['gt_pixel_masks'][0];valid=targets['valid_pixels'][0];classes=targets['gt_classes'][0]
    stages,labels,eligible=stage_masks(render)
    matrices={name:iou_matrix(mask[:2].permute(1,0,2,3),gt,valid) for name,mask in stages.items()}
    selected=trace.selected_masks(g);matrices['selected_sam']=iou_matrix(selected,gt,valid)
    initial_mq=model.understanding.mask_embedder(g['q_pre'])[0]
    fm=F.interpolate(g['F_m'].flatten(0,1),size=(256,256),mode='bilinear',align_corners=False)
    def image_masks(mq):return (torch.einsum('vchw,qc->vqhw',fm,mq[:100]).sigmoid()>=.5).permute(1,0,2,3)
    matrices['initial_query_2d_projection']=iou_matrix(image_masks(initial_mq),gt,valid)
    layer_info={};anchors=g['states'][-1]['anchor_membership'].shape[1]
    child,_=parent_pool(g['gaussian_membership'][0],g['lifting_mass'][0,:,0].clamp_min(0),anchors)
    generator=torch.Generator().manual_seed(1831+window['source_index']);permutations=[]
    for _ in range(16):
        permutation=torch.randperm(100,generator=generator)
        while (permutation==torch.arange(100)).any():permutation=torch.randperm(100,generator=generator)
        permutations.append(permutation.to(device))
    for state in g['states']:
        if state['layer'] not in (6,8,10,12):continue
        layer=state['layer'];matrices[f'L{layer}_query_2d_projection']=iou_matrix(image_masks(state['m_query'][0]),gt,valid)
        read=state['evidence_attention'][0].mean(0).T[:,:100];am=state['anchor_membership'][0,:,:100];route=state['route'][0,:,:100]
        info={}
        for name,value in [('read',read),('anchor',am),('route',route)]:
            info[name+'_child_js']=matrix_record(js_divergence(value,child[:,:100]))
            info[name+'_child_js_query_shuffle']=matrix_record(torch.stack([js_divergence(value,child[:,p]) for p in permutations]).mean(0))
        layer_info[str(layer)]=info
    matrices['inherited_anchor_render']=iou_matrix(render_membership(model,g,result,g['anchor_membership'].repeat_interleave(64,1)),gt,valid)
    state=g['states'][-1]
    child_features,_=model.panoptic.gaussian_child_features(state['anchor_embedding'],state['f_anchor'],g['gaussians'],state['mu'],state['radii'])
    geometry_membership=(child_features@state['m_query'].transpose(1,2)).sigmoid()
    matrices['geometry_child_render']=iou_matrix(render_membership(model,g,result,geometry_membership),gt,valid)
    error=(render['images_pred'][0].clamp(0,1)-batch['images_all'][0]).square().mean(1)
    object_mask=(batch['semantic_label_all'][0]>=2)&(batch['semantic_label_all'][0]<=19)&(batch['instance_label_all'][0]>0)
    row=dict(source_index=window['source_index'],scene=window['scene'],context=window['context'],frame_ids=batch['frame_ids'][0].tolist(),
        input_rgb_sha256=hashlib.sha256(batch['images_all'].float().cpu().contiguous().numpy().tobytes()).hexdigest(),
        gt_ids=targets['gt_instance_ids'][0].tolist(),gt_classes=classes.tolist(),gt_pixels=gt.sum((1,2,3)).tolist(),
        gt_view_pixels=gt.sum((2,3)).tolist(),raw_sam_best_IoU_per_view=matrix_record(trace.raw_proposal_best_per_view(gt,valid)),
        initial_proposal_counts=g['proposal_initialization'],mask_IoU={k:matrix_record(v) for k,v in matrices.items()},
        query_class_probabilities=matrix_record(render['p_class'][0]),query_labels=labels.tolist(),eligible=eligible.tolist(),
        matched_query=pairs[0][0].tolist(),matched_gt=pairs[0][1].tolist(),ownership=layer_info,
        object_region_rgb=dict(pixels=int(object_mask.sum()),sse=float(error[object_mask].sum())),
        official_packed_mismatch_fraction=golden_agreement(render,batch,window,run),observer_noop_witness=witness,observer_witness_details=witness_details,
        actual_write_strength=float(model.write_strength),parameter_updates=0,GT_used_only_after_native_prediction=True)
    if row['input_rgb_sha256']!=window['expected_rgb_sha256']:raise ValueError('Input RGB differs')
    if not all(torch.isfinite(g[k]).all() for k in ('gaussians','gaussian_membership','p_class')):raise FloatingPointError('Nonfinite prediction')
    figure(destination/f"{window['source_index']}.png",batch,gt,stages,selected,matrices)
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4:raise ValueError('Exactly four independent GPU workers required')
    torch.cuda.set_device(local);torch.set_num_threads(4);device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    _,_,windows=cohorts()
    if args.smoke:windows=windows[:4]
    args.output.mkdir(parents=True,exist_ok=True)
    for arm,run in ARMS.items():
        out=args.output/arm/f'rank{rank:02}';out.mkdir(parents=True,exist_ok=False)
        from tokengs.options import config_defaults
        from tokengs.models.object_locus_gfm_dual_stream_v4 import LocusGSGFMDualStreamV4
        opt=config_defaults['train_siu3r_object_locus_gfm_dual_stream_v4'];model=LocusGSGFMDualStreamV4(opt,write_strength=0.).to(device)
        blob=torch.load(run/'checkpoint_epoch_04.pt',map_location='cpu',weights_only=False,mmap=True)
        meta=blob['meta'];progress=blob['progress']
        if meta['manifest_sha256']!=digest(MANIFEST) or meta['heldout_sha256']!=digest(HELDOUT) or progress['full_updates']!=4172:raise ValueError('Model/cohort mismatch')
        restore_trainable_state(model,blob['model']);cache=install_frozen_caches(model,BASE/'attempt01/frozen_cache')
        if cache['identity']!=meta['frozen_cache_identity']:raise ValueError('Asset identity mismatch')
        model.eval();versions={n:p._version for n,p in model.named_parameters()};del blob
        trace=InitializerTrace(model.query_initializer);records=[]
        with torch.no_grad():
            for index in range(rank,len(windows),world):
                record=observe(model,opt,trace,windows[index],run,out,device,index==rank)
                write(out/f"{record['source_index']}.json",record);records.append(record['source_index'])
                print(arm,rank,len(records),flush=True)
        if any(p._version!=versions[n] for n,p in model.named_parameters()):raise RuntimeError('Diagnostic modified model parameters')
        trace.close();write(out/'COMPLETE.json',dict(records=records,parameter_updates=0,parameter_versions_unchanged=True,checkpoint=str(run/'checkpoint_epoch_04.pt'),meta=meta,smoke=args.smoke))
        del trace,model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
