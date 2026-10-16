"""Checkpoint/initializer audit: frozen weights, controlled forwards, gradient probes without updates."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import numpy as np
import torch
import torch.nn.functional as F
from scripts.train_dual_stream_v1 import build_batch,cohorts,restore_trainable_state,write
from scripts.dual_stream_frozen_cache import install_frozen_caches
from scripts.gfm_instance_diagnostic_math import InitializerTrace,iou_matrix,stage_masks,repeat_calibrated_witness
from scripts.gfm_initialization_audit_tools import InitializationObserver,intervene,candidate_hashes,group_norms,VARIANTS
from scripts.export_object_locus_v3_set_official import assemble_panoptic
from tokengs.models.object_locus_v3_set_loss import final_hungarian,v3_set_losses
PLAN=Path('/space/mawb/ssst/group_plus/gfm_initialization_audit_v1/attempt01/preflight.json')


def values(g,clone=True):
    sem,ins,_=assemble_panoptic(g)
    out={k:g[k] for k in ('gaussians','gaussian_membership','p_class','region_mass')};out['packed']=sem*1000+ins
    return {k:v.detach().clone() for k,v in out.items()} if clone else out


def row_from_prediction(model,g,batch,window,observer,variant,selected=None):
    targets,pairs=final_hungarian(g,batch);gt=targets['gt_pixel_masks'][0];valid=targets['valid_pixels'][0]
    masks,labels,eligible=stage_masks(g)
    matrices={name:iou_matrix(value.permute(1,0,2,3),gt,valid).cpu().tolist() for name,value in masks.items()}
    fm=F.interpolate(g['F_m'].flatten(0,1),size=(256,256),mode='bilinear',align_corners=False)
    def probe(mq):return (torch.einsum('vchw,qc->vqhw',fm,mq[:100]).sigmoid()>=.5).permute(1,0,2,3)
    matrices['initial_2d_probe']=iou_matrix(probe(model.understanding.mask_embedder(g['q_pre'])[0]),gt,valid).cpu().tolist()
    layer_records={}
    for state in g['states']:
        if state['layer'] not in (6,8,10,12):continue
        key=f"L{state['layer']}";matrices[key+'_2d_probe']=iou_matrix(probe(state['m_query'][0]),gt,valid).cpu().tolist()
        layer_records[key]=dict(observer.layers[key],class_probabilities=model.panoptic.classify(state['q'])['p_class'][0].cpu().tolist(),
            seed_query_cosine=F.cosine_similarity(g['q_pre'][0],state['q'][0,:100],dim=-1).cpu().tolist())
    n=g['proposal_initialization'][0]['proposal_queries']
    out=dict(scene=window['scene'],context=window['context'],source_index=window.get('source_index'),variant=variant,
        input_rgb_sha256=hashlib.sha256(batch['images_all'].float().cpu().contiguous().numpy().tobytes()).hexdigest(),
        gt_ids=targets['gt_instance_ids'][0].tolist(),gt_classes=targets['gt_classes'][0].tolist(),gt_pixels=gt.sum((1,2,3)).tolist(),
        proposal_queries=n,mask_IoU=matrices,p_class=g['p_class'][0].cpu().tolist(),eligible=eligible.cpu().tolist(),
        matched_query=pairs[0][0].tolist(),matched_gt=pairs[0][1].tolist(),layers=layer_records,
        initial_centers=g['initial_query_centers'][0].cpu().tolist(),initial_scales=g['initial_query_scales'][0].cpu().tolist(),
        actual_initial_q_class=model.panoptic.classify(g['q_pre'])['p_class'][0].cpu().tolist(),parameter_updates=0)
    if selected is not None:
        out['mask_IoU']['selected_sam']=iou_matrix(selected,gt,valid).cpu().tolist()
        out['candidate_hashes']=candidate_hashes(selected)
        out['initial_components']=observer.initial_stats()
    return out


def observe_window(model,opt,window,destination,first):
    batch=build_batch(opt,window,next(model.query_initializer.parameters()).device)
    baselines=[]
    if first:
        for _ in range(2):
            g=model.generate(batch['images_input']);baselines.append(values(g));del g
    trace=InitializerTrace(model.query_initializer);observer=InitializationObserver(model)
    try:
        g=model.generate(batch['images_input'])
        if first:
            proof=repeat_calibrated_witness(*baselines,values(g,False));write(destination/'observer_witness.json',proof)
            if not proof['passed']:raise RuntimeError('Read-only initializer observer witness failed')
            del baselines
        selected=trace.selected_masks(g);row=row_from_prediction(model,g,batch,window,observer,'native',selected)
        packed=np.packbits(selected.cpu().numpy(),axis=-1)
        np.savez_compressed(destination/'candidate_supports.npz',bits=packed,shape=np.asarray(selected.shape),candidate_hashes=np.asarray(row['candidate_hashes']))
        write(destination/'native.json',row);n=row['proposal_queries']
        seed_hash=row['candidate_hashes'];reconstruction=g['gaussians'].detach().clone();del g,selected,row
        return batch,n,seed_hash,reconstruction,trace,observer
    except BaseException:
        trace.close();observer.close();raise


def gradient_probe(model,opt,window,destination,steps):
    device=next(model.query_initializer.parameters()).device;batch=build_batch(opt,window,device)
    g=model.generate(batch['images_input']);components={}
    under,metrics=v3_set_losses(g,batch,opt,gradient_components=components)
    q=g['q_pre'];logits=g['thing_logits19'];n=g['proposal_initialization'][0]['proposal_queries']
    named=[(name,p) for name,p in model.named_parameters() if name.startswith('query_initializer.') and p.requires_grad]
    losses={'classification':components['classification'],'thing_mask':components['thing_bce']+components['thing_dice'],'all_understanding':under}
    output={};items=list(losses.items())
    for idx,(name,loss) in enumerate(items):
        gradients=torch.autograd.grad(loss,[q,logits]+[p for _,p in named],retain_graph=idx<len(items)-1,allow_unused=True)
        qgrad=gradients[0] if gradients[0] is not None else torch.zeros_like(q)
        norms={key:float(grad.detach().norm()) if grad is not None else None for (key,_),grad in zip(named,gradients[2:])}
        if any(value is not None and not np.isfinite(value) for value in norms.values()):raise FloatingPointError('Nonfinite audit gradient')
        output[name]=dict(value=float(loss.detach()),q_pre=group_norms(qgrad,n),parameter_norms=norms,
            final_no_object_logit_gradient=gradients[1][0,:,18].detach().cpu().tolist() if gradients[1] is not None else None)
    if any(p.grad is not None for p in model.parameters()):raise RuntimeError('autograd.grad probe unexpectedly populated .grad')
    write(destination,dict(scene=window['scene'],context=window['context'],source_index=window.get('source_index'),
        proposal_queries=n,matched_query=g['final_pairs'][0][0].tolist(),matched_gt=g['final_pairs'][0][1].tolist(),
        gt_classes=g['final_targets']['gt_classes'][0].tolist(),p_class=g['p_class'][0].detach().cpu().tolist(),
        understanding_effective_weight=.05*min(1.,(steps+1)/1043),gradients=output,parameter_updates=0,
        historical_gradient=False,gradient_domain='eval-mode current-checkpoint original understanding objective, before global clipping; no optimizer'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=4:raise ValueError('Four independent workers required')
    torch.cuda.set_device(local);torch.set_num_threads(4);device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads(PLAN.read_text());_,_,windows=cohorts()
    if args.smoke:windows=windows[:4]
    from tokengs.options import config_defaults
    from tokengs.models.object_locus_gfm_dual_stream_v4 import LocusGSGFMDualStreamV4
    opt=config_defaults['train_siu3r_object_locus_gfm_dual_stream_v4'];model=LocusGSGFMDualStreamV4(opt,write_strength=0.).to(device)
    cache=install_frozen_caches(model,Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt01/frozen_cache'))
    # Reconstruction values remain fixed; restrict gradient recording to understanding parameters.
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(('understanding.','query_initializer.','panoptic.')))
    model.eval();stages=[plan['sources'][-1]] if args.smoke else plan['sources']
    for source in stages:
        path=Path(source['path']);stat=path.stat()
        if stat.st_size!=source['size'] or stat.st_mtime_ns!=source['mtime_ns']:raise ValueError('Locked checkpoint changed')
        blob=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if blob['progress']['full_updates']!=source['full_updates'] or cache['identity']!=blob['meta']['frozen_cache_identity']:raise ValueError('Checkpoint/assets mismatch')
        restore_trainable_state(model,blob['model']);del blob
        versions={name:p._version for name,p in model.named_parameters()}
        root=args.output/source['stage']/f'rank{rank:02}';root.mkdir(parents=True,exist_ok=False)
        records=[]
        for index in range(rank,len(windows),world):
            window=windows[index];destination=root/str(window['source_index']);destination.mkdir()
            with torch.no_grad():
                batch,n,hashes,reconstruction,trace,observer=observe_window(model,opt,window,destination,index==rank)
                try:
                    if source['stage']=='epoch4':
                        for variant in VARIANTS[1:]:
                            handle=model.query_initializer.register_forward_hook(lambda module,inputs,output,variant=variant:intervene(module,inputs,output,variant))
                            try:
                                g=model.generate(batch['images_input'])
                                if g['proposal_initialization'][0]['proposal_queries']!=n:raise ValueError('Intervention changed candidate count')
                                torch.testing.assert_close(g['gaussians'],reconstruction,rtol=0,atol=0)
                                write(destination/f'{variant}.json',row_from_prediction(model,g,batch,window,observer,variant));del g
                            finally:handle.remove()
                finally:trace.close();observer.close()
                del batch,reconstruction,trace,observer
            records.append(window['source_index']);print(source['stage'],rank,len(records),flush=True)
        training_windows=plan['train_probe_windows'][:4] if args.smoke else plan['train_probe_windows']
        for index in range(rank,len(training_windows),world):
            with torch.enable_grad():gradient_probe(model,opt,training_windows[index],root/f'train_gradient_{index:02}.json',source['full_updates'])
        if any(p._version!=versions[name] for name,p in model.named_parameters()):raise RuntimeError('Audit modified model parameters')
        if any(p.grad is not None for p in model.parameters()):raise RuntimeError('Audit left parameter gradients populated')
        write(root/'COMPLETE.json',dict(records=records,source=source,smoke=args.smoke,parameter_updates=0,parameters_unchanged=True,
            no_optimizer_created=True,training_gradient_windows=list(range(rank,len(training_windows),world))))
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
