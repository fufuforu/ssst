"""Fixed-weight native mask tracing; no optimizer, backward, or model edits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import argparse,gc,hashlib,json,os,random
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image,ImageDraw
from tokengs.options import config_defaults
from tokengs.models.superpoint_structural_variants import MODELS
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.unified_query_loss import support_relation
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets
from tokengs.models.object_locus_v3_set import alpha_normalize_membership
from tokengs.models.unified_query_nms import context_mask_nms
from scripts.train_unified_query_v1 import restore_state
from scripts.train_dual_stream_v1 import build_batch,digest,MANIFEST,HELDOUT
from scripts.export_object_locus_v3_set_official import assemble_panoptic
from scripts.gfm_instance_diagnostic_math import repeat_calibrated_witness
from scripts.serial_query_diagnostic_math import original_readout,analyze_native,representation_stats,distribution
from tokengs.models.object_locus_panoptic_v1_lift import RendererTranspose


def witness_values(p):
    sem,ins,_=assemble_panoptic(p)
    return {**{k:p[k].detach().clone() for k in ['gaussians','gaussian_membership','p_class','region_mass']},'packed':sem*1000+ins}


class Observer:
    """Capture the unchanged encode_context return; never register layer hooks."""
    def __init__(self,model):
        self.model=model;self.memory=None
        self.had_override='encode_context' in model.__dict__
        self.previous_override=model.__dict__.get('encode_context')
        self.original=model.encode_context
        def capture(*args,**kwargs):
            output=self.original(*args,**kwargs)
            self.memory=output[2]
            return output
        model.encode_context=capture

    def close(self):
        if self.had_override:self.model.encode_context=self.previous_override
        else:delattr(self.model,'encode_context')

    def replay(self,prediction):
        q=prediction['semantic_query_initial'];seed=q.clone()
        memory=self.memory['semantic'];before=memory.clone();stages={}
        for i,block in enumerate(self.model.core.semantic_blocks):
            q=block(q,memory);stages[f'D{i+1}']=q.detach()
        torch.testing.assert_close(q,prediction['understanding_query'],rtol=1e-5,atol=1e-6)
        if not torch.equal(memory,before) or not torch.equal(seed,prediction['semantic_query_initial']):
            raise ValueError('Single-forward replay changed its inputs')
        return stages


def stage_prediction(model,p,q):
    field=original_readout(model.core,q,p);pc=field['thing_logits19'].softmax(-1)
    views=torch.linalg.inv(p['predicted_context_c2w']).transpose(-1,-2)
    r=model.gs.render_feature_channels(p['gaussians'],field['gaussian_membership'],views,p['predicted_context_intrinsics'])
    regions=alpha_normalize_membership(r['images_pred'],r['alphas_pred']);n=p['num_thing_queries']
    semantic=regions.new_zeros((1,2,20,*regions.shape[-2:]));semantic[:,:,:2]=regions[:,:,n:n+2]
    semantic[:,:,2:]=torch.einsum('bvqhw,bqc->bvchw',regions[:,:,:n],pc[...,:18]);semantic=semantic/(semantic.sum(2,keepdim=True)+1e-6)
    out=dict(p,**field,p_class=pc,region_mass=regions,pixel_membership=regions,alpha=r['alphas_pred'],semantic_scores=semantic)
    out['instance_eligible'],out['instance_ranking_scores']=context_mask_nms(pc,regions,r['alphas_pred'],model.config.mask_nms_iou)
    return out


def case_figure(path,batch,p,targets,allowed,matrices,stage_maps,fixed_gt=None):
    gt=targets['gt_pixel_masks'][0];n=p['num_thing_queries']
    if not len(gt):return None
    scores=matrices['raw_alpha'].masked_fill(~allowed,0);best=scores.max(0).values
    supported=allowed.any(0);choices=supported.nonzero().flatten()
    k=fixed_gt if fixed_gt is not None else (int(choices[best[choices].argmin()]) if len(choices) else int(gt.flatten(1).sum(1).argmax()))
    eligible=allowed[:,k].nonzero().flatten();q=int(eligible[scores[eligible,k].argmax()]) if len(eligible) else int(matrices['raw_alpha'][:,k].argmax())
    def heat(value):
        value=value.clamp(0,1).cpu().numpy();return np.stack((value,1-np.abs(2*value-1),1-value),-1)*255
    rows=[]
    for v in range(2):
        rgb=(batch['images_input'][0,v].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
        gt_image=rgb.copy();mask=gt[k,v].cpu().numpy();gt_image[mask]=(.4*gt_image[mask]+.6*np.array([255,50,30])).astype(np.uint8)
        final=rgb.copy();_,ids,_=assemble_panoptic(p);chosen=(ids[v]==q+1).cpu().numpy();final[chosen]=(.4*final[chosen]+.6*np.array([0,255,60])).astype(np.uint8)
        images=[('RGB',rgb),('GT '+str(int(targets['gt_instance_ids'][0][k])),gt_image),('alpha 0..1',heat(p['alpha'][0,v,0])),('native p(q) 0..1',heat(p['region_mass'][0,v,q])),('final chosen query',final)]
        for name,value in stage_maps.items():images.append((name+' p(q)',heat(value[0,v,q])))
        panels=[]
        for title,arr in images:
            im=Image.fromarray(arr.astype(np.uint8));ImageDraw.Draw(im).text((4,4),title,fill='white',stroke_width=1,stroke_fill='black');panels.append(np.asarray(im))
        rows.append(np.concatenate(panels,1))
    Image.fromarray(np.concatenate(rows,0)).save(path)
    return dict(gt_index=k,query=q,selection='serial worst supported GT; baseline reuses that GT; fixed selected query across counterfactual stages',probability_color_scale='blue=0, green=.5, red=1')


@torch.no_grad()
def observe(model,opt,window,index,scope,device,serial,do_witness,folder,reference_folder=None):
    batch=build_batch(opt,window,device);references=[]
    if do_witness:
        for _ in range(2):
            native=model.generate(batch['images_input']);references.append(witness_values(native));del native
    observer=Observer(model) if serial else None
    try:p=model.generate(batch['images_input'])
    finally:
        if observer:observer.close()
    calibrated=None
    if do_witness:
        try:calibrated=repeat_calibrated_witness(*references,witness_values(p))
        except ValueError as error:calibrated=dict(passed=False,error=str(error),policy='native_repeat_shape_check')
        # Full front-end/point-pooling repeats are a stability diagnostic, not an
        # observer identity test. Preserve the original limits and full receipt.
        (folder/f'WITNESS_{index:02d}.json').write_text(json.dumps(calibrated,indent=2)+'\n')
    captured_stages=observer.replay(p) if observer else {}
    if observer:(folder/f'REPLAY_{index:02d}.json').write_text(json.dumps(dict(passed=True,policy='same_native_seed_and_memory_no_layer_hooks',rtol=1e-5,atol=1e-6))+'\n')
    del references
    targets=build_context_instance_targets(batch)
    allowed,support=support_relation(p,batch,targets,model.config.support_purity,return_scores=True)
    result,labels,stages,matrices=analyze_native(p,targets,allowed,support,seed=1729+index)
    result.update(index=index,scope=scope,scene=window['scene'],context=window['context'],input_rgb_sha256=hashlib.sha256(batch['images_input'].cpu().contiguous().numpy().tobytes()).hexdigest(),witness=calibrated,native_repeat_warning=bool(calibrated and not calibrated['passed']),same_forward_replay_verified=bool(observer),
        gt_ids=targets['gt_instance_ids'][0].cpu().tolist(),gt_classes=targets['gt_classes'][0].cpu().tolist())
    # Existing renderer adjoint measures actual per-Gaussian alpha-compositing
    # contribution. Its internal derivative is to a temporary feature probe,
    # never to model parameters or geometry, and creates no optimizer/update.
    camera=torch.linalg.inv(p['predicted_context_c2w']).transpose(-1,-2)
    mass=RendererTranspose.apply(torch.ones_like(p['alpha']),p['gaussians'],camera,p['predicted_context_intrinsics'],model.gs)[0,:,0]
    gm=p['gaussian_membership'][0,:,:p['num_thing_queries']]
    result['visibility']=dict(actual_gaussian_contribution=distribution(mass),gaussians_zero_contribution=int((mass<=1e-8).sum()),
        weighted_membership_per_query=distribution((gm*mass[:,None]).sum(0)/mass.sum().clamp_min(1e-8)),
        high_membership_contribution_fraction_per_query=distribution(((gm>=.5)*mass[:,None]).sum(0)/mass.sum().clamp_min(1e-8)),
        method='read-only existing renderer adjoint; internal derivatives to dummy features only, no parameter gradients')
    if len(targets['gt_pixel_masks'][0]):
        gt_features=targets['gt_pixel_masks'][0][None].permute(0,2,1,3,4).float()
        gt_mass=RendererTranspose.apply(gt_features,p['gaussians'],camera,p['predicted_context_intrinsics'],model.gs)[0]
        qi=allowed.any(1).nonzero().flatten();ki=support.masked_fill(~allowed,-1)[qi].argmax(1)
        weighted=(gm.T@gt_mass)/gt_mass.sum(0).clamp_min(1e-8)[None]
        result['visibility']['positive_query_membership_on_gt_contribution']=distribution(weighted[qi,ki])
        result['visibility']['gt_gaussian_contribution']=gt_mass.sum(0).cpu().tolist()
    result['representation']={};result['counterfactual_stage_heads']={};stage_maps={} 
    tokens={'superpoint_initial':p['q_pre']}
    if serial:
        tokens.update(Q_rec=p['reconstruction_query'],seed=p['semantic_query_initial'],**captured_stages)
        final=original_readout(model.core,tokens['D4'],p)
        for key in ['gaussian_mask_logits','gaussian_membership','thing_logits19']:torch.testing.assert_close(final[key],p[key],rtol=1e-5,atol=1e-6)
        result['final_head_replay_verified']=True
    else:tokens['shared_decoder_final']=p['unified_query']
    for name,q in tokens.items():result['representation'][name]=representation_stats(q,labels,1729+index)
    if serial:
        for name in ['Q_rec','seed','D1','D2','D3']:
            counter=stage_prediction(model,p,tokens[name]);stats,_,_,_=analyze_native(counter,targets,allowed,support,detailed=False)
            result['counterfactual_stage_heads'][name]=stats
            if index<4 or 16<=index<20:stage_maps[name]=counter['region_mass'].detach().clone()
            del counter
        result['counterfactual_warning']='Final-trained heads applied off-distribution to earlier states; fixed geometry. Diagnostic only, not trained ablations or official performance.'
    if index<4 or 16<=index<20:
        fixed_gt=None
        reference=(Path(reference_folder) if reference_folder is not None else folder.parent/'serial')/f'{index:02d}.json'
        if (reference_folder is not None or not serial) and reference.exists():
            prior=json.loads(reference.read_text()).get('visualization')
            if prior is not None:fixed_gt=prior['gt_index']
        result['visualization']=case_figure(folder/f'{index:02d}.png',batch,p,targets,allowed,matrices,stage_maps,fixed_gt)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args()
    root=args.run_root;rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    preflight=json.loads((root/'preflight.json').read_text())
    if digest(MANIFEST)!=preflight['manifest_sha256'] or digest(HELDOUT)!=preflight['heldout_sha256']:raise ValueError('Diagnostic data changed')
    if rank==0:
        if (root/'STARTED.json').exists():raise ValueError('Refusing duplicate diagnostic')
        (root/'STARTED.json').write_text(json.dumps(dict(status='STARTED',world=world,parameter_updates=0,optimizer_created=False),indent=2)+'\n')
    if world>1:dist.barrier()
    constructors={m.architecture_name:m for m in MODELS.values()};opt=config_defaults['train_siu3r_unified_query_v1']
    for label,source in preflight['models'].items():
        checkpoint=Path(source['checkpoint'])
        if digest(checkpoint)!=source['sha256']:raise ValueError('Diagnostic checkpoint changed')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);meta=saved['meta']
        if saved['progress']['full_updates']!=256 or not meta['short_validation']:raise ValueError('Wrong short-run checkpoint')
        model=constructors[meta['architecture']](opt,config=UnifiedQueryConfig(**meta['config'])).to(device)
        if model.frozen_omega.source_identity!=meta['omega_identity'] or model.sam_proposals.source_identity!=meta['sam_identity']:raise ValueError('Frozen assets differ')
        restore_state(model,saved['model']);model.eval().requires_grad_(False);folder=root/label;folder.mkdir(exist_ok=True)
        for index,window in enumerate(preflight['windows']):
            if index%world!=rank:continue
            result=observe(model,opt,window,index,'train' if index<16 else 'heldout',device,label=='serial',label=='serial' and index==rank,folder)
            (folder/f'{index:02d}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            print(json.dumps(dict(rank=rank,model=label,index=index,scene=window['scene'],native_raw_iou=result['stage_metrics']['raw_alpha']['source_supported_only']['best_iou_mean'])),flush=True)
            (root/f'progress_rank{rank:02d}.json').write_text(json.dumps(dict(model=label,last_index=index,parameter_updates=0))+'\n')
        current=model.state_dict()
        if not all(torch.equal(current[k].cpu(),v) for k,v in saved['model'].items()):raise ValueError('Model parameters changed during diagnosis')
        if any(p.grad is not None for p in model.parameters()):raise ValueError('Diagnostic produced parameter gradients')
        (root/f'UNCHANGED_{label}_rank{rank:02d}.json').write_text(json.dumps(dict(unchanged=True,parameters=len(saved['model']),parameter_updates=0,optimizer_created=False,checkpoint_sha256=source['sha256']))+'\n')
        del model,saved,current;gc.collect();torch.cuda.empty_cache()
    if world>1:dist.barrier()
    if rank==0:(root/'INFERENCE_COMPLETE.json').write_text(json.dumps(dict(status='INFERENCE_COMPLETE',windows=32,models=2,parameter_updates=0,world=world))+'\n')
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
