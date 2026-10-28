"""Fixed Baseline16, 16 train +16 internal seen-scene windows; R0/Rcam/Rrgb."""
from pathlib import Path
import sys,json,argparse,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gs_quality_runtime import prepare_pinned_runtime
RUNTIME=prepare_pinned_runtime(ROOT)
import torch
import torch.distributed as dist
from scripts.membership_geometry_core import sha,tensor_sha,write
from scripts.reconstruction_gap_core import align_gt_cameras,appearance_gaussians

def export(model,opt,windows,root,rank,world,device,*,expected_windows):
    from scripts.train_dual_stream_v1 import build_batch
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import generate_and_render,_official_export_view
    from scripts.export_object_locus_v3_set_official import write_official_pair
    from scripts.evaluate_ssst_validation import save_rgb,save_depth
    from tokengs.models.input_types import ModelInputDecoder
    root=Path(root);run=root.parent;cfg=json.loads((run/'configuration.json').read_text())
    if len(windows)!=32 or world!=4 or sha(run/'diagnostic_cohort.json')!=cfg['cohort_sha256']:raise ValueError('Fixed diagnostic cohort differs')
    model.requires_grad_(False);model.eval();versions={n:p._version for n,p in model.named_parameters()};records=[]
    for i in range(rank,len(windows),world):
        w=windows[i];batch=build_batch(opt,w,device)
        if tensor_sha(batch['images_all'])!=w['input_rgb_sha256']:raise ValueError('Diagnostic input hash differs')
        with torch.no_grad():
            result=generate_and_render(model,batch['images_input'],batch['images_all']);generated=result['generated'];calib=result['camera_calibration']
            fixed=generated['gaussians'].detach().clone();camera=torch.linalg.inv(calib['c2w']).transpose(-1,-2).detach();intr=calib['intrinsics'].detach()
            gt=torch.linalg.inv(batch['cam_view_all'].transpose(-1,-2))[0]
            aligned,audit=align_gt_cameras(gt,generated['predicted_context_c2w'][0]);renders=dict(R0=result['render'])
            if aligned is not None:
                renders['Rcam']=model.render_reconstruction(model._reconstruction_from_gaussians(fixed),ModelInputDecoder(cam_view=torch.linalg.inv(aligned[None]).transpose(-1,-2),intrinsics=batch['intrinsics_all']))
        colours=torch.nn.Parameter(fixed[...,11:].clone());initial=colours.detach().clone();optim=torch.optim.Adam([colours],lr=.01);history=[]
        decoder=ModelInputDecoder(cam_view=camera[:,:2],intrinsics=intr[:,:2])
        for step in range(100):
            optim.zero_grad(set_to_none=True)
            render=model.render_reconstruction(model._reconstruction_from_gaussians(appearance_gaussians(fixed,colours)),decoder)
            loss=(render['images_pred']-batch['images_all'][:,:2]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite RGB fit')
            loss.backward()
            if colours.grad is None or not torch.isfinite(colours.grad).all():raise ValueError('Invalid RGB gradient')
            optim.step()
            with torch.no_grad():colours.clamp_(0,1)
            history.append(float(loss.detach()));del render,loss
        with torch.no_grad():
            fitted=appearance_gaussians(fixed,colours)
            if not torch.equal(fitted[...,:11],fixed[...,:11]) or torch.equal(initial,colours):raise ValueError('Appearance-only update audit failed')
            renders['Rrgb']=model.render_reconstruction(model._reconstruction_from_gaussians(fitted),ModelInputDecoder(cam_view=camera,intrinsics=intr))
            name=w['scene']+'_context'+'_'.join(map(str,w['context']));frames=batch['frame_ids'][0].cpu().tolist()
            # Shared GT/segmentation writer; changed arms replace only reconstruction.
            baseout=_official_export_view(result['render'])
            error={}
            for arm,render in renders.items():
                if any(not torch.isfinite(render[k]).all() for k in ('images_pred','depths_pred')):raise ValueError('Nonfinite render')
                pair=root/arm/'all'/name
                write_official_pair(baseout,batch,w,root/arm/'all',target_frames='all')
                for sub in ('rgb','rgb_gt','depth','depth_gt'):(pair/sub).mkdir(parents=True)
                for v,f in enumerate(frames):
                    fn=f"{w['scene']}_{f}.png"
                    save_rgb(pair/'rgb'/fn,render['images_pred'][0,v]);save_rgb(pair/'rgb_gt'/fn,batch['images_all'][0,v])
                    save_depth(pair/'depth'/fn,render['depths_pred'][0,v]/.15);save_depth(pair/'depth_gt'/fn,batch['depth_gt_m_all'][0,v])
                # Native visibility stratification held fixed for all diagnostic arms.
                err=(render['images_pred']-batch['images_all']).square().mean(2)[0]
                alpha=result['render']['alphas_pred'][0,:,0]
                error[arm]={}
                for scope,sl in [('context',slice(0,2)),('novel',slice(2,None))]:
                    error[arm][scope]={}
                    for label,mask in [('low_native_coverage',alpha[sl]<=.5),('high_native_coverage',alpha[sl]>.5)]:
                        error[arm][scope][label]=dict(pixels=int(mask.sum()),rgb_mse=float(err[sl][mask].mean()) if mask.any() else None)
            row=dict(name=name,frame_ids=frames,window_index=i,split=w['diagnostic_split'],backbone_scene_seen=True,arms=list(renders),camera_alignment=audit,
                appearance=dict(steps=100,lr=.01,objective='Context RGB MSE',novel_GT_used=False,colour_changed=True,geometry_opacity_rotation_bitwise_fixed=True,
                    fixed_noncolour_sha256=tensor_sha(fixed[...,:11]),initial_colour_sha256=tensor_sha(initial),final_colour_sha256=tensor_sha(colours),loss_history=history),
                input_rgb_sha256=tensor_sha(batch['images_all']),coverage_errors=error)
            write(root/'per_window'/(name+'.json'),row);records.append(row)
        del batch,result,generated,renders,fixed,fitted,colours,optim
    if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('Frozen Baseline16 modified')
    allrows=[None]*world;dist.all_gather_object(allrows,records)
    if rank==0:
        rows=sorted([r for a in allrows for r in a],key=lambda r:r['window_index'])
        if [r['window_index'] for r in rows]!=list(range(32)):raise ValueError('Diagnostic coverage incomplete')
        write(root/'export_complete.json',dict(status='PASSED',records=rows,windows=32,source_sha256=cfg['source_sha256'],renderer=RUNTIME,model_updates=0,
            internal_scenes_seen_by_backbone=True,not_official1860_selection=True,LPIPS_pilot_submitted=False))
    dist.barrier()

if __name__=='__main__':
    # Baseline16 has a different continuation metadata policy from the old
    # generic export loader; restore directly and validate its fixed identity.
    from scripts.train_unified_query_v1 import restore_state
    from scripts.train_dual_stream_v1 import MANIFEST,HELDOUT
    from tokengs.models.superpoint_structural_variants import build_structural_model
    from tokengs.models.unified_superpoints import UnifiedQueryConfig
    from tokengs.options import config_defaults
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();run=a.run_root
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);dist.init_process_group('nccl');device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=json.loads((run/'configuration.json').read_text());saved=torch.load(run/'source_checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
    if sha(run/'source_checkpoint.pt')!=cfg['source_sha256'] or saved['progress']!=dict(epoch=16,update_in_epoch=0,full_updates=16688):raise ValueError('Baseline16 source differs')
    if saved['meta']['manifest_sha256']!=sha(MANIFEST) or saved['meta']['heldout_sha256']!=sha(HELDOUT):raise ValueError('Source dataset differs')
    opt=config_defaults['train_siu3r_unified_query_v1'];model=build_structural_model('dual_v4init_isolated_noca',opt,UnifiedQueryConfig(**saved['meta']['config'])).to(device)
    if model.frozen_omega.source_identity!=saved['meta']['omega_identity']:raise ValueError('Omega differs')
    if (model.sam_proposals.source_identity if model.sam_proposals is not None else None)!=saved['meta']['sam_identity']:raise ValueError('SAM differs')
    restore_state(model,saved['model']);export(model,opt,json.loads((run/'diagnostic_cohort.json').read_text())['records'],run/'diagnostics',rank,world,device,expected_windows=32)
    dist.destroy_process_group()
