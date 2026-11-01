"""Frozen QPanchor1 export with the SAME GT camera contract as its training loss."""
from pathlib import Path
import sys,os,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import torch
import torch.distributed as dist
from tokengs.models.qp_anchor_gt_camera import AnchorQPJoint4GTCamera
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.options import config_defaults
from scripts.native_joint4_support import task_state
from scripts.resilient_esam_cache import install_caches
from scripts.train_dual_stream_v1 import build_batch
from scripts.verified_runtime_stage import load_run_config,sha
from scripts.gt_camera_protocol import camera_audit,PROTOCOL
from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _official_export_view
from scripts.export_object_locus_v3_set_official import write_official_pair
from scripts.evaluate_ssst_validation import save_rgb,save_depth
from scripts.scannet_data_snapshot import verify_job_snapshot

def main():
    run=Path(sys.argv[1]);root=Path(sys.argv[2]);cfg=load_run_config(run);assert cfg['context_scale_policy']['bounds']==[.25,4.] and cfg['context_scale_policy']['degenerate_baseline_public_epsilon']==1e-6;verify_job_snapshot(cfg)
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==4
    torch.cuda.set_device(local);dist.init_process_group('nccl');device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert json.loads((run/'train/TRAINING_COMPLETE.json').read_text())['full_updates']==4172
    source=run/'train/checkpoint_final_live.pt';saved=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    assert saved['progress']==dict(epoch=4,update_in_epoch=0,full_updates=4172) and saved['meta']['code_sha']==cfg['code_sha'] and saved['meta']['camera_protocol']==PROTOCOL
    opt=config_defaults['train_siu3r_unified_query_v1'].evolve(num_input_views=2,num_views=6)
    model=AnchorQPJoint4GTCamera(opt,config=UnifiedQueryConfig(**saved['meta']['config'])).to(device);target=task_state(model);assert set(target)==set(saved['model'])
    with torch.no_grad():
        for n,t in target.items():assert t.shape==saved['model'][n].shape;t.copy_(saved['model'][n])
    del saved;model.requires_grad_(False).eval();model._anchor_smoke=False;install_caches(model,cfg['readonly_frozen_cache']);versions={n:p._version for n,p in model.named_parameters()}
    localbase=Path(os.environ['SSST_RUNTIME_CONFIG']).parent.parent
    windows=json.loads((localbase/'cohort.json').read_text())['records'];refs={r['name']:r for r in json.loads((localbase/'reference_manifest.json').read_text())['records']};assert len(windows)==1860
    rows=[]
    with torch.no_grad():
        for i in range(rank,len(windows),world):
            w=windows[i];batch=build_batch(opt,w,device);name=w['scene']+'_context'+'_'.join(map(str,w['context']));ref=refs[name]
            rgbsha=hashlib.sha256(batch['images_all'].float().cpu().contiguous().numpy().tobytes()).hexdigest();assert rgbsha==ref['input_rgb_sha256']
            pred=model.generate_gt(batch);render=model.render_generated_gt(pred,batch)
            assert torch.isfinite(render['images_pred']).all() and torch.isfinite(render['depths_pred']).all()
            write_official_pair(_official_export_view(render),batch,w,root/'all',target_frames='all');pair=root/'all'/name
            frames=batch['frame_ids'][0].cpu().tolist();assert frames==ref['frames']
            for sub in ['rgb','rgb_gt','depth','depth_gt']:(pair/sub).mkdir(exist_ok=True)
            for v,f in enumerate(frames):
                fn=w['scene']+'_'+str(f)+'.png';save_rgb(pair/'rgb'/fn,render['images_pred'][0,v]);save_rgb(pair/'rgb_gt'/fn,batch['images_all'][0,v])
                # Original PNG writer; inverse of the SAME scene gauge used for GT camera translations.
                save_depth(pair/'depth'/fn,render['depths_pred'][0,v]/pred['GT_camera_gauge']['generated_units_per_meter']);save_depth(pair/'depth_gt'/fn,batch['depth_gt_m_all'][0,v])
            hashes={sub:[[p.name,sha(p)] for p in sorted((pair/sub).glob('*.png'))] for sub in ['rgb_gt','depth_gt','context_seg_gt','target_seg_gt']};assert hashes==ref['hashes']
            camera=camera_audit(batch,pred['GT_camera_gauge'])
            rows.append(dict(name=name,source_index=w['source_index'],frame_ids=frames,input_rgb_sha256=rgbsha,generation_frame_ids=frames[:2],generation_input_rgb_sha256=hashlib.sha256(batch['images_input'].float().cpu().contiguous().numpy().tobytes()).hexdigest(),GT_camera=camera,GT_provider_cam_view=batch['cam_view_all'][0].cpu().tolist(),GT_intrinsics=batch['intrinsics_all'][0].cpu().tolist(),empty_depth_views=[int(v) for v in range(6) if not (render['depths_pred'][0,v]>0).any()],hashes=hashes))
            (root/f'progress_rank{rank:02d}.json').write_text(json.dumps(dict(completed=len(rows),total=len(windows[rank::world]),last=name)))
            if len(rows)%25==0:print(json.dumps(dict(rank=rank,completed=len(rows),last=name)),flush=True)
            del pred,render,batch
    assert all(not p.requires_grad and p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
    gather=[None]*world;dist.all_gather_object(gather,rows)
    if rank==0:
        records=sorted([r for rows in gather for r in rows],key=lambda r:r['source_index']);assert len(records)==len({r['name'] for r in records})==1860
        with (root/'gt_reference_manifest.jsonl').open('w') as f:
            for r in records:f.write(json.dumps(dict(name=r['name'],frames=r['frame_ids'],hashes=r['hashes']))+'\n')
        (root/'export_complete.json').write_text(json.dumps(dict(status='EXPORTED',records=records,windows=1860,scenes=312,parameter_updates=0,prediction_protocol=PROTOCOL+'; Context RGB plus GT Context K/c2w; no Target RGB or Target camera in generation; GT cameras for supervised Context and Target',depth_units='rendered generated units divided by per-window .15*a to meters; original PNG writer; official scale+shift scoring',camera_protocol=PROTOCOL),indent=2)+'\n')
        final=dict(status='PASSED',source=str(source),sha256=sha(source),parameter_updates_during_export=0)
        (root.parent/'FINAL_CHECKPOINT_FROZEN.json').write_text(json.dumps(final,indent=2)+'\n')
        (root/'checkpoint_identity.json').write_text(json.dumps(dict(checkpoint=str(source),meta=cfg,parameter_updates=0,smoke_only=False),indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
