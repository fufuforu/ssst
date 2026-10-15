"""Export the same 128-scene cohort and existing panoptic assembly, no parameter updates."""
import hashlib
from pathlib import Path
import torch
import torch.distributed as dist


def export(model,opt,windows,root,rank,world,device,*,expected_windows=128):
    from scripts.train_dual_stream_v1 import build_batch,write
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import generate_and_render,_official_export_view
    from scripts.export_object_locus_v3_set_official import write_official_pair
    from scripts.evaluate_ssst_validation import save_rgb,save_depth
    if len(windows)!=expected_windows:raise ValueError('Export cohort length differs')
    root=Path(root);model.eval()
    versions={name:p._version for name,p in model.named_parameters()}
    records=[]
    with torch.no_grad():
        for index in range(rank,len(windows),world):
            window=windows[index];batch=build_batch(opt,window,device)
            result=generate_and_render(model,batch['images_input'],batch['images_all'])
            rendered=result['render']
            if any(not torch.isfinite(rendered[k]).all() for k in ('images_pred','depths_pred')):raise FloatingPointError('Nonfinite final official export')
            row=write_official_pair(_official_export_view(rendered),batch,window,root/'all',target_frames='all')
            name=window['scene']+'_context'+'_'.join(map(str,window['context']));pair=root/'all'/name
            frames=batch['frame_ids'][0].cpu().tolist()
            if len(frames)!=6:raise ValueError('Fixed held-out protocol requires 2 contexts + 4 extra frames')
            for sub in ('rgb','rgb_gt','depth','depth_gt'):(pair/sub).mkdir(parents=True,exist_ok=True)
            for vi,frame in enumerate(frames):
                filename=f"{window['scene']}_{frame}.png"
                save_rgb(pair/'rgb'/filename,rendered['images_pred'][0,vi]);save_rgb(pair/'rgb_gt'/filename,batch['images_all'][0,vi])
                # Preserve the old PNG serialization divisor, NOT a metric geometry conversion.
                # Both models normalize median context depth to .25. The unmodified official
                # evaluator fits scale+shift per image; cameras/anchors never use this .15.
                save_depth(pair/'depth'/filename,rendered['depths_pred'][0,vi]/.15);save_depth(pair/'depth_gt'/filename,batch['depth_gt_m_all'][0,vi])
            gold=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation')
            references=[p/'all'/name for p in gold.glob('rank*') if (p/'all'/name).is_dir()]
            if len(references)!=1:raise ValueError('Missing/ambiguous old C GT reference')
            reference=references[0]
            for sub in ('rgb_gt','depth_gt','context_seg_gt','target_seg_gt'):
                files=sorted((pair/sub).glob('*.png'));expected=sorted((reference/sub).glob('*.png'))
                if [p.name for p in files]!=[p.name for p in expected]:raise ValueError('GT file set changed')
                if any(hashlib.sha256(p.read_bytes()).digest()!=hashlib.sha256(q.read_bytes()).digest() for p,q in zip(files,expected)):
                    raise ValueError('GT PNG differs from old C fixed evaluation')
            records.append(dict(name=name,source_index=window['source_index'],frame_ids=frames,
                input_rgb_sha256=hashlib.sha256(batch['images_all'].float().cpu().contiguous().numpy().tobytes()).hexdigest(),official_pair=row))
            write(root/f'export_progress_rank{rank:02d}.json',dict(completed=len(records),total=len(windows[rank::world]),parameter_updates=0))
            del result,rendered,batch
    if any(p._version!=versions[name] for name,p in model.named_parameters()):raise ValueError('Export changed model parameters')
    gathered=[None]*world;dist.all_gather_object(gathered,records)
    if rank==0:
        allrecords=sorted([r for rows in gathered for r in rows],key=lambda r:r['source_index'])
        if len(allrecords)!=expected_windows or len({r['name'] for r in allrecords})!=expected_windows:raise ValueError('Incomplete export coverage')
        write(root/'export_complete.json',dict(status='EXPORTED' if expected_windows==128 else 'SMOKE_EXPORTED',records=allrecords,parameter_updates=0,
            windows=expected_windows,unique_scenes=expected_windows,depth_units='normalized scene depth / .15 for legacy PNG quantization; official per-image scale+shift fit; not metric geometry',
            prediction_protocol='context-only generate, independent target calibration',panoptic_assembly='unchanged existing thresholds'))
    dist.barrier()
