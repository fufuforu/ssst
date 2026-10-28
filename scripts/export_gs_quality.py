"""One fresh, signature-verified forward shared by all four ranking/calibration arms."""
from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.cache_gs_quality import RUNTIME,signature
import torch
import torch.distributed as dist
from scripts.gs_quality_core import *
from scripts.membership_geometry_core import tensor_sha
from scripts.fit_gs_quality import load_cache
from scripts.gs_forward_integrity import verified_forward

def export(model,opt,windows,root,rank,world,device,*,expected_windows):
    from scripts.train_dual_stream_v1 import build_batch
    from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _official_export_view
    from scripts.reciprocity_panoptic import write_official_pair
    from scripts.frozen_iou_head_core import make_reranked_pair
    from scripts.evaluate_ssst_validation import save_rgb,save_depth
    root=Path(root);run=root.parent;cfg=json.loads((run/'configuration.json').read_text())
    fit=json.loads((run/'FIT_COMPLETE.json').read_text())
    if fit['status']!='PASSED' or digest(run/'official_predictions.pt')!=fit['official_predictions_sha256']:raise ValueError('Head predictions not frozen')
    cached=[r for r in load_cache(run) if r['part']=='official'];saved=torch.load(run/'official_predictions.pt',map_location='cpu',weights_only=False)
    if saved['cache_sha256']!=digest(run/'cache/export_complete.json'):raise ValueError('Predictions/cache binding differs')
    refs={r['name']:r for r in json.loads((PARENT/'eval/export_complete.json').read_text())['records']}
    gtrefs={r['name']:r for r in map(json.loads,(PARENT/'eval/Baseline/evaluation/gt_reference_manifest.jsonl').read_text().splitlines())} if (PARENT/'eval/Baseline/evaluation/gt_reference_manifest.jsonl').exists() else None
    # The reliable G0 manifest binds GT even when the old head export has no manifest.
    from scripts.membership_geometry_core import G0
    gtrefs={r['name']:r for r in map(json.loads,(G0/'gt_reference_manifest.jsonl').read_text().splitlines())}
    offsets=[0]
    for r in cached:offsets.append(offsets[-1]+len(r['query_ids']))
    if len(windows)!=1860 or world!=8 or os.uname().nodename not in cfg['allowed_nodes'] or cfg['renderer_sha256']!=RUNTIME['sha256']:raise ValueError('Export identity differs')
    write_json(root/f'RUNTIME_rank{rank:02d}.json',dict(RUNTIME,world_size=world,node=os.uname().nodename,gpu=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device))))
    model.requires_grad_(False);model.eval();versions={n:p._version for n,p in model.named_parameters()};records=[]
    with torch.no_grad():
        for i in range(rank,len(windows),world):
            w=windows[i];r=cached[i];batch=build_batch(opt,w,device)
            name=w['scene']+'_context'+'_'.join(map(str,w['context']))
            forward_path=verified_forward(run/'cache',r)
            cached_forward=torch.load(forward_path,map_location='cpu',weights_only=False)
            if name!=r['name'] or cached_forward['signature']!=r['signature'] or cached_forward['query_ids']!=r['query_ids'] or cached_forward['source_sha256']!=SOURCE_SHA:raise ValueError('Same-forward identity differs')
            if tensor_sha(batch['images_all'])!=r['input_rgb_sha256'] or r['input_rgb_sha256']!=refs[name]['input_rgb_sha256']:raise ValueError('Official input changed')
            region=torch.zeros(1,6,r['raw_queries']+2,256,256,device=device)
            region[:,:,cached_forward['selected_channels'].to(device)]=cached_forward['region_mass'].to(device)
            raw_sem=cached_forward['raw_semantic'].long().to(device)
            semantic=torch.nn.functional.one_hot(raw_sem,20).permute(0,1,4,2,3).float()
            rendered={k:cached_forward[k].to(device) for k in ('images_pred','depths_pred')}
            out=dict(region_mass=region,p_class=cached_forward['p_class'].to(device),instance_eligible=cached_forward['instance_eligible'].to(device),
                alpha=cached_forward['alpha'].to(device),semantic_scores=semantic,render=rendered)
            native=root/'Native'/'all';write_official_pair(out,batch,w,native,target_frames='all');pair=native/name
            frames=batch['frame_ids'][0].cpu().tolist()
            for sub in ('rgb','rgb_gt','depth','depth_gt'):(pair/sub).mkdir(parents=True)
            for v,f in enumerate(frames):
                filename=f"{w['scene']}_{f}.png"
                save_rgb(pair/'rgb'/filename,rendered['images_pred'][0,v]);save_rgb(pair/'rgb_gt'/filename,batch['images_all'][0,v])
                save_depth(pair/'depth'/filename,rendered['depths_pred'][0,v]/.15);save_depth(pair/'depth_gt'/filename,batch['depth_gt_m_all'][0,v])
            hashes={sub:[[p.name,digest(p)] for p in sorted((pair/sub).glob('*.png'))] for sub in ('rgb_gt','depth_gt','context_seg_gt','target_seg_gt')}
            if hashes!=gtrefs[name]['hashes'] or frames!=gtrefs[name]['frames']:raise ValueError('Official GT differs')
            ids=torch.tensor(r['query_ids'],device=device,dtype=torch.long)-1
            for arm in ('A','C'):
                score=torch.zeros(r['raw_queries'],device=device);score[ids]=saved['predictions'][arm][offsets[i]:offsets[i+1]].to(device)
                make_reranked_pair(pair,root/(arm+'_Rank')/'all'/name,score)
                factor=torch.ones_like(score)
                if len(ids):factor[ids]=(score[ids]/score[ids].mean().clamp_min(1e-6)).clamp(.5,2).pow(.25)
                calibrated=dict(out,semantic_quality_factor=factor)
                destination=root/(arm+'_Cal')/'all';write_official_pair(calibrated,batch,w,destination,target_frames='all',score_override=score)
                for sub in ('rgb','rgb_gt','depth','depth_gt'):(destination/name/sub).symlink_to((pair/sub).resolve(),target_is_directory=True)
                for sub in ('context_seg_gt','target_seg_gt'):
                    if [[p.name,digest(p)] for p in sorted((destination/name/sub).glob('*.png'))]!=hashes[sub]:raise ValueError('Calibrated GT changed')
            records.append(dict(name=name,source_index=w['source_index'],input_rgb_sha256=r['input_rgb_sha256'],gt_hashes=hashes,same_forward=True,signature_verified=True))
            del batch,rendered,out,calibrated,cached_forward,region,semantic,raw_sem
    if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in model.named_parameters()):raise ValueError('Model changed')
    gathered=[None]*world;dist.all_gather_object(gathered,records)
    if rank==0:
        rows=sorted([r for g in gathered for r in g],key=lambda r:r['source_index'])
        if len(rows)!=1860 or len({r['name'] for r in rows})!=1860:raise ValueError('Incomplete official export')
        write_json(root/'export_complete.json',dict(status='PASSED',windows=1860,scenes=312,records=rows,source_sha256=SOURCE_SHA,
            same_forward_two_heads=True,cache_signatures_exact=True,gt_identical=True,renderer=RUNTIME,backbone_updates=0,gamma=.25,arms=['A8','C8']))
    dist.barrier()

if __name__=='__main__':
    from scripts import export_unified_query_checkpoint as loader
    loader.export=export;loader.main()
