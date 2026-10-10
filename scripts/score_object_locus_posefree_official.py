"""Score native PNG exports with pinned SIU3R; merge global official segmentation states."""
from __future__ import annotations
import argparse, json, os, sys, subprocess, tempfile, time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from scripts.invoke_siu3r_official_evaluator import evaluate, SIU3R_COMMIT
from scripts.posefree_official_state_helpers import create, update, states, merge, itemize


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(itemize(value),indent=2,allow_nan=False)+'\n');temp.replace(path)


def contract(root,names):
    """Real-window contract for state merging against unmodified Evaluator.evaluate."""
    checks=[]
    for arm in ('all','novel'):
        chosen=names[:2]
        with tempfile.TemporaryDirectory(prefix='official-contract-',dir=root) as temp:
            temp=Path(temp)
            for name in chosen:(temp/name).symlink_to((root/arm/name).resolve(),target_is_directory=True)
            expected=evaluate(temp,device='cpu',image_quality=False,depth_quality=False)
            blobs=[]
            for name in chosen:
                e=create(root/arm)
                for view in ('context','target'):
                    pair=root/arm/name
                    update(e,e.process_segmentation(pair/f'{view}_seg_pred',pair/f'{view}_seg_gt'),view)
                blobs.append({'names':[name],'states':states(e)})
            actual=merge(list(reversed(blobs)),root/arm)
            for view in ('context','target'):
                for metric in ('miou','pq'):
                    key=f'{view}_{metric}'
                    assert abs(actual[key]-expected[key])<1e-7,(arm,key,actual[key],expected[key])
                for metric in ('map','map_50','map_75'):
                    assert abs(actual[f'{view}_map'][metric]-expected[f'{view}_map'][metric])<1e-7,(arm,view,metric)
            checks.append({'arm':arm,'names':chosen,'status':'PASS'})
    write(root/'official_merge_contract.json',{'status':'PASS','checks':checks,'official_commit':SIU3R_COMMIT})


def score_shard(args):
    rank=int(os.environ.get('RANK','0')) if args.shard is None else args.shard
    device=f"cuda:{os.environ.get('LOCAL_RANK','0')}"
    torch.cuda.set_device(device);torch.set_num_threads(4)
    root=args.root/f'rank{rank:02d}'
    done=json.loads((root/'export_complete.json').read_text())
    names=sorted(row['name'] for row in done['records'])
    assert len(names)==done['identity']['windows'] and len(set(names))==len(names)
    if (root/'score_complete.json').exists():print(f'SCORE_REUSED rank={rank}',flush=True);return
    if names:
        if not (root/'official_reconstruction.json').exists():
            reconstruction=evaluate(root/'all',device=device,segmentation=False)
            write(root/'official_reconstruction.json',reconstruction)
        if not (root/'official_merge_contract.json').exists():contract(root,names)
    saved={}
    started=time.monotonic()
    for arm in ('all','novel'):
        e=create(root/arm)
        for i,name in enumerate(names,1):
            pair=root/arm/name
            for view in ('context','target'):
                data=e.process_segmentation(pair/f'{view}_seg_pred',pair/f'{view}_seg_gt')
                update(e,data,view)
                del data
            write(root/'score_progress.json',{'stage':'SEGMENTATION','arm':arm,'completed':i,'total':len(names),
                'elapsed_seconds':time.monotonic()-started,'latest':name})
            if i%10==0:print(f'SCORE rank={rank} arm={arm} completed={i}/{len(names)}',flush=True)
        saved[arm]={'names':names,'states':states(e)}
    temp=root/'official_states.pt.tmp';torch.save(saved,temp);temp.replace(root/'official_states.pt')
    write(root/'score_complete.json',{'status':'COMPLETE','windows':len(names),'official_commit':SIU3R_COMMIT})


def reduce(args):
    blobs=[];names=[];records=[];metadata=[];identities=[]
    for rank in range(args.shards):
        root=args.root/f'rank{rank:02d}'
        receipt=json.loads((root/'score_complete.json').read_text());assert receipt['status']=='COMPLETE'
        export=json.loads((root/'export_complete.json').read_text())
        records.extend(export['records']);identities.append(export['identity'])
        metadata.append(json.loads((root/'checkpoint_metadata.json').read_text()))
        blobs.append(torch.load(root/'official_states.pt',map_location='cpu',weights_only=False))
        names.extend(blobs[-1]['all']['names'])
    assert len(set(names))==len(names)
    manifest=json.loads(Path('/space/mawb/SIU3R/data/scannet/val_pair.json').read_text())
    expected={r['scan']+'_context'+'_'.join(map(str,r['context_ids'])) for r in manifest}
    if not args.partial:
        assert set(names)==expected and len(names)==1860
        assert len({n.split('_context')[0] for n in names})==312
        assert len({r['manifest_index'] for r in records})==1860
        assert all(i['limit'] is None for i in identities)
    byname={r['name']:r for r in records}
    observations={scope:[] for scope in ('context','target-all','true-novel')}
    for rank in range(args.shards):
        root=args.root/f'rank{rank:02d}'
        for name in blobs[rank]['all']['names']:
            pair=root/'all'/name
            rgb=json.loads((pair/'render_scores.json').read_text());depth=json.loads((pair/'depth_scores.json').read_text())
            d={r['item']:r for r in depth}
            assert len(rgb)==len(depth)==6
            ctx=set(byname[name]['frame_ids'][:2])
            for r in rgb:
                frame=int(Path(r['item']).stem.rsplit('_',1)[1])
                row={k:r[k] for k in ('psnr','ssim','lpips')}|{k:d[r['item']][k] for k in ('absrel','rmse')}
                assert all(np.isfinite(v) for v in row.values())
                observations['target-all'].append(row)
                observations['context' if frame in ctx else 'true-novel'].append(row)
    recon={scope:{k:float(np.mean([r[k] for r in rows])) for k in ('absrel','rmse','psnr','ssim','lpips')}|{'images':len(rows)}
           for scope,rows in observations.items()}
    allseg=merge([b['all'] for b in blobs],args.root)
    novelseg=merge([b['novel'] for b in blobs],args.root)
    scopes={}
    for scope,seg,view in (('context',allseg,'context'),('target-all',allseg,'target'),('true-novel',novelseg,'target')):
        scopes[scope]=recon[scope]|{'mIoU_s':seg[f'{view}_miou'],'mAP':seg[f'{view}_map']['map'],
            'PQ':seg[f'{view}_pq'],'mIoU_t':None,'mIoU_t_status':'NOT_TRAINED'}
    assert all(m['git_sha']=='e80c99380a4eb04456dc7fb69c7382ff11af1051' for m in metadata)
    report={'status':'PARTIAL_ENGINEERING_CHECK' if args.partial else 'EVALUATED',
        'windows':len(names),'unique_scenes':len({n.split('_context')[0] for n in names}),
        'scopes':scopes,'official_segmentation':{'all':allseg,'novel':novelseg},
        'official_commit':SIU3R_COMMIT,'evaluation_code_sha':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        'training_checkpoint_metadata':metadata[0],'export_identity':identities[0],
        'checkpoint_sha256':'40e93e3e9d1157f1d6af97e51414bb098a116f6f2444b190993dbc0a33a47824',
        'image_protocol':'Unmodified pinned SIU3R Evaluator.evaluate on exported RGB uint8 PNG and depth millimetre uint16 PNG; per-image RGB/depth metrics averaged over images; GT-positive per-image scale-and-shift depth alignment. Context/novel reconstruction scopes filter the native per-image scores.',
        'segmentation_protocol':'Unchanged existing panoptic exporter and pinned process_segmentation/metric classes; merged sufficient states and canonical pair order; global COCO AP, never a scene/shard AP average. Real two-window contracts match native evaluator at 1e-7.',
        'text_metric_note':'mIoU_t means text-referred segmentation. The locked visual model has no trained text branch; unmeasured, not zero and not target-view semantic IoU.',
        'camera_disclosure':'场景生成只使用两张context；监督/目标相机使用独立图像标定。指定新视角渲染仍需要目标相机。',
        'optimizer_updates':0,'job_id':os.environ.get('SLURM_JOB_ID')}
    write(args.root/'metrics.json',report)
    lines=['# Final epoch 8 official SIU3R evaluation','',f"Status: {report['status']}; {len(names)} windows / {report['unique_scenes']} scenes.",'',
        '| Scope | AbsRel↓ | RMSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | mIoUₛ↑ | mAP↑ | PQ↑ | mIoUₜ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for scope,row in scopes.items():
        values=[f"{row[k]:.6f}" for k in ('absrel','rmse','psnr','ssim','lpips','mIoU_s','mAP','PQ')]
        lines.append('| '+scope+' | '+' | '.join(values)+' | NOT_TRAINED |')
    lines+=['',report['text_metric_note'],'',report['image_protocol'],'',report['segmentation_protocol'],'',report['camera_disclosure'],
        '',f"Training: 8 epochs / 8344 updates / 66752 new exposures; SHA {metadata[0]['git_sha']}.",
        f"SIU3R evaluator: {SIU3R_COMMIT}; evaluation code: {report['evaluation_code_sha']}."]
    (args.root/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(scopes,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--shards',type=int,required=True)
    p.add_argument('--shard',type=int);p.add_argument('--reduce',action='store_true');p.add_argument('--partial',action='store_true')
    a=p.parse_args()
    assert subprocess.check_output(['git','-C','/space/mawb/SIU3R','rev-parse','HEAD'],text=True).strip()==SIU3R_COMMIT
    assert not subprocess.check_output(['git','-C','/space/mawb/SIU3R','diff','--name-only','HEAD','--','src'],text=True).strip()
    torch.set_num_threads(4)
    reduce(a) if a.reduce else score_shard(a)

if __name__=='__main__':main()
