"""CPU HDBSCAN on exported full-resolution features, shared evaluation, summaries."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,multiprocessing,statistics,time
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import sklearn
from scripts.fixed_instance_diagnostic_math import RECIPE,cluster_view,label_masks,evaluate_masks,supported_best_iou


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def evaluate_window(task):
    root,name,index=task;folder=Path(root)/name;torch.set_num_threads(1)
    data=torch.load(folder/f'{index:02d}_inference.pt',map_location='cpu',weights_only=False)
    started=time.monotonic();rows=[];maps=[]
    for view in range(2):
        labels,cluster_info,coverage=cluster_view(data['features'][view].numpy(),data['alpha'][view].numpy())
        clusters,ids=label_masks(labels);raw=data['raw_membership_masks'][:,view].numpy();gt=data['gt_masks'][:,view].numpy()
        valid=data['valid'][view].numpy();support=data['supported_gt'].numpy();thing=data['predicted_semantic'][view].numpy()>=2
        native_labels=data['assembled_instance'][view].numpy();assembled,_=label_masks(native_labels,background=0)
        methods=dict(membership_raw=raw,membership_nms=raw[data['nms_eligible'].numpy()],membership_assembled=assembled,
            hdbscan8d=clusters,membership_raw_common_coverage=raw&coverage,
            membership_raw_predicted_thing=raw&coverage&thing,hdbscan8d_predicted_thing=clusters&thing)
        # GT is consumed only here, after cluster_view has returned its prediction.
        metrics={name:dict(all_gt=evaluate_masks(masks,gt,valid),supported_gt=supported_best_iou(masks,gt,valid,support))
                 for name,masks in methods.items()}
        gt_thing=gt.any(0)
        row=dict(model=name,index=index,view=view,scope='train' if index<4 else 'heldout',scene=data['scene'],context=data['context'],
            cluster=cluster_info,metrics=metrics,gt_thing_pixels=int((gt_thing&valid).sum()),
            inference_coverage_on_gt_thing=float((coverage & gt_thing & valid).sum()/max(1,(gt_thing&valid).sum())),
            cluster_coverage_on_gt_thing=float(((labels>=0)&gt_thing&valid).sum()/max(1,(gt_thing&valid).sum())),
            predicted_thing_coverage_on_gt_thing=float((thing & gt_thing & valid).sum()/max(1,(gt_thing&valid).sum())),
            clustered_valid_stuff_pixels=int(((labels>=0)&valid&~gt_thing).sum()))
        rows.append(row);maps.append(labels)
    write(folder/f'{index:02d}_clustering.json',dict(rows=rows,seconds=time.monotonic()-started))
    np.savez_compressed(folder/f'{index:02d}_cluster_maps.npz',labels=np.stack(maps))
    print(json.dumps(dict(model=name,window=index,clustering_seconds=time.monotonic()-started)),flush=True)
    return rows


def summarize_metrics(rows):
    result=[]
    for name in sorted({r['model'] for r in rows}):
        for scope in ('train','heldout'):
            selected=[r for r in rows if r['model']==name and r['scope']==scope]
            if not selected:continue
            for method in selected[0]['metrics']:
                chosen=[r['metrics'][method]['all_gt'] for r in selected]
                ng=sum(r['gt_count'] for r in chosen);npred=sum(r['pred_count'] for r in chosen);tp=sum(r['tp50'] for r in chosen)
                supported=[r['metrics'][method]['supported_gt'] for r in selected];ns=sum(r['count'] for r in supported)
                result.append(dict(model=name,scope=scope,method=method,images=len(chosen),scenes=len({r['scene'] for r in selected}),
                    gt_count=ng,pred_count=npred,tp50=tp,precision50=tp/npred if npred else 0.,recall50=tp/ng if ng else None,
                    f1_50=2*tp/(npred+ng) if npred+ng else None,
                    best_gt_iou=sum(sum(r['best_gt_iou_values']) for r in chosen)/ng if ng else None,
                    one_to_one_gt_iou=sum(sum(r['one_to_one_gt_iou_values']) for r in chosen)/ng if ng else None,
                    supported_gt_count=ns,supported_best_iou=sum(r['count']*r['best_gt_iou'] for r in supported if r['count'])/ns if ns else None))
    return result


def summarize_mechanism(root,pre):
    rows=[];summary=[]
    for model in pre['models']:
        local=[json.loads((root/model['name']/f'{i:02d}_mechanism.json').read_text()) for i in range(8)]
        rows.extend(local)
        for scope in ('train','heldout'):
            selected=[r for r in local if r['scope']==scope]
            for feature in ('child256','instance8d'):
                chosen=[r['gaussian_features'][feature] for r in selected]
                def mean_path(*keys):
                    values=[]
                    for r in chosen:
                        value=r
                        for key in keys:
                            value=value.get(key) if isinstance(value,dict) else None
                        if isinstance(value,(int,float)):values.append(value)
                    return dict(mean=statistics.mean(values) if values else None,eligible_scenes=len(values))
                summary.append(dict(model=model['name'],scope=scope,feature=feature,
                    global_auroc=mean_path('global_pairs','auroc'),shuffle_auroc=mean_path('correspondence_shuffled_auroc'),
                    local_auroc=mean_path('local_pairs','auroc'),same_gt_same_parent=mean_path('same_gt_same_parent','mean'),
                    same_gt_cross_parent=mean_path('same_gt_cross_parent','mean'),different_gt=mean_path('different_gt','mean'),
                    local_same_distance=mean_path('local_same_distance','mean'),local_different_distance=mean_path('local_different_distance','mean')))
    return rows,summary


def visualize(root,pre):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def colors(labels,background):
        a=np.asarray(labels);rgb=np.zeros((*a.shape,3))
        for identity in np.unique(a):
            if identity==background:continue
            rgb[a==identity]=np.random.default_rng(int(identity)+183).uniform(.15,1.,3)
        return rgb
    folder=root/'figures';folder.mkdir(exist_ok=True)
    for index in range(8):
        fig,axes=plt.subplots(4,5,figsize=(17,13))
        for mi,spec in enumerate(pre['models']):
            data=torch.load(root/spec['name']/f'{index:02d}_inference.pt',map_location='cpu',weights_only=False)
            labels=np.load(root/spec['name']/f'{index:02d}_cluster_maps.npz')['labels']
            for view in range(2):
                raw=data['raw_membership_masks'][:,view].numpy();winner=raw.argmax(0)+1;winner[~raw.any(0)]=0
                gt_labels=data['gt_instance'][view].numpy().copy()
                gt_labels[(data['gt_semantic'][view].numpy()<2)|~data['valid'][view].numpy()]=0
                images=[data['context_rgb'][0,view].permute(1,2,0).numpy(),colors(gt_labels,0),
                    colors(winner,0),colors(data['assembled_instance'][view],0),colors(labels[view],-1)]
                for col,(label,img) in enumerate(zip(['RGB','GT (void ignored in metrics)','Membership .5 (lowest active ID shown)','Native assembled','8D HDBSCAN (black=noise/excluded)'],images)):
                    ax=axes[mi*2+view,col];ax.imshow(img);ax.set_title(label,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
                    if col==0:ax.set_ylabel(f"{spec['name']} / view {view}")
        fig.suptitle(f"{data['scene']} | arbitrary colors across methods; raw masks can overlap; per-view clustering IDs")
        fig.tight_layout();fig.savefig(folder/f'{index:02d}_{data["scene"]}.png',dpi=130);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);a=parser.parse_args();root=a.run_root
    pre=json.loads((root/'preflight.json').read_text())
    assert sklearn.__version__==pre['sklearn_version'] and asdict(RECIPE)==pre['cluster_recipe']
    assert json.loads((root/'GPU_COMPLETE.json').read_text())['windows']==16
    checks=[]
    for rank in range(4):
        r=json.loads((root/pre['models'][rank//2]['name']/f'FIXED_WEIGHTS_rank{rank:02d}.json').read_text())
        assert r['status']=='PASSED' and r['source']==r['before']==r['after'] and r['parameter_updates']==0
        assert r['windows']==list(range(rank%2,8,2));checks.append(r)
    assert checks[0]['source']==checks[1]['source'] and checks[2]['source']==checks[3]['source']
    tasks=[(str(root),m['name'],i) for m in pre['models'] for i in range(8)]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=[r for group in pool.map(evaluate_window,tasks) for r in group]
    mechanism,mechanism_summary=summarize_mechanism(root,pre)
    report=dict(status='FIXED_MECHANISM_AND_8D_CLUSTERING_COMPLETE',parameter_updates=0,optimizer_created=False,
        official128=False,models=2,windows=16,images=32,mask_summary=summarize_metrics(rows),
        mechanism_summary=mechanism_summary,mechanism_rows=mechanism,mask_rows=rows,fixed_weight_checks=checks,
        limits=['Four train/four heldout windows, one training seed; not official mAP/PQ.',
            'Primary masks evaluated per image, class agnostic, all valid GT including unsupported GT.',
            'Recall50 uses strictly IoU>0.5; Hungarian max-cardinality matching; unmatched GT IoU is zero.',
            'All foreground/background clusters retained; predicted-thing filter is a separately named shared postprocessing control.',
            'No GT, XY or desired cluster count used for HDBSCAN; fixed parameters, no heldout tuning.',
            'Noise/excluded pixels stay unassigned. No connected-component splitting or cross-view ID association.',
            'Per-image metrics differ from historical two-view source-matched supported-GT probe IoU.',
            'Gaussian sets differ by checkpoint; GT-balanced sampling with identical code is not primitive-by-primitive correspondence.',
            'Pair-probe Euclidean kNN uses up to eight local neighbors; direct loss additionally uses radius-gated other-GT neighbors.'])
    write(root/'SUMMARY.json',report)
    visualize(root,pre)
    write(root/'COMPLETE.json',dict(status=report['status'],models=2,windows=16,images=32,parameter_updates=0,official128=False))
    print(json.dumps(report['mask_summary'],indent=2),flush=True)


if __name__=='__main__':main()
