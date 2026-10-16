"""Summarize independent fixed-weight shards, with explicit stage and oracle caveats."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    arms={};tables=[];cases=[]
    for arm in ('isolated','weight005'):
        rows=[]
        for rank in range(4):
            root=args.output/arm/f'rank{rank:02}'
            complete=json.loads((root/'COMPLETE.json').read_text())
            if complete['smoke'] or not complete['parameter_versions_unchanged'] or complete['parameter_updates']:raise ValueError('Incomplete/modified diagnostic')
            records=[json.loads((root/f'{i}.json').read_text()) for i in complete['records']];rows+=records
        if len(rows)!=128 or len({r['source_index'] for r in rows})!=128:raise ValueError('Incomplete heldout128')
        stages={};groups={k:[] for k in ('good_correct','good_no_object','good_wrong_class','good_below_score','poor_mask','small','duplicate')}
        paired=[]
        for row in rows:
            view_area=np.asarray(row['gt_view_pixels']);raw_sam=np.asarray(row['raw_sam_best_IoU_per_view'])
            if raw_sam.size:stages.setdefault('raw_sam_visible_view_oracle',[]).extend(raw_sam[view_area>0].tolist())
            n=len(row['gt_ids'])
            if not n:continue
            matrices={k:np.asarray(v,dtype=float).reshape(-1,n) for k,v in row['mask_IoU'].items()}
            for key,value in matrices.items():stages.setdefault(key,[]).extend(value.max(0).tolist() if len(value) else [0.]*n)
            raw=matrices['raw'];best_query=raw.argmax(0);best=raw.max(0)
            pc=np.asarray(row['query_class_probabilities']);gtclasses=np.asarray(row['gt_classes']);pixels=np.asarray(row['gt_pixels'])
            for k,q in enumerate(best_query):
                category='poor_mask' if best[k]<.5 else 'good_no_object' if pc[q].argmax()==18 else 'good_wrong_class' if pc[q,:18].argmax()+2!=gtclasses[k] else 'good_below_score' if pc[q,:18].max()<.05 else 'good_correct'
                groups[category].append(float(best[k]))
                if pixels[k]<1024:groups['small'].append(float(best[k]))
                if (raw[:,k]>=.5).sum()>1:groups['duplicate'].append(float(best[k]))
                ownership={}
                for layer,values in row['ownership'].items():ownership[layer]={name:float(value[q]) for name,value in values.items()}
                paired.append(dict(source_index=row['source_index'],gt_id=row['gt_ids'][k],query=int(q),category=category,pixels=int(pixels[k]),raw_iou=float(best[k]),final_same_query_iou=float(matrices['final'][q,k]),ownership=ownership))
            loss=best-matrices['final'].max(0)
            cases.append(dict(arm=arm,source_index=row['source_index'],loss=float(loss.max()),figure=str((next((args.output/arm).glob(f"rank*/{row['source_index']}.png"),Path('missing'))).resolve())))
        stage_stats={k:dict(instances=len(v),mean_best_iou=float(np.mean(v)),recall50=float(np.mean(np.asarray(v)>=.5)),recall75=float(np.mean(np.asarray(v)>=.75))) for k,v in stages.items()}
        associations={}
        for layer in ('6','8','10','12'):
            associations[layer]={}
            for name in ('read_child_js','anchor_child_js','route_child_js'):
                x=np.asarray([r['ownership'][layer][name] for r in paired]);y=np.asarray([r['raw_iou'] for r in paired])
                associations[layer][name]=dict(pearson_with_raw_iou=float(np.corrcoef(x,y)[0,1]) if len(x)>1 and x.std()>0 and y.std()>0 else None,
                    mean=float(x.mean()) if len(x) else None,shuffle_mean=float(np.mean([r['ownership'][layer][name+'_query_shuffle'] for r in paired])) if paired else None)
        arms[arm]=dict(windows=128,instances=len(paired),stages=stage_stats,categories={k:dict(count=len(v),mean_raw_iou=float(np.mean(v)) if v else None) for k,v in groups.items()},ownership_associations=associations,
            official_packed_max_mismatch_fraction=max(max(r['official_packed_mismatch_fraction']) for r in rows),
            first_window_noop_witnesses=sum(r['observer_noop_witness'] for r in rows),instances_detail=paired)
        tables.append(f'### {arm}\n\n| Stage | Best IoU | Recall@0.5 | Recall@0.75 |\n|---|---:|---:|---:|')
        for key,value in stage_stats.items():tables.append(f"| {key} | {value['mean_best_iou']:.4f} | {value['recall50']:.2%} | {value['recall75']:.2%} |")
    summary=dict(status='COMPLETE',parameter_updates=0,models=arms,cases=sorted(cases,key=lambda x:-x['loss'])[:20])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    text='# GFM instance diagnostic\n\n'+ '\n'.join(tables)+'\n\nThese are diagnostic GT-oracle best-mask recalls, not official mAP/PQ. Context-only GT instances; small means <1024 total valid context pixels, not COCO AP-small. Raw SAM is a separate per-visible-view oracle with a different denominator; selected SAM supports are cross-view slots including actual dedup/merge/cap, without learned fallback supports. Initial/layer 2D projections are probes, not the trained official Gaussian readout. Ownership Read may use global context; differences or correlations alone do not establish errors. Write is disabled in both models; route is a hypothetical routing distribution, not actual written feedback. Shuffle controls preserve marginals. CPU/GPU differences from golden outputs are reported explicitly.\n'
    (args.output/'report.md').write_text(text)
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',windows_per_model=128,parameter_updates=0))+'\n')


if __name__=='__main__':main()
