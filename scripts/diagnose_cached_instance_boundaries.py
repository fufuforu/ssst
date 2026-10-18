"""CPU-only first pass on existing FP16 rendered features, no model inference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,statistics
import torch
from scripts.instance_boundary_diagnostic_math import score_map,calibrate_on_train,binary_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    source=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/instance_coupling_three_short_20261018')
    design=json.loads((source/'design.json').read_text());rows=[]
    torch.set_num_threads(4)
    for variant,path in design['runs'].items():
        root=Path(path)
        for stage in ('initial','final'):
            for index in range(8):
                d=torch.load(root/'probes'/stage/f'{index:02d}_coupling.pt',map_location='cpu',weights_only=False)
                row,_=score_map(d['rendered_instance_features'][0].float(),d['gt_semantic'][0],d['gt_instance'][0])
                row.update(variant=variant,stage=stage,scope='train' if index<4 else 'heldout',probe_index=index)
                rows.append(row)
    summary=[]
    for variant in design['runs']:
        for stage in ('initial','final'):
            selected=[r for r in rows if r['variant']==variant and r['stage']==stage]
            fit=calibrate_on_train([r for r in selected if r['scope']=='train'])
            for row in selected:
                if fit is not None:row['calibrated_sample_metrics']=binary_metrics(row['positive_samples'],row['negative_samples'],fit['threshold'],fit['temperature'])
            for scope in ('train','heldout'):
                items=[r for r in selected if r['scope']==scope and 'auroc' in r['metrics']]
                summary.append(dict(variant=variant,stage=stage,scope=scope,scenes=len(items),calibration=fit,
                    mean_auroc=statistics.mean(r['metrics']['auroc'] for r in items),
                    mean_shuffle_auroc=statistics.mean(r['metrics']['shuffle_auroc'] for r in items),
                    mean_calibrated_youden=statistics.mean(r['calibrated_sample_metrics']['tpr']-r['calibrated_sample_metrics']['fpr'] for r in items) if fit else None))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'CPU_CACHE_SUMMARY.json').write_text(json.dumps(dict(status='CPU_CACHE_ANALYZED',source=str(source),
        limitation='Saved FP16 post-render features, no alpha cached; CPU results are preliminary. GPU will use native FP32 and coverage controls.',
        summary=summary,rows=rows),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
