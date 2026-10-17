"""Aggregate the fixed-weight diagnostic; no claims of official AP or tuned thresholds."""
import argparse,json,statistics,html
from pathlib import Path


def mean(values):
    values=[v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def aggregate_stage(rows,path):
    values=[]
    for row in rows:
        value=row
        for key in path:value=value[key]
        values.extend(value['best_iou'])
    return dict(gt_count=len(values),mean_iou=mean(values),recall50=mean([float(x>=.5) for x in values]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();root=args.run_root
    complete=json.loads((root/'INFERENCE_COMPLETE.json').read_text());assert complete['parameter_updates']==0 and complete['windows']==32
    data={label:[json.loads((root/label/f'{i:02d}.json').read_text()) for i in range(32)] for label in ['serial','unified']}
    preflight=json.loads((root/'preflight.json').read_text())
    for label,rows in data.items():
        for rank in range(4):
            receipt=json.loads((root/f'UNCHANGED_{label}_rank{rank:02d}.json').read_text());assert receipt['unchanged'] and not receipt['optimizer_created']
        assert all(r['witness']['passed'] for r in rows if r['witness'] is not None)
    assert all(r['final_head_replay_verified'] for r in data['serial'])
    for a,b in zip(data['serial'],data['unified']):
        assert (a['index'],a['scene'],a['context'],a['input_rgb_sha256'],a['gt_ids'],a['gt_classes'])==(b['index'],b['scene'],b['context'],b['input_rgb_sha256'],b['gt_ids'],b['gt_classes'])
    report=dict(status='DIAGNOSTICS_COMPLETE',parameter_updates=0,models={},same_32_context_inputs_and_gt=True,official_metrics=False,source_checkpoints=preflight['models'],counterfactual_warning='Final-trained heads on earlier states are off-distribution diagnostics; threshold scans are not evaluation changes.')
    lines=['# Serial Query fixed-weight diagnosis','', '32 windows per model (16 training, 16 heldout). No parameter updates. Native production assembly and witness checks passed. These are GT-assisted diagnostics, not official mAP/PQ.','']
    for label,rows in data.items():
        report['models'][label]={}
        for scope in ['train','heldout']:
            r=[v for v in rows if v['scope']==scope]
            result=dict(windows=len(r),gt=sum(v['gt_count'] for v in r),supported_gt=sum(v['supported_gt'] for v in r),queries=sum(v['query_count'] for v in r),class_eligible=sum(v['class_eligible'] for v in r),nms_kept=sum(v['nms_kept'] for v in r),final_instances=sum(v['final_instances'] for v in r),
                queries_gaussian_below_half=sum(v['queries_gaussian_below_half'] for v in r),queries_no_rendered_half=sum(v['queries_no_rendered_half'] for v in r),queries_no_visible_half=sum(v['queries_no_visible_half'] for v in r),alpha_visible_fraction=mean([v['alpha_visible_fraction'] for v in r]),
                stage_metrics={name:{kind:aggregate_stage(r,['stage_metrics',name,kind]) for kind in ['class_agnostic','correct_class','source_supported_all_gt','source_supported_only']} for name in r[0]['stage_metrics']},
                representation={name:{metric:mean([v['representation'][name].get(metric) for v in r]) for metric in ['mean_channel_variance','same_gt_cosine','different_gt_cosine','cosine_separation','shuffle_separation_mean','shuffle_separation_std']} for name in r[0]['representation']},
                thresholds={threshold:{kind:aggregate_stage(r,['threshold_diagnostic_only',threshold,kind]) for kind in ['alpha','no_alpha']} for threshold in r[0]['threshold_diagnostic_only']})
            result['visibility']=dict(gaussians=sum(v['visibility']['actual_gaussian_contribution']['count'] for v in r),
                gaussians_zero_contribution=sum(v['visibility']['gaussians_zero_contribution'] for v in r),
                contribution_weighted_membership_scene_mean=mean([v['visibility']['weighted_membership_per_query']['mean'] for v in r]))
            for threshold,curve in result['thresholds'].items():
                raw=[v['threshold_diagnostic_only'][threshold] for v in r]
                tp=sum(v['positive_query_intersection_pixels'] for v in raw);pp=sum(v['positive_query_prediction_pixels'] for v in raw);yy=sum(v['positive_query_gt_pixels'] for v in raw)
                curve.update(positive_query_micro_precision=tp/pp if pp else 0.,positive_query_micro_recall=tp/yy if yy else None)
            result['counterfactual_stage_heads']={name:{stage:aggregate_stage(r,['counterfactual_stage_heads',name,'stage_metrics',stage,'source_supported_only']) for stage in ['raw_alpha','final']} for name in r[0]['counterfactual_stage_heads']}
            result['shuffle_correspondence_iou_scene_mean']=mean([v['shuffled_query_correspondence_best_iou_mean'] for v in r])
            report['models'][label][scope]=result
            lines += [f'## {label} / {scope}','',f"GT: {result['gt']}, source-supported GT: {result['supported_gt']}, queries: {result['queries']}. Class eligible: {result['class_eligible']}; NMS kept: {result['nms_kept']}; final instances: {result['final_instances']}.",'', '| stage | all-GT oracle IoU | class-correct oracle IoU | source-supported-only IoU |','|---|---:|---:|---:|']
            for name,stats in result['stage_metrics'].items():
                def number(x):return 'NA' if x is None else f'{x:.4f}'
                lines.append('| '+name+' | '+' | '.join(number(stats[k]['mean_iou']) for k in ['class_agnostic','correct_class','source_supported_only'])+' |')
            lines += ['',f"Queries with Gaussian max < .5: {result['queries_gaussian_below_half']}; no rendered >=.5: {result['queries_no_rendered_half']}; no visible >=.5: {result['queries_no_visible_half']}.",'']
    (root/'SUMMARY.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    gallery=['<!doctype html><meta charset="utf-8"><title>Serial query diagnosis</title><h1>Fixed-weight native and counterfactual mask diagnostics</h1><p>Probability colors: blue=0, green=.5, red=1. Earlier-state heads are off-distribution diagnostics, not trained ablations. Model images use the same GT instance per window.</p>']
    for index in list(range(4))+list(range(16,20)):
        row=data['serial'][index];gallery.append('<h2>'+html.escape(str(index)+' '+row['scope']+' '+row['scene'])+'</h2>')
        for label in ['serial','unified']:
            if (root/label/f'{index:02d}.png').exists():gallery.append(f'<h3>{label}</h3><a href="{label}/{index:02d}.png"><img style="max-width:100%" src="{label}/{index:02d}.png"></a>')
    (root/'gallery.html').write_text('\n'.join(gallery))
    (root/'COMPLETE.json').write_text(json.dumps(dict(status='DIAGNOSTICS_COMPLETE',parameter_updates=0,model_windows=64,report=str(root/'REPORT.md'),gallery=str(root/'gallery.html')),indent=2)+'\n')
    print(json.dumps(dict(status='DIAGNOSTICS_COMPLETE',model_windows=64,parameter_updates=0)))

if __name__=='__main__':main()
