"""Score baseline and learned IoU ranking with the pinned SIU3R evaluator."""
import argparse,csv,json,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_REPO,SIU3R_COMMIT
from scripts.score_dual_stream_v1 import itemize,json_safe
from scripts.score_fixed_query_diagnostics import flatten
from scripts.frozen_iou_head_core import write_json,digest

REFERENCE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/fixed_query_oracle_diagnostics_20261023_v4_paired')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();run=args.run_root
    exported=json.loads((run/'eval/export_complete.json').read_text())
    if exported['windows']!=1860 or exported['backbone_updates']!=0 or not all(r['ranking_only_verified'] for r in exported['records']):
        raise ValueError('Frozen ranking export incomplete')
    if subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()!=SIU3R_COMMIT or subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip():
        raise ValueError('Official evaluator identity differs')
    prior=json.loads((REFERENCE/'diagnostic_summary.json').read_text());oracle={x['arm']:x for x in prior['metrics']}
    reference_native=json.loads((REFERENCE/'E0/evaluation/metrics.json').read_text())['official_native_result']
    scores={}
    for arm in ('Baseline','IoU_MLP'):
        raw=itemize(evaluate(run/'eval'/arm/'evaluation/all',device='cuda:0',image_quality=False,depth_quality=False))
        metrics=flatten(raw,reference_native)
        if any(not math.isfinite(x) or x<0 for x in metrics.values()):raise ValueError('Invalid official metric')
        scores[arm]=metrics
        write_json(run/'eval'/arm/'evaluation/metrics.json',dict(status='SCORED',windows=1860,scenes=312,backbone_updates=0,
                   head_sha256=exported['head_sha256'],official_commit=SIU3R_COMMIT,metrics=metrics,official_native_result=json_safe(raw),
                   reconstruction_metrics_reused=True,prediction_score='original' if arm=='Baseline' else 'predicted_iou_direct'))
    delta={k:scores['IoU_MLP'][k]-v for k,v in scores['Baseline'].items()}
    for k in ('context_miou','context_pq','target_miou','target_pq'):
        if abs(delta[k])>1e-10:raise ValueError('Reranking changed score-independent segmentation metrics')
    with (run/'metrics_and_deltas.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['arm']+list(delta));writer.writeheader()
        for arm,row in scores.items():writer.writerow(dict(arm=arm,**row))
        writer.writerow(dict(arm='IoU_MLP_minus_Baseline',**delta))
    audits=[json.loads(p.read_text()) for p in (run/'eval/quality_audit').glob('*.json')]
    if len(audits)!=1860:raise ValueError('Incomplete held-out quality diagnostics')
    errors=[p-y for a in audits for p,y in zip(a['predicted'],a['context_gt_iou'])]
    history={k:sum(a['historical_baseline_audit'][k] for a in audits) for k in ('historical_differing_pngs','historical_differing_pixels')}
    history['metric_delta_from_prior_E0']={k:scores['Baseline'][k]-oracle['E0'][k] for k in delta}
    report=dict(status='COMPLETE',metrics=scores,delta=delta,backbone_updates=0,head_fit=json.loads((run/'head/FIT_COMPLETE.json').read_text()),
                heldout_quality=dict(queries=len(errors),mse=sum(x*x for x in errors)/max(1,len(errors)),mae=sum(abs(x) for x in errors)/max(1,len(errors))),
                prior_context_only_oracle_ranking={k:oracle['E2_rank'][k] for k in delta},historical_baseline_audit=history,
                target_scope='all6_including_context',recall_definition='COCO AR, IoU .50:.05:.95',official_evaluation_used_for_model_selection=False,
                caution='One frozen base, one head, internal scene dev selection; quality labels diagnostic only on official evaluation.')
    write_json(run/'RESULTS.json',report)
    write_json(run/'COMPLETE.json',dict(status='FROZEN_BACKBONE_IOU_MLP_COMPLETE',windows=1860,scenes=312,backbone_updates=0,
               head_sha256=exported['head_sha256'],head_updated=True,ranking_only=True))


if __name__=='__main__':main()
