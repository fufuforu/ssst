"""Fresh A/B/C/D, ranking and fixed-gamma pixel-calibration official results."""
from pathlib import Path
import sys,argparse,json,math,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gs_quality_core import *
from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_REPO,SIU3R_COMMIT
from scripts.score_dual_stream_v1 import itemize,json_safe
from scripts.score_fixed_query_diagnostics import flatten

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();run=a.run_root
    proof=json.loads((run/'eval/export_complete.json').read_text())
    if proof['status']!='PASSED' or proof['windows']!=1860 or not proof['cache_signatures_exact']:raise ValueError('Export gate failed')
    if subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()!=SIU3R_COMMIT or subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip():raise ValueError('Official scorer changed')
    scores={}
    for arm in ['Native']+[a+'_'+mode for mode in ('Rank','Cal') for a in ('A','B','C','D')]:
        raw=itemize(evaluate(run/'eval'/arm/'all',device='cuda:0',image_quality=arm=='Native',depth_quality=arm=='Native'))
        scores[arm]=flatten(raw,raw if arm=='Native' else scores['Native'])
        if any(not math.isfinite(v) for v in scores[arm].values()):raise ValueError('Nonfinite official score')
        if arm.endswith('_Rank'):
            for k in ('context_miou','target_miou','context_pq','target_pq','context_mar_100','target_mar_100'):
                if abs(scores[arm][k]-scores['Native'][k])>1e-8:raise ValueError('Ranking changed score-independent metric')
        write_json(run/'eval'/arm/'metrics.json',dict(metrics=scores[arm],official_native_result=json_safe(raw),official_commit=SIU3R_COMMIT))
    deltas={mode:{f'{x}_minus_{y}':{k:scores[x+'_'+mode][k]-scores[y+'_'+mode][k] for k in scores['Native']} for x,y in [('C','A'),('C','D'),('B','A')]} for mode in ('Rank','Cal')}
    history=json.loads((PARENT/'RESULTS.json').read_text())['metrics']['IoU_MLP']
    write_json(run/'RESULTS.json',dict(status='COMPLETE',metrics=scores,deltas=deltas,historical_A=history,
        fresh_A_minus_historical_A={k:scores['A_Rank'][k]-history[k] for k in history},source_sha256=SOURCE_SHA,official_commit=SIU3R_COMMIT,
        single_seed=True,heads_gt_free=True,gamma=.25,target_scope='all6',historical_A_used_for_selection=False,
        caveat='Fresh A is the causal token control; historical drift is reported separately. D is a within-cohort unlabeled cross-scene negative control, not a deployable method.'))
    write_json(run/'COMPLETE.json',dict(status='PASSED',windows=1860,arms=['A','B','C','D'],backbone_updates=0))
if __name__=='__main__':main()
