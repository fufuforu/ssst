"""Five independent metric cases, then paired A8/C8 and descriptive epoch4 comparison."""
from pathlib import Path
import sys,json,os,argparse,subprocess,math,csv
from datetime import timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from scripts.gs_quality_core import SOURCE_SHA,write_json,digest
from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_REPO,SIU3R_COMMIT
from scripts.score_dual_stream_v1 import itemize,json_safe
from scripts.score_fixed_query_diagnostics import flatten
ARMS=['Native','A_Rank','C_Rank','A_Cal','C_Cal']
BASE4=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/frozen_baseline4_gs_quality_20261024_v4_cachefinalized')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();run=a.run_root
    proof=json.loads((run/'eval/export_complete.json').read_text());fit=json.loads((run/'FIT_COMPLETE.json').read_text())
    if proof['status']!='PASSED' or proof['windows']!=1860 or not proof['cache_signatures_exact'] or not proof['same_forward_two_heads']:raise ValueError('Export incomplete')
    if fit['status']!='PASSED' or set(fit['arms'])!={'A','C'} or digest(run/'official_predictions.pt')!=fit['official_predictions_sha256']:raise ValueError('Predictions changed')
    if subprocess.check_output(['git','-C',SIU3R_REPO,'rev-parse','HEAD'],text=True).strip()!=SIU3R_COMMIT or subprocess.check_output(['git','-C',SIU3R_REPO,'status','--porcelain','--','src'],text=True).strip():raise ValueError('Scorer changed')
    local=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=5:raise ValueError('Five scoring cases use five GPUs')
    torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=timedelta(hours=3))
    arm=ARMS[rank]
    raw=itemize(evaluate(run/'eval'/arm/'all',device=f'cuda:{local}',image_quality=arm=='Native',depth_quality=arm=='Native'))
    write_json(run/'eval'/arm/'RAW_METRICS.json',dict(status='PASSED',official_result=json_safe(raw),official_commit=SIU3R_COMMIT))
    dist.barrier()
    if rank==0:
        source=json.loads((run/'configuration.json').read_text());scores={};native=json.loads((run/'eval/Native/RAW_METRICS.json').read_text())['official_result']
        for arm in ARMS:
            result=json.loads((run/'eval'/arm/'RAW_METRICS.json').read_text())['official_result']
            values=flatten(result,native)
            if any(not math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite metric')
            scores[arm]=values
            if arm.endswith('_Rank'):
                for key in ['context_miou','target_miou','context_pq','target_pq','context_mar_100','target_mar_100']:
                    if abs(values[key]-scores['Native'][key])>1e-8:raise ValueError('Ranking changed score-independent metric')
            write_json(run/'eval'/arm/'metrics.json',dict(status='PASSED',metrics=values,official_native_result=result,official_commit=SIU3R_COMMIT))
        deltas={mode:{k:scores['C_'+mode][k]-scores['A_'+mode][k] for k in scores['Native']} for mode in ['Rank','Cal']}
        prior=json.loads((BASE4/'RESULTS.json').read_text())['metrics']
        history={arm:{k:scores[arm][k]-prior[arm][k] for k in scores[arm]} for arm in ARMS}
        reference=json.loads(Path(source['historical_baseline8_metrics']).read_text())['official_native_result']
        ref=flatten(reference,reference)
        write_json(run/'RESULTS.json',dict(status='COMPLETE',metrics=scores,C8_minus_A8=deltas,
          epoch8_minus_epoch4_same_head_type=history,Native8_minus_historical_Baseline8={k:scores['Native'][k]-ref[k] for k in ref},
          source_sha256=SOURCE_SHA,backbone_updates=0,heads=['A8','C8'],head_selection='dev MSE only',formal_used_for_selection=False,
          target_scope='all6 including Context',single_seed=True,reconstruction_measured_once=True,
          caveat='A8/C8 share exact forward. Epoch4 comparison is cross-checkpoint and cross-run, not solely GS contribution; no epoch8 shuffled-GS arm.'))
        with (run/'metrics_and_deltas.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['arm']+list(scores['Native']));writer.writeheader()
            for arm,values in scores.items():writer.writerow(dict(arm=arm,**values))
            for mode,values in deltas.items():writer.writerow(dict(arm='C8_minus_A8_'+mode,**values))
            for arm,values in history.items():writer.writerow(dict(arm=arm+'_epoch8_minus_epoch4',**values))
        write_json(run/'COMPLETE.json',dict(status='PASSED',windows=1860,scenes=312,heads=['A8','C8'],outputs=ARMS,backbone_updates=0))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
