"""Summarize persisted global path observations and disclose replay drift."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    reports={}
    for arm in 'ABC':
        root=a.root/arm
        complete=json.loads((root/'COMPLETE.json').read_text())
        assert complete['status']=='GRADIENT_REPLAY_COMPLETED' and complete['replayed_updates']==141
        ranks=[[json.loads(x) for x in (root/f'replay_rank{i}.jsonl').read_text().splitlines()] for i in range(4)]
        assert all([x['completed_updates'] for x in rows]==list(range(301,442)) for rows in ranks)
        observations={}
        for step in (301,400,440,441):
            saved=json.loads((root/f'global_gradients_update{step:04d}.json').read_text())
            assert saved['status']=='GLOBAL_PATHS_MEASURED' and len(saved['rank_rows'])==4
            observations[str(step)]=dict(original_norm=saved['original_norm'],replay_norm=saved['replay_norm'],
                global_paths=saved['rank_rows'][0])
        error=max(abs(x[k]-x['original_logged_values'][k]) for r in ranks for x in r for k in ('loss_recon','loss_understanding'))
        norm441=observations['441'];reproduced=abs(norm441['replay_norm']-norm441['original_norm'])<=.05*max(norm441['original_norm'],1.)
        reports[arm]=dict(source=complete['source'],observations=observations,
            max_logged_loss_absolute_replay_error=error,original_step441_norm_reproduced_within5percent=reproduced,
            original_spike_reproduction_must_be_checked_before_historical_attribution=True)
    summary=dict(status='GRADIENT_REPLAY_ANALYZED',arms=reports,total_replayed_updates=423,
        model_or_optimizer_interventions=0,formal_training_checkpoints_untouched=True,
        caveat='Replay may differ numerically; report drift rather than asserting the original spike was reproduced. Clip multiplier is not Adam parameter-update multiplier.')
    (a.root/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Saved A/B/C joint gradient replay','', '141 replayed calls per arm,301..441; original GC/clip1/AdamW/RNG. Main training unaffected.','',
        '| Arm | Original norm441 | Replay norm441 | Historical norm within5% | Max R/U loss replay error |','|---|---:|---:|---|---:|']
    for arm,r in reports.items():
        s=r['observations']['441'];lines.append(f"|{arm}|{s['original_norm']:.6f}|{s['replay_norm']:.6f}|{r['original_step441_norm_reproduced_within5percent']}|{r['max_logged_loss_absolute_replay_error']:.8g}|")
    lines+=['',summary['caveat'],'','Global per-module R/U vectors, actual pre/post clipping norms and measured parameter changes: analysis.json. No inference metrics are invented.']
    (a.root/'report.md').write_text('\n'.join(lines)+'\n')
    (a.root/'COMPLETE.json').write_text(json.dumps(dict(status='GRADIENT_REPLAY_ANALYZED',replayed_updates=423,main_training_untouched=True),indent=2)+'\n')


if __name__=='__main__':main()
