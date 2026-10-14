"""Audit completed classification controls and write one official13-column table."""
import argparse
import csv
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.train_vggt_classification_readback_v1 import ARMS,slots,SOURCE_SHA
from scripts.format_object_locus_posefree_siu3r_table import FIELDS,table_values
from scripts.analyze_vggt_feedback_factorial_v1 import markdown
from scripts.eval_vggt_joint_short_controls_v1 import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    output=a.root/'analysis';output.mkdir(parents=True,exist_ok=False)
    table=[];understanding={};reference=json.loads((a.root/'reference/metrics.json').read_text())
    assert json.loads((a.root/'reference/COMPLETE.json').read_text())['status']=='REFERENCE128_EVALUATED'
    # Use the same canonical image order as the four-shard native merge. This
    # avoids score-tie ordering differences for the CPU-only reference.
    import torch
    from scripts.posefree_official_state_helpers import merge
    torch.set_num_threads(4)
    blob=torch.load(a.root/'reference/official_states.pt',map_location='cpu',mmap=True,weights_only=False)['all']
    assert len(blob['names'])==len(set(blob['names']))==128
    reference['official_segmentation']['all']=merge([blob],a.root/'reference')
    write(output/'reference_canonical_metrics.json',reference)
    table.append(('A frozen reference',table_values(reference)))
    baseline=json.loads(Path('/space/mawb/ssst/group_plus/vggt_full_b005_bottleneck_recheck_v1/review_final_20261014/results.json').read_text())['matched_context']['full4172']
    init=set();source_set=set();peaks={}
    for label,arm in zip('BCD',ARMS):
        training=a.root/arm/'training';evaluation=a.root/arm/'evaluation'
        done=json.loads((training/'COMPLETE.json').read_text());assert done['status']=='CLASSIFICATION_TRAINED' and done['completed_updates']==300
        assert json.loads((evaluation/'COMPLETE.json').read_text())['status']=='CLASSIFICATION_HELDOUT128_EVALUATED'
        manifest=json.loads((training/'run_manifest.json').read_text())
        source_set.add(manifest['source']['checkpoint_sha256']);init.update(x['source_classifier_sha256'] for x in manifest['rank_initialization'])
        rows=[]
        for rank in range(4):
            logs=[json.loads(x) for x in (training/f'training_rank{rank}.jsonl').read_text().splitlines()]
            assert len(logs)==300 and [r['completed_updates'] for r in logs]==list(range(1,301))
            assert all(r['source_indices']==slots(i,rank) and r['frozen_core_unchanged'] and r['actual_feedback_beta']==.05 for i,r in enumerate(logs))
            rows+=logs
        peaks[arm]=max(r['peak_allocated'] for r in rows)
        metrics=json.loads((evaluation/'metrics.json').read_text());values=table_values(metrics)
        assert all(abs(x-y)<1e-12 for x,y in zip(values[:5],table_values(reference)[:5]))
        table.append((label+' '+arm,values))
        understanding[arm]=dict(matched_query=json.loads((evaluation/'matched_query_diagnostics.json').read_text()),
            ranking=json.loads((evaluation/'ranking_diagnostics.json').read_text()),
            native_table=values,delta_vs_reference=[None if x is None else y-x for x,y in zip(table_values(reference),values)])
    assert source_set=={SOURCE_SHA} and len(init)==1
    with (output/'siu3r_comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(('experiment',)+FIELDS)
        for label,values in table:writer.writerow((label,)+tuple(values))
    (output/'siu3r_comparison.md').write_text(markdown(table)+'\n')
    write(output/'results.json',dict(status='CLASSIFICATION_CONTROLS_AUDITED',source_checkpoint_sha256=SOURCE_SHA,
        frozen_reference_matched_queries=baseline,arms=understanding,all_R_metrics_exactly_unchanged=True,
        same2400_training_windows_and_same128_heldout_verified=True,initial_classifier_identical=True,
        training_updates_per_arm=300,peak_allocated_by_arm=peaks,
        caveat='Interventions isolate classifier calibration vs added scene/object support on fixed core/masks. No unique-module fault or multi-seed significance claim; GT scores are diagnosis only.'))
    write(a.root/'COMPLETE.json',dict(status='CLASSIFICATION_CONTROLS_EVALUATED',arms=['reference',*ARMS],training_updates=900,heldout_windows_per_arm=128))


if __name__=='__main__':main()
