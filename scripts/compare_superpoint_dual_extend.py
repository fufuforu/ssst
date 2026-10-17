"""Compare A epoch4/6/8 through validated identical-cohort official metric tables."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,csv,json
from scripts.dual_stream_evaluation_checks import check_complete,table_row
from scripts.format_object_locus_posefree_siu3r_table import FIELDS
from scripts.superpoint_dual_continuation import POLICY


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();root=a.run_root
    baseline=root.parent/'attempt05_legacy_dual_superpoint_full_node13_8gpu'
    rows={};cohort=None
    for label,run,epoch in [('A_epoch4',baseline,4),('A_epoch6',root/'official_epoch6',6),('A_epoch8',root/'official_epoch8',8)]:
        row,export=check_complete(run)
        identity=sorted((r['name'],r['source_index'],tuple(r['frame_ids']),r['input_rgb_sha256']) for r in export['records'])
        if cohort is not None and cohort!=identity:raise ValueError('Continuation evaluation cohort/RGB differs')
        cohort=identity
        if epoch!=4:
            cp=json.loads((run/'evaluation/checkpoint_identity.json').read_text())
            assert cp['evaluated_epoch']==epoch and cp['parameter_updates']==0 and cp['meta']['continuation_policy']==POLICY
            assert cp['meta']['continuation_sha256']==json.loads((root/'preflight.json').read_text())['source_sha256']
        rows[label]=row
    old=table_row('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation/siu3r_table1.csv')
    output=root/'comparison_epoch4_6_8';output.mkdir(exist_ok=True)
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['model']+list(FIELDS));writer.writerow(['old_C']+old)
        for label,row in rows.items():writer.writerow([label]+row)
        for label in ['A_epoch6','A_epoch8']:writer.writerow([label+'_minus_A_epoch4']+['' if x=='' else float(y)-float(x) for x,y in zip(rows['A_epoch4'],rows[label])])
    lines=['| model | '+' | '.join(FIELDS)+' |','| '+' | '.join(['---']*14)+' |']
    for label,row in [('old_C',old)]+list(rows.items()):lines.append('| '+label+' | '+' | '.join(row)+' |')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n\nSame official128 RGB/cohort, 11 metrics and 2 blank text columns. Additional training uses preserved Adam/RNG with an explicit LR rewarm; not a pure epoch-only intervention.\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status='A_EPOCH4_6_8_OFFICIAL_COMPARISON_COMPLETE',same_128_rgb_cohort=True,parameter_updates=0),indent=2)+'\n')


if __name__=='__main__':main()
