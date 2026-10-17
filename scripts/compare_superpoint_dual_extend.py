"""Compare a model's epoch4/6/8 with validated identical-cohort official tables."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,csv,json
from scripts.dual_stream_evaluation_checks import check_complete,table_row
from scripts.format_object_locus_posefree_siu3r_table import FIELDS
from scripts.superpoint_dual_continuation import VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--variant',choices=tuple(VARIANTS),default='dual_superpoint');a=p.parse_args();root=a.run_root
    spec=VARIANTS[a.variant];label=spec['label'];baseline=root.parent/spec['source_run']
    rows={};cohort=None
    for model_label,run,epoch in [(label+'_epoch4',baseline,4),(label+'_epoch6',root/'official_epoch6',6),(label+'_epoch8',root/'official_epoch8',8)]:
        row,export=check_complete(run)
        identity=sorted((r['name'],r['source_index'],tuple(r['frame_ids']),r['input_rgb_sha256']) for r in export['records'])
        if cohort is not None and cohort!=identity:raise ValueError('Continuation evaluation cohort/RGB differs')
        cohort=identity
        if epoch!=4:
            cp=json.loads((run/'evaluation/checkpoint_identity.json').read_text())
            assert cp['evaluated_epoch']==epoch and cp['parameter_updates']==0 and cp['meta']['continuation_policy']==spec['policy']
            assert cp['meta']['architecture']==spec['architecture']
            assert cp['meta']['continuation_sha256']==json.loads((root/'preflight.json').read_text())['source_sha256']
        rows[model_label]=row
    old=table_row('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation/siu3r_table1.csv')
    output=root/'comparison_epoch4_6_8';output.mkdir(exist_ok=True)
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['model']+list(FIELDS));writer.writerow(['old_C']+old)
        for model_label,row in rows.items():writer.writerow([model_label]+row)
        for model_label in [label+'_epoch6',label+'_epoch8']:writer.writerow([model_label+'_minus_'+label+'_epoch4']+['' if x=='' else float(y)-float(x) for x,y in zip(rows[label+'_epoch4'],rows[model_label])])
    lines=['| model | '+' | '.join(FIELDS)+' |','| '+' | '.join(['---']*14)+' |']
    for model_label,row in [('old_C',old)]+list(rows.items()):lines.append('| '+model_label+' | '+' | '.join(row)+' |')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n\nSame official128 RGB/cohort, 11 metrics and 2 blank text columns. Additional training uses preserved Adam/RNG with an explicit LR rewarm; not a pure epoch-only intervention.\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(status=label.upper()+'_EPOCH4_6_8_OFFICIAL_COMPARISON_COMPLETE',same_128_rgb_cohort=True,parameter_updates=0),indent=2)+'\n')


if __name__=='__main__':main()
