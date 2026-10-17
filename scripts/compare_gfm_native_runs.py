"""Compare native official scores, preserving any invalid values as NaN."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.score_dual_stream_v1 import native_table_values
from scripts.format_object_locus_posefree_siu3r_table import FIELDS


def validate_training_receipt(trained,meta=None):
    if 'progress' in trained:
        if not meta or meta.get('short_validation') or meta.get('epochs')!=4 or not meta.get('architecture','').startswith('OMEGA_3D_SUPERPOINT_'):
            raise ValueError('Not a completed full-data superpoint run')
        if trained.get('status')!='TRAINING_COMPLETE' or trained['progress']!=dict(epoch=4,update_in_epoch=0,full_updates=4172):
            raise ValueError('Training incomplete')
    elif trained.get('full_updates')!=4172 or not trained.get('frozen_unchanged'):
        raise ValueError('Training incomplete')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='append',required=True,help='label=/absolute/run/root')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    rows={};cohort=None;status={}
    reference=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation/siu3r_table1.csv')
    with reference.open() as stream:
        reader=csv.reader(stream);assert next(reader)==list(FIELDS);rows['old_C']=[None if x=='' else float(x) for x in next(reader)]
    for entry in args.run:
        label,path=entry.split('=',1)
        if label in rows:raise ValueError('Duplicate model label')
        root=Path(path);trained=json.loads((root/'TRAINING_COMPLETE.json').read_text())
        meta=json.loads((root/'STARTED.json').read_text()) if 'progress' in trained else None
        validate_training_receipt(trained,meta)
        exported=json.loads((root/'evaluation/export_complete.json').read_text())
        if exported['status']!='EXPORTED' or len(exported['records'])!=128:raise ValueError('Incomplete export')
        identity=sorted((r['name'],r['source_index'],tuple(r['frame_ids']),r['input_rgb_sha256']) for r in exported['records'])
        if cohort is not None and identity!=cohort:raise ValueError('Different RGB/cohort identities')
        cohort=identity
        rows[label]=native_table_values(json.loads((root/'evaluation/all/results.json').read_text()))
        invalid=[k for k,v in zip(FIELDS,rows[label]) if v is not None and not math.isfinite(v)]
        status[label]=dict(run_root=str(root),invalid_required_metrics=invalid,valid_required_count=11-len(invalid))
    args.output.mkdir(parents=True,exist_ok=True)
    def cell(value):return '' if value is None else ('NaN' if not math.isfinite(value) else value)
    with (args.output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['model']+list(FIELDS))
        for label,values in rows.items():writer.writerow([label]+[cell(x) for x in values])
    lines=['| model | '+' | '.join(FIELDS)+' |','| '+' | '.join(['---']*14)+' |']
    for label,values in rows.items():lines.append('| '+label+' | '+' | '.join('' if x is None else ('NaN' if not math.isfinite(x) else f'{x:.5f}') for x in values)+' |')
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n\nOfficial native scores, same heldout128. Invalid depth stays NaN; text mIoU stays blank. No frame exclusion or metric substitution.\n')
    (args.output/'comparison_status.json').write_text(json.dumps(dict(models=status,same_128_rgb_cohort=True,parameter_updates=0),indent=2)+'\n')

if __name__=='__main__':main()
