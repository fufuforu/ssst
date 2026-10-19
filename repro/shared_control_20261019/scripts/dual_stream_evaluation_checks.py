"""Validate 13-column/11-value artifacts, or assemble the schedule comparison."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.format_object_locus_posefree_siu3r_table import FIELDS,table_values
from scripts.invoke_siu3r_official_evaluator import SIU3R_COMMIT


def table_row(path):
    with Path(path).open() as stream:rows=list(csv.reader(stream))
    if len(rows)!=2 or rows[0]!=list(FIELDS) or len(rows[1])!=13:raise ValueError('Official table schema differs')
    row=rows[1]
    if row[8]!='' or row[12]!='':raise ValueError('Text mIoU must remain blank')
    if sum(x!='' for x in row)!=11 or any(not math.isfinite(float(x)) for x in row if x!=''):raise ValueError('Expected 11 finite metrics')
    return row


def check_complete(root):
    root=Path(root)
    if json.loads((root/'COMPLETE.json').read_text())['status']!='TRAINING_AND_HELDOUT128_OFFICIAL_EVAL_COMPLETE':raise ValueError('Run incomplete')
    exported=json.loads((root/'evaluation/export_complete.json').read_text())
    if exported['status']!='EXPORTED' or len(exported['records'])!=128 or len({r['name'] for r in exported['records']})!=128:raise ValueError('Incomplete heldout128 cohort')
    row=table_row(root/'evaluation/siu3r_table1.csv')
    report=json.loads((root/'evaluation/metrics.json').read_text())
    if report['official_commit']!=SIU3R_COMMIT:raise ValueError('Evaluation commit differs')
    values=table_values(report)
    if any(float(x)!=y for x,y in zip(row,values) if x!=''):raise ValueError('CSV differs from native evaluation metrics')
    return row,exported


def compare(direct,staged,output):
    a,ea=check_complete(direct);b,eb=check_complete(staged)
    def identities(exported):return sorted((r['name'],r['source_index'],tuple(r['frame_ids']),r['input_rgb_sha256']) for r in exported['records'])
    if identities(ea)!=identities(eb):raise ValueError('Comparison cohorts/RGB differ')
    old=table_row('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01/object_support/evaluation/siu3r_table1.csv')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    delta=['' if x=='' else float(y)-float(x) for x,y in zip(a,b)]
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['model']+list(FIELDS))
        writer.writerows([['old_VGGT_MASt3R_C']+old,['Omega_direct_joint4']+a,['Omega_adapt2_joint4']+b,['adapt2_joint4_minus_direct_joint4']+delta])
    lines=['| model | '+' | '.join(FIELDS)+' |','| '+' | '.join(['---']*14)+' |']
    for label,row in [('old_VGGT_MASt3R_C',old),('Omega_direct_joint4',a),('Omega_adapt2_joint4',b),('staged minus direct',delta)]:
        lines.append('| '+label+' | '+' | '.join(str(x) for x in row)+' |')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n\n13 columns, 11 numeric metrics; text mIoU blank. Official target includes context + four extra views. Shared short306 source; staged run adds 2 full reconstruction epochs, not equal compute.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-complete',type=Path)
    parser.add_argument('--direct',type=Path);parser.add_argument('--staged',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.check_complete:check_complete(args.check_complete);print('COMPLETE_VALID_11_METRICS')
    else:
        if not all((args.direct,args.staged,args.output)):parser.error('Comparison requires direct, staged and output')
        compare(args.direct,args.staged,args.output)
