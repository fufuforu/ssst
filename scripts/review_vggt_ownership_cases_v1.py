"""Select interpretable frozen-output cases, including GT recovery by duplicates."""
import argparse
import csv
import json
from pathlib import Path

from scripts.analyze_vggt_ownership_v1 import case_gallery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.analysis / 'reviewed_gallery'
    output.mkdir()
    rows = list(csv.DictReader((args.analysis/'GT_aligned_instances.csv').open()))
    cases = [dict(stratum=label, name=name, GT_id=iid) for label, name, iid in [
        ('good_mask_rejected_GT_not_rescued', 'scene0050_00_context1441_1460', 2),
        ('wrong_class_despite_good_ownership', 'scene0144_01_context1068_1086', 9),
        ('duplicate_query_competition_GT_survives', 'scene0231_02_context2510_2543', 3),
        ('visual_rescue_weak_anchor', 'scene0549_00_context376_392', 12),
        ('small_object_poor_mask_global_write', 'scene0277_01_context36_108', 5),
        ('joint_drift_no_object_despite_stable_mask', 'scene0553_00_context607_629', 6),
        ('correct_class_good_mask_with_global_read', 'scene0658_00_context6_68', 1),
    ]]
    records = {}
    for arm in ('C', 'baseline300'):
        records[arm] = {}
        for case in cases:
            paths = list((args.root/arm).glob(f'rank*/{case["name"]}.json'))
            assert len(paths)==1
            records[arm][case['name']] = (json.loads(paths[0].read_text()), paths[0])
    case_gallery(args.root, output, cases, records)
    for case in cases:
        case['paired_measurements'] = [r for r in rows if r['name']==case['name'] and int(r['GT_id'])==case['GT_id']]
    (output/'cases.json').write_text(json.dumps(cases, indent=2)+'\n')
    (output/'gallery.md').write_text('# 固定权重典型案例\n\n'
        '按GT实例对齐。pred/GT为18类thing的0起始类别；pred=18表示no-object。'
        '绿色TP、红色FP、蓝色FN；每个ownership面板独立色阶，不能横向比较亮度。'
        '图内IoU为所选一帧；配对JSON为两context帧联合IoU。\n\n' +
        '\n\n'.join(f'## {c["stratum"]}\n\n{c["name"]}，GT {c["GT_id"]}\n\n![case]({c["figure"]})' for c in cases)+'\n')
    print(output)


if __name__=='__main__':
    main()
