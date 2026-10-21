"""Compare completed epoch 4/8 official evidence without inference or scoring."""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.format_object_locus_posefree_siu3r_table import FIELDS, HEADERS, PRECISION, PROTOCOL_NOTE, table_values


def load(root):
    report = json.loads((root / 'metrics.json').read_text())
    assert report['status'] == 'EVALUATED'
    assert report['windows'] == 1860 and report['unique_scenes'] == 312
    records, scores = {}, {}
    for rank in sorted(root.glob('rank*')):
        done = json.loads((rank / 'export_complete.json').read_text())
        for row in done['records']:
            name = row['name']
            assert name not in records
            records[name] = row
            scores[name] = json.loads((rank / 'all' / name / 'render_scores.json').read_text())
    assert len(records) == 1860
    return report, records, scores


def compare(root4, root8, output):
    a, records4, scores4 = load(root4)
    b, records8, scores8 = load(root8)
    assert a['training_checkpoint_metadata']['epoch'] == 4
    assert b['training_checkpoint_metadata']['epoch'] == 8
    assert a['official_commit'] == b['official_commit']
    for key in ('manifest_sha256', 'cohort_source_sha256', 'vggt_revision', 'geometry_quality_policy', 'shards'):
        assert a['export_identity'][key] == b['export_identity'][key], key
    assert set(records4) == set(records8)
    for name in records4:
        assert records4[name]['frame_ids'] == records8[name]['frame_ids']
        assert records4[name]['manifest_index'] == records8[name]['manifest_index']
    values4, values8 = table_values(a), table_values(b)
    delta = [None if x is None else y-x for x, y in zip(values4, values8)]
    assert all(d is None or math.isfinite(d) for d in delta)
    paired = []
    for name in sorted(scores4):
        x = {r['item']: r['psnr'] for r in scores4[name]}
        y = {r['item']: r['psnr'] for r in scores8[name]}
        assert set(x) == set(y) and len(x) == 6
        paired.extend(y[k]-x[k] for k in sorted(x))
    assert len(paired) == 11160
    assert abs(statistics.mean(paired)-delta[2]) < 1e-10
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'siu3r_table1_epoch04_vs_epoch08.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows((values4, values8))  # exactly 13 metric columns; row order is explicit below.
    payload = {
        'status': 'COMPLETE', 'columns': list(FIELDS), 'row_order_epochs': [4, 8],
        'rows': [values4, values8], 'delta_epoch8_minus_epoch4': delta,
        'windows': 1860, 'unique_scenes': 312, 'matched_frame_observations': 11160,
        'paired_psnr_dB': {'mean_delta': statistics.mean(paired), 'median_delta': statistics.median(paired),
                           'images_improved': sum(v > 0 for v in paired), 'images_worsened': sum(v < 0 for v in paired)},
        'sources': {'epoch4': str(root4), 'epoch8': str(root8)},
        'checkpoint_sha256': {'epoch4': a['checkpoint_sha256'], 'epoch8': b['checkpoint_sha256']},
        'official_commit': a['official_commit'], 'protocol_note': PROTOCOL_NOTE,
        'optimizer_updates': 0, 'inference_repeated': False, 'scoring_repeated': False,
        'note': 'Descriptive checkpoint comparison; no bootstrap, no checkpoint reselection, no best-of-column mixing.',
    }
    (output / 'comparison.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines = ['# Epoch 4 与 epoch 8：官方 SIU3R 13 列对比', '',
             '同一训练轨迹、1860 窗口 / 312 场景；两个文本 mIoU 列留空。CSV 恰好 13 个指标列，数据行依次为 epoch4、epoch8。', '',
             '| Checkpoint | '+' | '.join(HEADERS)+' |', '|---|'+'---:|'*13]
    for epoch, values in ((4, values4), (8, values8)):
        cells = ['' if v is None else f'{v:.{p}f}' for v, p in zip(values, PRECISION)]
        lines.append(f'| epoch{epoch} | '+' | '.join(cells)+' |')
    lines += ['', PROTOCOL_NOTE, '',
              'Δ=epoch8−epoch4；AbsRel、RMSE、LPIPS 负值为改善，其余已测指标正值为改善。', '',
              '| 指标 | Δ |', '|---|---:|']
    lines += [f'| {field} | {value:+.6f} |' for field, value in zip(FIELDS, delta) if value is not None]
    lines += ['', f"逐图像配对 PSNR：均值 Δ={statistics.mean(paired):+.6f}dB，"
              f"中位数 Δ={statistics.median(paired):+.6f}dB；{sum(v>0 for v in paired)}/{len(paired)} 张改善。",
              '', '该对比只检验现有训练从第4轮到第8轮的变化，不能单独识别冻结骨干、相机误差或训练时长的因果影响。']
    (output / 'comparison.md').write_text('\n'.join(lines)+'\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch4-root', type=Path, required=True)
    parser.add_argument('--epoch8-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.epoch4_root, args.epoch8_root, args.output_root)
    print(json.dumps({'status': result['status'], 'delta_epoch8_minus_epoch4': result['delta_epoch8_minus_epoch4']}, indent=2))
