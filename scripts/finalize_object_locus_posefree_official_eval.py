"""Validate and record a completed official evaluation; no inference or scoring."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def finalize(root):
    report, job = read(root/'metrics.json'), read(root/'job.json')
    assert report['status'] == 'EVALUATED'
    meta = report['training_checkpoint_metadata']
    assert meta['epoch'] == job['checkpoint_epoch']
    assert meta['completed_updates'] == job['completed_updates']
    assert meta['completed_exposures'] == job['completed_exposures']
    assert report['checkpoint_sha256'] == job['checkpoint_sha256']
    assert meta['geometry_quality_policy'] == 'monitor_v1'
    assert report['official_commit'] == job['official_siu3r_commit']
    assert report['inference_code_sha'] == report['evaluation_code_sha'] == job['execution_git_sha']
    state = subprocess.check_output([
        'sacct', '-X', '-j', str(job['job_id']), '-n', '-P',
        '--format=JobID,State,ExitCode,Elapsed,Start,End'], text=True).strip().splitlines()
    state = [row.split('|') for row in state if row.split('|')[0] == str(job['job_id'])]
    assert len(state) == 1
    jid, status, exit_code, elapsed, start, end = state[0][:6]
    assert status == 'COMPLETED' and exit_code == '0:0', state
    manifest = read(Path('/space/mawb/SIU3R/data/scannet/val_pair.json'))
    expected = {r['scan']+'_context'+'_'.join(map(str, r['context_ids'])) for r in manifest}
    records, geometry, reasons = [], [], Counter()
    ranks = sorted(root.glob('rank*'))
    assert len(ranks) == job['scene_shards'] == 7
    for rank in ranks:
        export, receipt = read(rank/'export_complete.json'), read(rank/'score_complete.json')
        assert export['status'] == 'EXPORTED' and receipt['status'] == 'COMPLETE'
        assert receipt['segmentation_device'].startswith('cuda')
        assert read(rank/'checkpoint_metadata.json') == meta
        contract = read(rank/'official_merge_contract_cuda.json')
        assert contract['status'] == 'PASS' and contract['official_commit'] == report['official_commit']
        assert receipt['windows'] == len(export['records'])
        records.extend(export['records'])
        for line in (rank/f'geometry_monitor_rank{int(rank.name[4:])}.jsonl').read_text().splitlines():
            row = json.loads(line)
            assert row['fit_status'] == 'VALID' and row['geometry_quality_policy'] == 'monitor_v1'
            geometry.append(row)
            reasons.update({reason for view in row['quality_warning_reasons'] for reason in view})
    names = [row['name'] for row in records]
    assert len(names) == len(set(names)) == 1860 and set(names) == expected
    assert {row['manifest_index'] for row in records} == set(range(1860))
    assert report['windows'] == 1860 and report['unique_scenes'] == 312
    assert len({name.split('_context')[0] for name in names}) == 312
    assert len(geometry) == 1860
    assert {r['manifest_index'] for r in geometry} == set(range(1860))
    by_index = {r['manifest_index']: r for r in records}
    for row in geometry:
        record = by_index[row['manifest_index']]
        assert row['scene'] == record['name'].split('_context')[0]
        assert row['context_ids'] + row['novel_ids'] == record['frame_ids']
    counts = Counter(row['quality_status'] for row in geometry)
    assert set(counts) <= {'OK', 'WARNING'}
    scopes = report['scopes']
    for scope, count in [('context', 3720), ('target-all', 11160), ('true-novel', 7440)]:
        assert scopes[scope]['images'] == count
        assert scopes[scope]['mIoU_t'] is None
        assert all(math.isfinite(scopes[scope][key]) for key in ('absrel','rmse','psnr','ssim','lpips','mIoU_s','mAP','PQ'))
    for key in ('absrel','rmse','psnr','ssim','lpips'):
        assert abs((scopes['context'][key]+2*scopes['true-novel'][key])/3-scopes['target-all'][key]) < 1e-10
    for key in ('context_miou','context_pq'):
        assert abs(report['official_segmentation']['all'][key]-report['official_segmentation']['novel'][key]) < 1e-7
    for key in ('map', 'map_50', 'map_75'):
        assert abs(report['official_segmentation']['all']['context_map'][key]-report['official_segmentation']['novel']['context_map'][key]) < 1e-7
    table = list(csv.reader((root/'siu3r_table1.csv').open()))
    assert len(table) == 2 and all(len(row) == 13 for row in table)
    assert table[1][8] == table[1][12] == ''
    coverage = {'status':'PASS','windows':1860,'unique_scenes':312,
                'context_images':3720,'target_all_images':11160,'true_novel_images':7440,
                'omitted_windows':0,'duplicate_windows':0,
                'geometry_quality_counts':{'windows':1860,**dict(counts)},
                'geometry_warning_reasons':dict(reasons),'all_fits_numerically_valid':True,
                'geometry_quality_policy':'monitor_v1','score_status':'EVALUATED'}
    write(root/'coverage_and_geometry.json', coverage)
    write(root/'final_validation.json', {'status':'PASS','windows':1860,'unique_scenes':312,
          'all_metrics_finite':True,'all_scope_image_counts_correct':True,
          'native_reconstruction_mean_weighting_verified':True,
          'context_segmentation_identical_in_all_and_novel_exports':True,
          'real_gpu_cpu_contract_shards_passed':len(list(root.glob('rank*'))),
          'final_scoring_job_id':job['job_id'],'checkpoint_epoch':meta['epoch'],
          'text_metric_status':'NOT_TRAINED','metric_columns':13,
          'inference_code_sha':report['inference_code_sha'],'scoring_code_sha':report['evaluation_code_sha'],
          'metrics_sha256':hashlib.sha256((root/'metrics.json').read_bytes()).hexdigest()})
    completion = {'status':'EVALUATED','job_id':job['job_id'],'slurm_state':status,'exit_code':exit_code,
                  'completed_at':end+' Asia/Shanghai','elapsed':elapsed,'windows':1860,'unique_scenes':312,
                  'training_checkpoint_epoch':meta['epoch'],'training_completed_updates':meta['completed_updates'],
                  'training_completed_exposures':meta['completed_exposures'],
                  'inference_code_sha':report['inference_code_sha'],'scoring_code_sha':report['evaluation_code_sha'],
                  'metrics_path':str(root/'metrics.json'),'summary_path':str(root/'summary.md'),
                  'geometry_quality_policy':'monitor_v1','geometry_warning_windows':counts['WARNING'],
                  'mIoU_t_status':'NOT_TRAINED','optimizer_updates':0}
    write(root/'COMPLETE.json', completion)
    job.update(status='EVALUATED', slurm_state=status, exit_code=exit_code, completed_at=completion['completed_at'])
    write(root/'job.json', job)
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.root), ensure_ascii=False, indent=2))
