"""Audit paired full official validation of adapter U coefficients1 and0.01."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_vggt_interaction_controls_v1 import read, write
from analyze_vggt_feedback_factorial_v1 import markdown
from format_object_locus_posefree_siu3r_table import FIELDS, table_values


def audit(root):
    report = read(root/'metrics.json'); meta = report['training_checkpoint_metadata']
    assert report['status'] == 'EVALUATED' and report['optimizer_updates'] == 0
    assert (report['windows'],report['unique_scenes']) == (1860,312)
    assert report['official_commit'] == '8ea80166be76854f938e90521f1a5b688b755c87'
    assert (meta['completed_updates'],meta['completed_exposures']) == (6258,50064)
    assert (meta['epoch'],meta['phase'],meta['phase_epoch']) == (6,'frozen_joint',4)
    assert meta['geometry_quality_policy'] == 'monitor_v1'
    rows, geometry, records = [], [], []
    for i in range(4):
        rank = root/f'rank{i:02d}'
        exported, scored = read(rank/'export_complete.json'), read(rank/'score_complete.json')
        assert exported['status'] == 'EXPORTED' and scored['status'] == 'COMPLETE'
        assert scored['windows'] == len(exported['records'])
        assert scored['official_commit'] == report['official_commit']
        assert scored['segmentation_device'] == f'cuda:{i}'
        assert read(rank/'checkpoint_metadata.json') == meta
        assert read(rank/'official_merge_contract_cuda.json')['status'] == 'PASS'
        identity = exported['identity']
        for key in ('checkpoint','checkpoint_size','checkpoint_mtime_ns','manifest_sha256',
                    'cohort_source_sha256','vggt_revision','geometry_quality_policy'):
            assert identity[key] == report['export_identity'][key]
        assert identity['shard'] == i and identity['shards'] == 4 and identity['limit'] is None
        geometry.extend(json.loads(x) for x in (rank/f'geometry_monitor_rank{i}.jsonl').read_text().splitlines())
        for record in sorted(exported['records'],key=lambda x:x['manifest_index']):
            records.append((record['manifest_index'],record['name'],record['frame_ids']))
            pair = rank/'all'/record['name']
            render, depth = read(pair/'render_scores.json'), read(pair/'depth_scores.json')
            assert len(render) == len(depth) == 6
            assert sorted(x['item'] for x in render) == sorted(x['item'] for x in depth)
            rows.append((record['manifest_index'],record['name'].split('_context')[0],
                [np.mean([x[k] for x in depth]) for k in ('absrel','rmse')]+
                [np.mean([x[k] for x in render]) for k in ('psnr','ssim','lpips')]))
    records.sort(); rows.sort()
    manifest = read(Path('/space/mawb/SIU3R/data/scannet/val_pair.json'))
    assert [x[0] for x in records] == list(range(1860))
    assert len({x[1] for x in records}) == 1860
    assert {x[1] for x in records} == {x['scan']+'_context'+'_'.join(map(str,x['context_ids'])) for x in manifest}
    assert len(geometry) == 1860 and {x['manifest_index'] for x in geometry} == set(range(1860))
    assert all(x['fit_status'] == 'VALID' and x['geometry_quality_policy'] == 'monitor_v1' for x in geometry)
    assert set(x['quality_status'] for x in geometry) <= {'OK','WARNING'}
    values = np.asarray([x[2] for x in rows]); assert np.isfinite(values).all()
    table = table_values(report)
    assert np.allclose(values.mean(0),table[:5],rtol=0,atol=1e-12)
    return report, records, rows, values, dict(Counter(x['quality_status'] for x in geometry))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    old, oldrecords, oldrows, oldvalues, oldgeometry = audit(a.old)
    new, records, rows, values, geometry = audit(a.new)
    assert records == oldrecords
    for k in ('manifest_sha256','cohort_source_sha256','vggt_revision','geometry_quality_policy'):
        assert old['export_identity'][k] == new['export_identity'][k]
    oldmeta,newmeta = old['training_checkpoint_metadata'],new['training_checkpoint_metadata']
    assert oldmeta['adapted_vggt_identity']['sha256'] == newmeta['adapted_vggt_identity']['sha256']
    training=read(a.training/'COMPLETE.json')
    assert training['recipe'] == newmeta['recipe'] == 'vggt_reconstruction_adapt2_frozen_joint4_adapter_u001_v1'
    assert training['execution_git_sha'] == newmeta['git_sha']
    assert (training['new_completed_updates'],training['new_completed_exposures']) == (4172,33376)
    assert training['reused_reconstruction_boundary']['sha256'] == 'a50af40e1cf639514c2eb1d7b3127a79ecbc096d6d3a071a70d9a9fd2587750c'
    logged=[json.loads(x) for x in (a.training/'training_rank0.jsonl').read_text().splitlines()]
    assert [x['completed_updates'] for x in logged] == list(range(20,4172,20))+[4172]
    assert all(x['adapter_understanding_scale'] == .01 and x['phase'] == 'frozen_joint' for x in logged)
    assert all(np.isfinite(x[k]) for x in logged for k in ('loss_recon','loss_understanding','preclip_norm'))
    old_training = Path(old['export_identity']['checkpoint'].replace('/datapool/mawb/','/space/mawb/')).parent
    oldlogged = [json.loads(x) for x in (old_training/'training_rank0.jsonl').read_text().splitlines()]
    oldlogged = [x for x in oldlogged if x.get('phase') == 'frozen_joint']
    assert len(oldlogged) == len(logged)
    assert all(x[k] == y[k] for x,y in zip(oldlogged,logged) for k in
               ('phase_update','window_indices','group_lr','beta','understanding_weight','lr_multiplier'))
    startup = read(a.new.parent.parent/'startup_validation.json')
    assert startup['status'] == 'STARTUP_AUDIT_PASSED'
    assert len(startup['original_rank_rng_and_fresh_joint_optimizer_verified']) == 4
    assert all(x['rank_torch_rng_restored'] and x['optimizer_state_empty']
               for x in startup['original_rank_rng_and_fresh_joint_optimizer_verified'])
    scene_names=sorted({x[1] for x in rows}); assert len(scene_names) == 312
    delta=values-oldvalues
    sums=np.asarray([delta[[x[1] == name for x in rows]].sum(0) for name in scene_names])
    counts=np.asarray([sum(x[1] == name for x in rows) for name in scene_names])
    indices=np.random.default_rng(42).integers(0,312,size=(10000,312))
    # Resample whole scenes, preserving the official window/image-weighted ratio.
    samples=sums[indices].sum(1)/counts[indices].sum(1)[:,None]
    bounds=np.quantile(samples,[.025,.975],axis=0)
    oldtable,newtable=table_values(old),table_values(new)
    change={k:None if x is None else y-x for k,x,y in zip(FIELDS,oldtable,newtable)}
    bootstrap={k:dict(mean_delta=float(delta[:,i].mean()),scene_cluster_95_interval=bounds[:,i].tolist())
               for i,k in enumerate(FIELDS[:5])}
    result=dict(status='FULL_PAIRED_OFFICIAL_RESULTS_VERIFIED',windows=1860,scenes=312,images=11160,
        source_boundary_sha256=training['reused_reconstruction_boundary']['sha256'],
        old_checkpoint_sha256=old['checkpoint_sha256'],new_checkpoint_sha256=new['checkpoint_sha256'],
        same_frame_ids_and_manifest=True,all_four_score_receipts_and_merge_contracts_passed=True,
        logged_rank0_sampling_lr_and_warmups_identical=True,all_rank_boundary_rng_verified=True,
        depth_rgb_saved_scores_reproduce_official_means=True,geometry_quality_counts=dict(old=oldgeometry,new=geometry),
        coefficients=dict(old=1,new=.01),old_table=oldtable,new_table=newtable,deltas=change,
        bootstrap_reconstruction=bootstrap,bootstrap_samples=10000,seed=42,
        inference_code_sha=new['inference_code_sha'],reduction_code_sha=new['evaluation_code_sha'],
        bootstrap_scope='Dataset uncertainty only. Paired312 scene clusters, window/image-weighted ratio; global AP/PQ/IoU not averaged by scene.',
        optimizer_updates=0)
    write(a.output/'audit_and_comparison.json',result)
    with (a.output/'siu3r_table1.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(('experiment',)+FIELDS)
        writer.writerow(('adapter_U1',)+tuple(oldtable));writer.writerow(('adapter_U0.01',)+tuple(newtable))
    (a.output/'siu3r_table1.md').write_text(markdown([('Adapter U=1',oldtable),('Adapter U=0.01',newtable)])+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':main()
