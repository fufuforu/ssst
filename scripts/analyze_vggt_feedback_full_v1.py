"""Audit paired full-data feedback0.1/0.05 runs without training or GPU inference."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from analyze_vggt_adapter_full_v1 import audit
from analyze_vggt_interaction_controls_v1 import read, write
from analyze_vggt_feedback_factorial_v1 import markdown
from format_object_locus_posefree_siu3r_table import FIELDS, table_values


def audit_training(old_root, new_root, old_report, new_report):
    manifests = [read(root/'run_manifest.json') for root in (old_root, new_root)]
    ignored = {'recipe', 'beta', 'feedback_peak', 'nominal_logged_beta', 'feedback_implementation'}
    configs = [{k: v for k, v in m['training_configuration'].items() if k not in ignored}
               for m in manifests]
    assert configs[0] == configs[1]
    for key in ('source_checkpoint', 'vggt_source', 'manifest_sha256', 'hardware',
                'reused_reconstruction_boundary', 'adapter_understanding_scale'):
        assert manifests[0][key] == manifests[1][key]
    assert manifests[0]['training_configuration']['beta'] == '0.1*min(new_exposure/1000,1)'
    assert manifests[1]['training_configuration']['beta'] == '0.05*min(new_exposure/1000,1)'
    for root, report, manifest in zip((old_root, new_root), (old_report, new_report), manifests):
        complete = read(root/'COMPLETE.json')
        assert complete['status'] == 'TRAINING_COMPLETED'
        assert complete['recipe'] == report['training_checkpoint_metadata']['recipe'] == manifest['recipe']
        assert complete['execution_git_sha'] == report['training_checkpoint_metadata']['git_sha']
        assert (complete['new_completed_updates'], complete['new_completed_exposures']) == (4172,33376)
        assert complete['reused_reconstruction_boundary']['sha256'] == 'a50af40e1cf639514c2eb1d7b3127a79ecbc096d6d3a071a70d9a9fd2587750c'
        for rank in range(4):
            initial = read(root/f'joint_initialization_rank{rank}.json')
            assert initial['status'] == 'EXACT_BOUNDARY_RESTORED' and initial['rank'] == rank
            assert initial['source_sha256'] == complete['reused_reconstruction_boundary']['sha256']
            assert initial['rank_torch_rng_restored'] and initial['optimizer_state_empty']
            assert initial['joint_phase_completed_updates'] == initial['new_exposures'] == 0
            assert initial['adapter_understanding_scale'] == .01
    logs = [[json.loads(line) for line in (root/'training_rank0.jsonl').read_text().splitlines()]
            for root in (old_root, new_root)]
    for logged in logs:
        assert [x['completed_updates'] for x in logged] == list(range(20,4172,20))+[4172]
        assert all(x['adapter_understanding_scale'] == .01 and x['phase'] == 'frozen_joint' for x in logged)
        assert all(np.isfinite(x[k]) for x in logged for k in ('loss_recon','loss_understanding','preclip_norm'))
    for old, new in zip(*logs):
        for key in ('phase_update', 'window_indices', 'group_lr', 'beta', 'understanding_weight',
                    'lr_multiplier', 'completed_updates', 'completed_exposures'):
            assert old[key] == new[key]
        assert new['feedback_peak'] == .05
        assert abs(new['actual_feedback_beta'] - old['beta']*.5) < 1e-14
    startup = read(new_root.parent.parent/'group_plus'/new_root.name/'startup_validation.json')
    assert startup['status'] == 'STARTUP_AUDIT_PASSED'
    return dict(status='SAME_BOUNDARY_BUDGET_DATA_SCHEDULE_AND_HARDWARE_VERIFIED',
                all_four_rank_boundary_rng_and_fresh_optimizer_verified=True,
                rank0_logged_steps_checked=len(logs[0]),
                rank0_sampled_sampling_lr_and_warmups_identical=True,
                actual_feedback_beta_half_native_nominal_beta=True,
                science_change='Feedback peak0.1 to0.05 only; adapter understanding gradient scale0.01 unchanged.',
                new_optimizer_calls=4172, new_exposures=33376,
                cumulative_optimizer_calls=6258, cumulative_exposures=50064,
                sampled_log_caveat='Rank0 logs every20 calls plus final call; not per-call or all-rank loss/gradient records.')


def native_state_check(old_root, new_root, old_report, new_report, output):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.posefree_official_state_helpers import merge
    torch.set_num_threads(4)
    all_blobs = []
    for root, report in ((old_root, old_report), (new_root, new_report)):
        blobs = [torch.load(root/f'rank{i:02d}/official_states.pt', map_location='cpu',
                            mmap=True, weights_only=False)['all'] for i in range(4)]
        names = [name for blob in blobs for name in blob['names']]
        assert len(names) == len(set(names)) == 1860
        merged = merge(blobs, output)
        for view in ('context', 'target'):
            for field in ('miou', 'pq'):
                key = view+'_'+field
                assert abs(merged[key] - report['official_segmentation']['all'][key]) < 1e-12
            for field in ('map','map_50','map_75'):
                assert abs(merged[view+'_map'][field] - report['official_segmentation']['all'][view+'_map'][field]) < 1e-12
        all_blobs.append(blobs)
    gt_windows = 0
    for old, new in zip(*all_blobs):
        assert old['names'] == new['names']
        for view in ('context', 'target'):
            left, right = old['states'][view+'_mAP'], new['states'][view+'_mAP']
            for key in left:
                if not key.startswith('groundtruth_'):
                    continue
                assert len(left[key]) == len(right[key])
                for x,y in zip(left[key], right[key]):
                    assert torch.equal(x,y) if torch.is_tensor(x) else x == y
        gt_windows += len(old['names'])
    return dict(status='NATIVE_GLOBAL_SEGMENTATION_REMERGE_VERIFIED', windows=gt_windows,
                all_six_miou_map_pq_plus_ap50_ap75_reproduced_to_1e12=True,
                both_models_context_and_target_map_groundtruth_identical=True,
                global_AP_not_averaged_across_shards=True, device='cpu', new_optimizer_updates=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('old', 'new', 'old-training', 'new-training', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--native-state-check', action='store_true')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    old, oldrecords, oldrows, oldvalues, oldgeometry = audit(a.old)
    new, records, rows, values, geometry = audit(a.new)
    assert records == oldrecords
    assert [x[:2] for x in rows] == [x[:2] for x in oldrows]
    for key in ('manifest_sha256','cohort_source_sha256','vggt_revision','geometry_quality_policy'):
        assert old['export_identity'][key] == new['export_identity'][key]
    assert old['training_checkpoint_metadata']['adapted_vggt_identity']['sha256'] == new['training_checkpoint_metadata']['adapted_vggt_identity']['sha256']
    training = audit_training(a.old_training, a.new_training, old, new)
    scene_names = sorted({x[1] for x in rows}); assert len(scene_names) == 312
    delta = values-oldvalues
    sums = np.asarray([delta[[x[1] == name for x in rows]].sum(0) for name in scene_names])
    counts = np.asarray([sum(x[1] == name for x in rows) for name in scene_names])
    indices = np.random.default_rng(42).integers(0,312,size=(10000,312))
    samples = sums[indices].sum(1)/counts[indices].sum(1)[:,None]
    bounds = np.quantile(samples,[.025,.975],axis=0)
    oldtable, newtable = table_values(old), table_values(new)
    result = dict(status='FULL_PAIRED_OFFICIAL_RESULTS_VERIFIED', windows=1860, scenes=312, images=11160,
        feedback_peak=dict(old=.1,new=.05), adapter_understanding_scale=.01,
        old_checkpoint_sha256=old['checkpoint_sha256'], new_checkpoint_sha256=new['checkpoint_sha256'],
        old_table=oldtable, new_table=newtable,
        deltas={k:None if x is None else y-x for k,x,y in zip(FIELDS,oldtable,newtable)},
        bootstrap_reconstruction={k:dict(mean_delta=float(delta[:,i].mean()),
            scene_cluster_95_interval=bounds[:,i].tolist()) for i,k in enumerate(FIELDS[:5])},
        bootstrap_samples=10000, seed=42,
        bootstrap_scope='Dataset uncertainty only; paired312 whole-scene clusters preserving window/image weighting. No multi-seed uncertainty or segmentation significance claim.',
        training_audit=training, same_manifest_and_frame_ids=True,
        all_four_rank_export_score_receipts_and_cuda_merge_contracts_passed=True,
        depth_rgb_saved_scores_reproduce_official_means=True,
        geometry_quality_counts=dict(old=oldgeometry,new=geometry),
        inference_code_sha=new['inference_code_sha'], reduction_code_sha=new['evaluation_code_sha'],
        optimizer_updates=0)
    if a.native_state_check:
        result['native_state_audit'] = native_state_check(a.old,a.new,old,new,a.output)
    write(a.output/'audit_and_comparison.json',result)
    with (a.output/'siu3r_table1.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(('experiment',)+FIELDS)
        writer.writerow(('adapter0.01_feedback0.1',)+tuple(oldtable))
        writer.writerow(('adapter0.01_feedback0.05',)+tuple(newtable))
    (a.output/'siu3r_table1.md').write_text(markdown([('Feedback0.1',oldtable),('Feedback0.05',newtable)])+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
