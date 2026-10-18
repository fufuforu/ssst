import copy
import csv
import json
import pytest
import torch
from scripts import direct_instance_full as full
from scripts.direct_instance_full import ARCH, validate_full_receipt, validate_pair_exports


def test_full_receipt_refuses_short_or_unequal_budget():
    done = dict(status='TRAINING_COMPLETE', progress=dict(epoch=4, update_in_epoch=0, full_updates=4172))
    meta = dict(architecture=ARCH, epochs=4, world_size=8, global_batch=8, short_validation=False,
                learning_rate=1e-4, understanding_weight=.05, understanding_warmup=1043)
    validate_full_receipt(done, meta, architecture=ARCH)
    for key, value in [('short_validation', True), ('epochs', 8), ('world_size', 4), ('understanding_warmup', 20)]:
        with pytest.raises(ValueError):
            validate_full_receipt(done, dict(meta, **{key: value}), architecture=ARCH)
    with pytest.raises(ValueError):
        validate_full_receipt(dict(done, progress=dict(epoch=4, update_in_epoch=0, full_updates=256)), meta, architecture=ARCH)


def test_pair_requires_all_128_identical_inputs():
    left = dict(status='EXPORTED', records=[dict(name=str(i), source_index=i, frame_ids=[0, 1, 2, 3, 4, 5],
                                              input_rgb_sha256=str(i)) for i in range(128)])
    validate_pair_exports(left, copy.deepcopy(left))
    for field, value in [('input_rgb_sha256', 'different'), ('frame_ids', [1, 0, 2, 3, 4, 5])]:
        right = copy.deepcopy(left)
        right['records'][0][field] = value
        with pytest.raises(ValueError):
            validate_pair_exports(left, right)
    with pytest.raises(ValueError):
        validate_pair_exports(left, dict(left, records=left['records'][:-1]))


def test_audit_replays_full_order_and_rejects_inherited_or_wrong_data(tmp_path, monkeypatch):
    from scripts.dual_stream_training_policy import epoch_order, rank_microbatches
    monkeypatch.setattr(full, 'RUN', tmp_path)
    initial = dict(model={'weight': torch.tensor([1.])}, optimizer=dict(state={}),
                   progress=dict(epoch=0, update_in_epoch=0, full_updates=0))
    torch.save(initial, tmp_path / 'checkpoint_untrained_initial.pt')
    source_sha = full.digest(tmp_path / 'checkpoint_untrained_initial.pt')
    meta = dict(architecture=ARCH, epochs=4, world_size=8, global_batch=8, short_validation=False,
                learning_rate=1e-4, understanding_weight=.05, understanding_warmup=1043,
                code_sha='code', initial_sha256=source_sha, training_windows=None,
                manifest_sha256='data', heldout_sha256='held', omega_identity={}, sam_identity={})
    initial['meta'] = meta
    torch.save(initial, tmp_path / 'checkpoint_initial.pt')
    final = dict(model={'weight': torch.tensor([2.])}, optimizer=dict(state={0: dict(step=torch.tensor(4172.))}),
                 meta=meta, progress=dict(epoch=4, update_in_epoch=0, full_updates=4172))
    torch.save(final, tmp_path / 'checkpoint_latest.pt')
    full.write(tmp_path / 'TRAINING_COMPLETE.json', dict(status='TRAINING_COMPLETE', progress=final['progress']))
    full.write(tmp_path / 'preflight.json', dict(meta, untrained_initial_sha256=source_sha,
               extra_checkpoint_metadata={}, trainable_parameter_names=['weight']))
    full.write(tmp_path / 'FIRST_UPDATE_VERIFIED.json', dict(status='PASSED',
               ranks=[dict(sha256='updated', changed=True) for _ in range(8)]))
    orders = [epoch_order(8337, i) for i in range(4)]
    for rank in range(8):
        rows = []
        for step in range(4172):
            epoch, update = divmod(step, 1043)
            rows.append(dict(update=step+1, epoch=epoch, local_indices=rank_microbatches(orders[epoch], update, rank, 8),
                             understanding_weight=.05, instance_feature_effective_weight=.0005,
                             seconds=1., peak_cuda_memory_bytes=0))
        (tmp_path / f'train_rank{rank:02d}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    full.audit()
    assert full.read(tmp_path / 'FULL_TRAINING_AUDIT.json')['full_sample_order_verified']
    path = tmp_path / 'train_rank00.jsonl'
    text = path.read_text()
    first, rest = text.split('\n', 1)
    row = json.loads(first)
    row['local_indices'] = [-1]
    path.write_text(json.dumps(row)+'\n'+rest)
    with pytest.raises(AssertionError):
        full.audit()
    path.write_text(text)
    initial['progress']['full_updates'] = 2
    torch.save(initial, tmp_path / 'checkpoint_initial.pt')
    with pytest.raises(AssertionError):
        full.audit()


def test_official_comparison_completes_with_native_metrics(tmp_path, monkeypatch):
    from scripts.format_object_locus_posefree_siu3r_table import FIELDS, table_values
    from scripts.invoke_siu3r_official_evaluator import SIU3R_COMMIT
    run, base = tmp_path / 'run', tmp_path / 'base'
    monkeypatch.setattr(full, 'RUN', run)
    monkeypatch.setattr(full, 'BASE', base)
    records = [dict(name=str(i), source_index=i, frame_ids=list(range(6)), input_rgb_sha256=str(i)) for i in range(128)]
    for folder in (run, base):
        ev = folder / 'evaluation'
        ev.mkdir(parents=True)
        full.write(ev / 'export_complete.json', dict(status='EXPORTED', records=records))
        full.write(folder / 'COMPLETE.json', dict(status='TRAINING_AND_HELDOUT128_OFFICIAL_EVAL_COMPLETE'))
        raw = dict(absrel=.1, rmse=.2, psnr=18., ssim=.6, lpips=.5,
                   context_miou=.4, context_map=dict(map=.2), context_pq=.45,
                   target_miou=.4, target_map=dict(map=.2), target_pq=.45)
        report = dict(official_commit=SIU3R_COMMIT, official_segmentation=dict(all=raw),
                      scopes={'context': dict(mIoU_t=None), 'target-all': dict(raw, mIoU_t=None)})
        full.write(ev / 'metrics.json', report)
        with (ev / 'siu3r_table1.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerow(table_values(report))
    full.write(run / 'preflight.json', dict(baseline_export_sha256=full.digest(base / 'evaluation/export_complete.json'), limits=[]))
    full.write(run / 'FULL_TRAINING_AUDIT.json', dict(status='DIRECT_ATTRIBUTE_FULL4_TRAINING_AUDIT_PASSED', final_parameter_sha256='fixed'))
    readout = run / 'readouts_final'
    readout.mkdir()
    full.write(readout / 'COMPLETE.json', dict(windows=8))
    full.write(readout / 'SUMMARY.json', dict(rows=[{}]*8, optimizer_created=False, summary=[],
               fixed_weight_checks=[dict(before='fixed', after='fixed', no_parameter_gradients=True,
                                         all_parameter_versions_unchanged=True)]))
    full.compare()
    assert (run / 'FULL_EXPERIMENT_COMPLETE.json').exists()
    comparison = full.read(run / 'comparison_to_baseline_full4/comparison_status.json')
    assert all(v == 0 for v in comparison['delta'] if v != '')
