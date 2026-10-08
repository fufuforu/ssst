#!/usr/bin/env python3
"""Build a new C/D-only attempt from verified, immutable attempt02 caches."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from scripts.object_locus_probe_metrics import verify_gc_cache_manifest, verify_r3d_cache_manifest

SOURCE = Path('/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt02')
ENTRYPOINT_PREFLIGHT = Path('/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt03/entrypoint_preflight.json')
OLD_D_EVIDENCE = Path('/space/mawb/ssst/group_plus/object_locus_frozen_representation_diagnostic_v1/attempts/attempt03/slurm')
EXPECTED_GC = '72f7440b5b3cf2fd877dfc534ae5c4a409c76c9884729fe23247008aab8c0d58'
EXPECTED_R3D = '9f06d78c58bd5e00b84ed712840e767c3151db1fe170f3408c6079fca263db2c'
OLD_CODE = '6489441d1814af23cafab9a9170ba77c4b1e5da3'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def copy_input(root, rel, ledger):
    src = SOURCE / rel
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    shutil.copy2(src, dst)
    digest = sha(src)
    if sha(dst) != digest:
        raise RuntimeError(f'copied input SHA mismatch: {rel}')
    ledger.append({'relative_path': rel, 'source_path': str(src), 'sha256': digest,
                   'size': src.stat().st_size, 'method': 'byte_copy'})


def link_tree(root, rel, ledger):
    srcroot = SOURCE / rel
    if not srcroot.is_dir():
        raise FileNotFoundError(srcroot)
    files = sorted(p for p in srcroot.rglob('*') if p.is_file())
    if not files:
        raise RuntimeError(f'required immutable input tree is empty: {rel}')
    for src in files:
        source_real = src.resolve(strict=True)
        sub = src.relative_to(SOURCE)
        dst = root / sub
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            raise FileExistsError(dst)
        os.symlink(str(source_real), str(dst))
        if not dst.is_file() or dst.stat().st_size != src.stat().st_size or sha(dst) != sha(src):
            raise RuntimeError(f'linked input verification failed: {dst}')
        ledger.append({'relative_path': sub.as_posix(), 'source_path': str(source_real),
                       'sha256': sha(src), 'size': src.stat().st_size, 'method': 'file_symlink'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=Path, required=True)
    ap.add_argument('--execution-sha', required=True)
    a = ap.parse_args()
    root = a.attempt
    if root.resolve() == SOURCE.resolve():
        raise RuntimeError('resume attempt must be separate from attempt02')
    if not root.is_dir():
        raise FileNotFoundError(root)
    # Do not overwrite caller-retained cancellation evidence; all other staged
    # data paths must be absent before construction.
    ledger = []
    endpoint = json.loads((SOURCE / 'endpoint_load_check.json').read_text())
    gc_receipt = json.loads((SOURCE / 'extraction_complete.json').read_text())
    freeze = json.loads((SOURCE / 'freeze_check.json').read_text())
    r3d = json.loads((SOURCE / 'r3d_frozen_inference_receipt.json').read_text())
    r3d_endpoint = json.loads((SOURCE / 'r3d_endpoint_manifest.json').read_text())
    gc_cache_manifest = json.loads((SOURCE / 'cache_manifest.json').read_text())
    r3d_cache_manifest = json.loads((SOURCE / 'r3d_cache_manifest.json').read_text())
    cohort = json.loads((SOURCE / 'cohort_manifest.json').read_text())
    if endpoint.get('checkpoint_sha256') != EXPECTED_GC or not endpoint.get('tensor_values_exact') or endpoint.get('state_tensors') != 1445:
        raise RuntimeError('GC001 strict load receipt does not match locked endpoint')
    if gc_receipt.get('status') != 'PASS' or gc_receipt.get('complete') is not True or gc_receipt.get('cached_windows') != 1040:
        raise RuntimeError('GC001 full extraction receipt is incomplete')
    if len(gc_cache_manifest.get('files', [])) != 1040 or gc_cache_manifest.get('window_count') != 1040:
        raise RuntimeError('GC001 cache manifest does not cover all 1040 windows')
    if (sum(row.get('split') == 'train' for row in gc_cache_manifest['files']) != 1008 or
            sum(row.get('split') == 'dev' for row in gc_cache_manifest['files']) != 8 or
            sum(row.get('split') == 'test' for row in gc_cache_manifest['files']) != 24):
        raise RuntimeError('GC001 cache manifest split counts do not match the fixed cohort')
    if freeze.get('status') != 'PASS' or freeze.get('state_sha256_before') != freeze.get('state_sha256_after') or freeze.get('requires_grad_parameters') != 0 or freeze.get('non_null_grads') != 0:
        raise RuntimeError('GC001 frozen-state receipt failed')
    if freeze.get('forward_step') != 58128 or freeze.get('beta') != 0.1:
        raise RuntimeError('GC001 frozen forward-step/beta receipt mismatches the lock')
    if r3d.get('status') != 'PASS' or r3d.get('windows') != 32 or r3d.get('dev') != 8 or r3d.get('test') != 24 or r3d.get('endpoint_sha256') != EXPECTED_R3D:
        raise RuntimeError('R3D inference receipt is not the complete locked 32-window run')
    if len(r3d.get('records', [])) != 32 or len(r3d_cache_manifest.get('records', [])) != 32 or r3d_cache_manifest.get('window_count') != 32:
        raise RuntimeError('R3D receipt/cache manifest does not enumerate all 32 windows')
    if not all(record.get('finite') is True for record in r3d['records']):
        raise RuntimeError('R3D receipt contains nonfinite window records')
    def identity(row):
        return (row['window_id'], row['split'], row['scene'], tuple(row['context']), tuple(row['novel']), tuple(row['frame_ids']))
    fixed_identities = []
    for split in ('test', 'dev'):
        for index, row in enumerate(cohort[split]):
            window_id = f'{split}_{index:04d}_{row["scene"]}_c{"_".join(map(str, row["context"]))}'
            frame_ids = list(row['context']) + list(row['novel'])
            fixed_identities.append((window_id, split, row['scene'], tuple(row['context']),
                                     tuple(row['novel']), tuple(frame_ids)))
    if [identity(row) for row in r3d['records']] != fixed_identities:
        raise RuntimeError('R3D record window/frame identity differs from the fixed cohort manifest')
    if [identity(row) for row in r3d_cache_manifest['records']] != fixed_identities:
        raise RuntimeError('R3D cache manifest window/frame identity differs from the fixed cohort manifest')
    pair_identity = json.loads((SOURCE / 'r3d_h0_cache_pair_identity.json').read_text())
    if (pair_identity.get('same_manifest_and_window_identity') is not True or
            pair_identity.get('same_GT_frame_ids') is not True or
            pair_identity.get('probe_readout_does_not_reuse_R3D_mask') is not True or
            [identity(row) for row in pair_identity.get('windows', [])] != fixed_identities):
        raise RuntimeError('R3D/H0 pair identity receipt differs from the fixed cohort')
    if r3d.get('state_sha256_before') != r3d.get('state_sha256_after') or r3d.get('requires_grad_parameters') != 0 or r3d.get('non_null_grads') != 0 or r3d.get('all_finite') is not True:
        raise RuntimeError('R3D frozen inference state checks failed')
    if r3d_endpoint.get('checkpoint_sha256') != EXPECTED_R3D or r3d_endpoint.get('source_checkpoint_sha256') != '68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a' or r3d_endpoint.get('plan_sha256') != '0a7c8173b3dcc187d7ce3074e69d9c8649204a933262ded3a773b08898650ea8' or r3d_endpoint.get('model_exposure') != 58128:
        raise RuntimeError('R3D endpoint identity receipt mismatches locked source')
    if r3d_endpoint.get('completed_updates') != 1008 or r3d_endpoint.get('new_exposures') != 8064 or r3d_endpoint.get('epoch') != 8 or r3d_endpoint.get('training_code_sha') != 'd4c096b80c4a28e93e095abac7dd777ed1af5f3a':
        raise RuntimeError('R3D training receipt does not match the locked epoch8 budget')
    gc_manifest_result = verify_gc_cache_manifest(SOURCE)
    r3d_manifest_result = verify_r3d_cache_manifest(SOURCE)

    # Original provenance is retained byte-for-byte in a namespaced directory.
    provenance = root / 'provenance/source_attempt02'
    for rel in ('git_provenance.json', 'execution_files_manifest.json', 'retry02_preflight.json',
                'effective_execution_protocol.json',
                'retry02_startup_status.json', 'jobs.json', 'parallel_execution_registration.json'):
        src = SOURCE / rel
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = provenance / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha(src) != sha(dst):
            raise RuntimeError(f'original provenance byte copy mismatch: {rel}')
        ledger.append({'relative_path': str(dst.relative_to(root)), 'source_path': str(src),
                       'sha256': sha(src), 'size': src.stat().st_size, 'method': 'byte_copy_original_provenance'})
    cancellation_dir = root / 'provenance/attempt03_old_d_cancellation'
    for name in ('job_59167_before_cancel.txt', 'job_59167_before_cancel.sacct.txt',
                 'job_59167_after_cancel.txt', 'job_59167_after_cancel.sacct.txt',
                 'old_d_job_output_absence.txt', 'attempt02_frozen-unified-eval-59167.err',
                 'attempt02_frozen-unified-eval-59167.out'):
        src = OLD_D_EVIDENCE / name
        if not src.is_file():
            if name in ('attempt02_frozen-unified-eval-59167.err', 'attempt02_frozen-unified-eval-59167.out'):
                continue
            raise FileNotFoundError(src)
        dst = cancellation_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha(src) != sha(dst):
            raise RuntimeError(f'old D cancellation evidence copy mismatch: {name}')
        ledger.append({'relative_path': str(dst.relative_to(root)), 'source_path': str(src),
                       'sha256': sha(src), 'size': src.stat().st_size,
                       'method': 'byte_copy_attempt03_cancellation_evidence'})

    metadata = ['source_manifest.json', 'protocol.json', 'data_contract.json', 'cohort_identity.json',
                'cohort_manifest.json', 'public_cohort_identity_receipt.json', 'endpoint_source_hashes.json',
                'endpoint_load_check.json', 'freeze_check.json', 'extraction_complete.json',
                'cache_manifest.json', 'r3d_endpoint_manifest.json',
                'r3d_frozen_inference_receipt.json', 'r3d_cache_manifest.json', 'r3d_h0_cache_pair_identity.json',
                'r3d_registration_normalized.json', 'r3d_registration_decision_contract_checks.json',
                'retry02_preflight.json', 'previous_attempt.json', 'environment_preflight.json',
                'gpu_runtime.json', 'smoke.json', 'h0_cache_replay_parity_first_dev.json', 'startup_confirmation.json']
    for rel in metadata:
        copy_input(root, rel, ledger)
    prior_protocol_path = provenance / 'effective_execution_protocol.json'
    effective = json.loads(prior_protocol_path.read_text())
    effective['attempt_root'] = str(root)
    effective['execution_code_sha'] = a.execution_sha
    effective['recovery_protocol'] = 'Frozen Probe + R3D: reuse completed caches, restore C/D user prompt'
    effective['execution'] = {
        'node': '3dimage-11', 'partition': '3090',
        'A': {'gpus': 1, 'cpus': 4, 'memory': '64G', 'time': '24:00:00',
              'state': 'REUSED_COMPLETED_SOURCE', 'source_job_id': '59164', 'new_job_submitted': False},
        'B': {'gpus': 1, 'cpus': 4, 'memory': '64G', 'time': '24:00:00',
              'state': 'REUSED_COMPLETED_SOURCE', 'source_job_id': '59165', 'new_job_submitted': False},
        'C': {'gpus': 3, 'cpus': 12, 'memory': '64G', 'time': '24:00:00', 'dependency': None},
        'D': {'gpus': 0, 'cpus': 4, 'memory': '64G', 'time': '48:00:00',
              'dependency': 'afterok:<new-C-job-id>'},
    }
    effective['execution_recovery'] = {
        'rerun_A': False, 'rerun_B': False, 'reuse_attempt02_complete_cache': True,
        'submit_new_C_and_D_once': True, 'old_C_job_id': '59166', 'old_D_job_id': '59167',
        'old_D_cancelled_before_start': True, 'compute_node_git_checkout': False,
        'retry_after_failure': False,
    }
    effective['previous_attempt02_protocol_sha256'] = sha(prior_protocol_path)
    effective_path = root / 'effective_execution_protocol.json'
    effective_path.write_text(json.dumps(effective, indent=2) + '\n')
    ledger.append({'relative_path': 'effective_execution_protocol.json',
                   'source_path': str(prior_protocol_path),
                   'sha256': sha(prior_protocol_path), 'size': prior_protocol_path.stat().st_size,
                   'result_sha256': sha(effective_path), 'result_size': effective_path.stat().st_size,
                   'method': 'recovery_protocol_derived_from_byte_preserved_attempt02_protocol'})
    preflight_source = root / 'entrypoint_preflight.json'
    if not preflight_source.is_file():
        preflight_source = ENTRYPOINT_PREFLIGHT
    if not preflight_source.is_file():
        raise FileNotFoundError(preflight_source)
    preflight_copy = root / 'entrypoint_preflight.json'
    if preflight_source != preflight_copy:
        shutil.copy2(preflight_source, preflight_copy)
    if sha(preflight_copy) != sha(preflight_source):
        raise RuntimeError('entrypoint preflight record copy SHA mismatch')
    entrypoint_record = json.loads(preflight_copy.read_text())
    if entrypoint_record.get('status') != 'PASS' or any(
            row.get('exit_code') != 0 or not row.get('help_usage_present') or row.get('module_not_found')
            for row in entrypoint_record.get('checks', [])) or len(entrypoint_record.get('checks', [])) != 6:
        raise RuntimeError('actual C/D module --help preflight record is not a six-entry PASS')
    code_root = Path(__file__).resolve().parents[1]
    for rel, info in entrypoint_record.get('source_files', {}).items():
        if sha(code_root / rel) != info['sha256']:
            raise RuntimeError(f'entrypoint preflight source hash differs from committed code: {rel}')
    ledger.append({'relative_path': 'entrypoint_preflight.json', 'source_path': str(preflight_source),
                   'sha256': sha(preflight_source), 'size': preflight_source.stat().st_size,
                   'method': 'byte_copy_actual_entrypoint_preflight'})
    for rel in ('labels/train_labels.csv', 'labels/dev_test_labels.csv', 'labels/original_context_hungarian.csv'):
        copy_input(root, rel, ledger)

    for rel in ('cache', 'features', 'reference_gpu', 'reconstruction_cache', 'r3d'):
        link_tree(root, rel, ledger)

    # Validate cache manifests from the new root and explicitly verify all
    # staged tree entries resolve to the already verified attempt02 bytes.
    gc_view = verify_gc_cache_manifest(root)
    r3d_view = verify_r3d_cache_manifest(root)
    if gc_view != gc_manifest_result or r3d_view != r3d_manifest_result:
        raise RuntimeError(f'new-root manifest check differs from source: {gc_manifest_result}/{r3d_manifest_result} vs {gc_view}/{r3d_view}')
    must_exist = [root / 'cache/train' / f'{i:04d}.npz' for i in (0, 1007)]
    must_exist += [root / 'cache/train_iou' / f'{i:04d}.npz' for i in (0, 1007)]
    must_exist += [root / 'features' / 'dev_test_q_z.npz']
    for split, count in (('dev', 8), ('test', 24)):
        for i in range(count):
            w = json.loads((root / 'cohort_manifest.json').read_text())[split][i]
            prefix = f'{split}_{i:04d}_{w["scene"]}_c{"_".join(map(str,w["context"]))}'
            must_exist.extend([root / 'features' / f'{prefix}.npz', root / 'cache/dev_test' / f'{prefix}_iou.npz',
                               root / 'reference_gpu' / f'{w["scene"]}_context{"_".join(map(str,w["context"]))}'])
            rp = f'{w["scene"]}_context{"_".join(map(str,w["context"]))}'
            must_exist.extend([root / 'r3d/features' / f'{prefix}.npz', root / 'r3d/cache' / f'{prefix}.npz',
                               root / 'r3d/reference_gpu' / rp,
                               root / 'reconstruction_cache/GC001' / split / f'{rp}.npz',
                               root / 'reconstruction_cache/R3D' / split / f'{rp}.npz'])
    missing = [str(p) for p in must_exist if not p.exists()]
    if missing:
        raise RuntimeError(f'new-root consumed input paths missing: {missing[:20]}')

    root_number = int(root.name.removeprefix('attempt'))
    previous_root = root.parent / f'attempt{root_number - 1:02d}'
    source_provenance = {
        'source_attempt_root': str(SOURCE), 'new_attempt_root': str(root),
        'attempt_root_selection_reason': f'{previous_root} already contains recovery outputs; retained it and selected this next empty root.',
        'GC001': {'job_id': '59164', 'status': 'COMPLETED 0:0', 'execution_sha': OLD_CODE,
                  'checkpoint_sha256': EXPECTED_GC, 'receipt': 'extraction_complete.json',
                  'manifest_check': gc_manifest_result, 'new_root_manifest_check': gc_view},
        'R3D': {'job_id': '59165', 'status': 'COMPLETED 0:0', 'execution_sha': OLD_CODE,
                'checkpoint_sha256': EXPECTED_R3D, 'windows': 32, 'dev': 8, 'test': 24,
                'receipt': 'r3d_frozen_inference_receipt.json', 'manifest_check': r3d_manifest_result,
                'new_root_manifest_check': r3d_view},
        'failed_C': {'job_id': '59166', 'execution_sha': OLD_CODE,
                     'failure': 'ModuleNotFoundError: No module named scripts; before worker start'},
        'cancelled_old_D': {'job_id': '59167', 'before_cancel': 'PENDING DependencyNeverSatisfied afterok:59166(failed)',
                            'after_cancel': 'CANCELLED before start'},
        'recovery_execution_sha': a.execution_sha,
        'model_extraction_or_inference_execution_sha': OLD_CODE,
        'probe_training_and_unified_evaluation_execution_sha': a.execution_sha,
        'shared_modules_have_cuda_noop': True,
    }
    (root / 'source_stage_provenance.json').write_text(json.dumps(source_provenance, indent=2) + '\n')
    code_identity = {
        'status': 'LOCKED', 'attempt_root': str(root), 'execution_sha': a.execution_sha,
        'source_attempt02_code_sha': OLD_CODE,
        'A_extraction_code_sha': OLD_CODE, 'B_r3d_inference_code_sha': OLD_CODE,
        'C_probe_training_code_sha': a.execution_sha,
        'D_unified_evaluation_code_sha': a.execution_sha,
        'source_models_reloaded_or_forwarded_during_resume_preparation': False,
        'committed_worktree_is_read_only_after_preparation': True,
    }
    (root / 'execution_code_identity.json').write_text(json.dumps(code_identity, indent=2) + '\n')
    git_provenance = {
        'task_branch': 'object-locus-frozen-representation-diagnostic-v1',
        'execution_sha': a.execution_sha, 'recovery_base_sha': OLD_CODE,
        'remote_origin': 'inherited from provenance/source_attempt02/git_provenance.json',
        'A_job_id': '59164', 'A_execution_sha': OLD_CODE,
        'B_job_id': '59165', 'B_execution_sha': OLD_CODE,
        'C_and_D_execution_sha': a.execution_sha,
        'recovery_protocol': 'docs/object_locus_frozen_probe_resume_attempt03.md',
    }
    (root / 'git_provenance.json').write_text(json.dumps(git_provenance, indent=2) + '\n')
    (root / 'jobs.json').write_text(json.dumps({
        'attempt_root': str(root), 'execution_sha': a.execution_sha,
        'source_jobs': {'A_gc001': {'job_id': '59164', 'state': 'COMPLETED', 'exit_code': '0:0', 'execution_sha': OLD_CODE},
                        'B_r3d': {'job_id': '59165', 'state': 'COMPLETED', 'exit_code': '0:0', 'execution_sha': OLD_CODE}},
        'old_failed_jobs': {'C_heads': {'job_id': '59166', 'state': 'FAILED', 'reason': 'ModuleNotFoundError: No module named scripts'},
                            'D_unified_eval': {'job_id': '59167', 'state': 'CANCELLED', 'reason': 'DependencyNeverSatisfied afterok:59166(failed)', 'cancelled_before_start': True}},
        'new_jobs': {'C_heads': {'job_id': None, 'state': 'NOT_SUBMITTED'},
                     'D_unified_eval': {'job_id': None, 'state': 'NOT_SUBMITTED', 'dependency': 'afterok:<new-C-job-id>'}},
        'A_B_reused_after_cache_verification': True,
    }, indent=2) + '\n')
    reuse = {'status': 'PASS', 'source_root': str(SOURCE), 'new_root': str(root),
             'files': sorted(ledger, key=lambda x: x['relative_path']),
             'GC001_manifest_source_and_view': gc_manifest_result,
             'R3D_manifest_source_and_view': r3d_manifest_result,
             'consumed_relative_paths_checked': len(must_exist), 'missing_consumed_paths': []}
    (root / 'cache_reuse_manifest.json').write_text(json.dumps(reuse, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'attempt_root': str(root), 'reused_files': len(ledger),
                      'GC001': gc_view, 'R3D': r3d_view, 'consumed_paths': len(must_exist)}), flush=True)


if __name__ == '__main__':
    main()
