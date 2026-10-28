"""Prepare the authorized paired-control retry, reusing unchanged source audits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/fixed_query_oracle_diagnostics_20261023_v3_gradfix')
RUN = PARENT.with_name('fixed_query_oracle_diagnostics_20261023_v4_paired')
SOURCE_SHA = '694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-log', type=Path, required=True)
    args = parser.parse_args()
    if RUN.exists():
        raise FileExistsError('Inspect existing v4 run; never recreate: ' + str(RUN))
    log = args.test_log.read_text()
    assert 'Ran 11 tests' in log and 'Ran 6 tests' in log and len(re.findall(r'^OK$', log, re.M)) == 2
    assert not re.search(r'^FAILED', log, re.M)
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    code_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert sha(PARENT/'source_checkpoint.pt') == SOURCE_SHA
    configuration = json.loads((PARENT/'configuration.json').read_text())
    source_cfg = json.loads((PARENT/'source_configuration.json').read_text())
    meta = json.loads((PARENT/'source_meta.json').read_text())
    assert meta['epochs'] == 4 and meta['understanding_weight'] == 1. and meta['training_view_protocol'] == '2context+2novel'
    assert sha(configuration['cohort']) == configuration['cohort_sha256']
    cohort = json.loads(Path(configuration['cohort']).read_text())['records']
    assert len(cohort) == 1860 and len({w['scene'] for w in cohort}) == 312
    assert sha(source_cfg['manifest']) == meta['manifest_sha256']
    assert sha('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json') == meta['heldout_sha256']
    oldcode = Path(configuration['code_snapshot'])
    for rel in ('scripts/fixed_query_oracles.py', 'scripts/export_object_locus_v3_set_official.py',
                'scripts/invoke_siu3r_official_evaluator.py', 'scripts/run_fixed_query_diagnostics.sh',
                'tokengs/models/v4init_isolated_noca.py', 'tokengs/models/object_locus_panoptic_v1_lift.py'):
        assert (ROOT/rel).read_bytes() == (oldcode/rel).read_bytes(), rel
    assert shutil.disk_usage(RUN.parent).free > 40*1024**3
    RUN.mkdir()
    os.link(PARENT/'source_checkpoint.pt', RUN/'source_checkpoint.pt')
    (RUN/'source_checkpoint.sha256').write_text(SOURCE_SHA+'  '+str(RUN/'source_checkpoint.pt')+'\n')
    for name in ('source_meta.json', 'source_configuration.json', 'source_code_hashes.json'):
        shutil.copyfile(PARENT/name, RUN/name)
    shutil.copyfile(args.test_log, RUN/'cpu_tests.log')
    shutil.copyfile(ROOT/'docs/fixed_query_paired_e0_policy_20261023.md', RUN/'EXPERIMENT.md')
    shutil.copyfile(PARENT/'EXPERIMENT.md', RUN/'HISTORICAL_DEFINITION_V1.md')
    shutil.copyfile(PARENT/'FAILURE_AUDIT_71223_20261023.json', RUN/'HISTORICAL_FAILURE_71223.json')
    configuration.update(code_snapshot=str(ROOT), code_sha=code_sha, parent_run=str(PARENT), retry_of_jobs=[71223,71224],
                         baseline_control='same_forward_native_E0', historical_prediction_comparison='observational_full_audit',
                         historical_bitwise_reproduction_required=False, input_and_GT_identity_required=True,
                         recipe='fixed_weight_context_oracle_E0_E1_E2_paired_control', training=False, parameter_updates=0)
    write(RUN/'configuration.json', configuration)
    parent_pf = json.loads((PARENT/'PREFLIGHT.json').read_text())
    parent_pf.update(status='PASSED_CPU_AND_SOURCE_CHECKS', parent_preflight=str(PARENT/'PREFLIGHT.json'),
                     source_sha256_rechecked=True, cohort_manifest_heldout_rechecked=True,
                     baseline_control='same_forward_native_E0', gpu_execution_verified=False)
    write(RUN/'PREFLIGHT.json', parent_pf)
    write(RUN/'CPU_TESTS.json', dict(status='PASSED', tests=17, oracle_and_adjoint=11, paired_control=6,
          log=str(RUN/'cpu_tests.log'), scoring_test='official evaluate mocked; actual six-arm scoring orchestration',
          gpu_verified=False))
    print(RUN)


if __name__ == '__main__':
    main()
