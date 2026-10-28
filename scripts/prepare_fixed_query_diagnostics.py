"""Prepare one isolated no-training run; refuses an existing run directory."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/fixed_query_oracle_diagnostics_20261023_v1')
REFERENCE = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_noca_v4recipe_20261022_v1/eval')
SOURCE = REFERENCE / 'source_checkpoint.pt'
COHORT = Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json')
SOURCE_SHA = '694d2991ea9f2a6cc02106a569de35ff907db219185cb58c94b8a352540c51fc'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    if RUN.exists():
        raise FileExistsError('Refusing duplicate preparation: ' + str(RUN))
    assert digest(SOURCE) == SOURCE_SHA
    baseline_cfg = json.loads((REFERENCE / 'configuration.json').read_text())
    assert digest(COHORT) == baseline_cfg['cohort_sha256']
    cohort = json.loads(COHORT.read_text())['records']
    assert len(cohort) == 1860 and len({w['scene'] for w in cohort}) == 312
    verified = json.loads((REFERENCE / 'EVALUATION_VERIFIED.json').read_text())
    assert verified['status'] == 'PASSED' and verified['matched_inputs'] and verified['matched_gt']
    import torch
    from scripts.v4init_isolated_recipe import recipe_identity, KEY
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
    meta = saved['meta']
    assert saved['progress'] == dict(epoch=4, update_in_epoch=0, full_updates=4172)
    assert meta['architecture'] == 'OMEGA_V4INIT_NO_ANCHOR_CA_ISOLATED_V4_RECIPE_V1'
    assert meta['epochs'] == 4 and meta['understanding_weight'] == 1.0
    assert all(meta.get(k) == v for k,v in recipe_identity(KEY).items())
    assert len(saved['model']) == 632
    assert all(torch.isfinite(t).all() for t in saved['model'].values() if torch.is_tensor(t) and t.is_floating_point())
    assert digest(Path(baseline_cfg['manifest'])) == meta['manifest_sha256']
    from scripts.train_dual_stream_v1 import HELDOUT
    assert digest(Path(HELDOUT)) == meta['heldout_sha256']
    old = Path(baseline_cfg['code_snapshot'])
    source_hashes = {}
    # Every pre-existing model/data/render/export Python file must match the
    # evaluated snapshot except four reviewed additions for views/continuation.
    reviewed = {'tokengs/models/superpoint_structural_variants.py', 'scripts/export_unified_query_checkpoint.py',
                'scripts/v4init_isolated_recipe.py', 'scripts/train_unified_query_v1.py'}
    differences = []
    for sub in ('tokengs', 'scripts'):
        for f in old.joinpath(sub).rglob('*.py'):
            rel = str(f.relative_to(old))
            current = digest(ROOT / rel)
            if digest(f) != current:
                differences.append(rel)
                assert rel in reviewed, 'Unreviewed source drift: ' + rel
            source_hashes[rel] = current
    if shutil.disk_usage(RUN.parent).free < 40 * 1024**3:
        raise OSError('Need 40GiB free for diagnostic exports')
    # Inspect reference artifact coverage without reading all RGB/depth images.
    refs = json.loads((REFERENCE / 'evaluation' / 'export_complete.json').read_text())['records']
    assert len(refs) == 1860
    for r in refs:
        pair = REFERENCE / 'evaluation' / 'all' / r['name']
        for sub in ('rgb', 'rgb_gt', 'depth', 'depth_gt', 'context_seg_pred', 'context_seg_gt', 'target_seg_pred', 'target_seg_gt'):
            expected = 2 if sub.startswith('context_') else 6
            assert len(list((pair / sub).glob('*.png'))) == expected, (pair, sub)
    RUN.mkdir()
    os.link(SOURCE, RUN / 'source_checkpoint.pt')
    (RUN / 'source_checkpoint.sha256').write_text(SOURCE_SHA + '  ' + str(RUN / 'source_checkpoint.pt') + '\n')
    write(RUN / 'source_meta.json', meta)
    write(RUN / 'source_configuration.json', baseline_cfg)
    write(RUN / 'source_code_hashes.json', source_hashes)
    write(RUN / 'configuration.json', dict(recipe='fixed_weight_context_oracle_E0_E1_E2', source_checkpoint=str(SOURCE),
          source_sha256=SOURCE_SHA, progress=saved['progress'], cohort=str(COHORT), cohort_sha256=digest(COHORT),
          source_code_differences_reviewed=sorted(differences), code_snapshot=str(ROOT), training=False,
          arms=['E0','E1','E2_class','E2_filter','E2_rank','E2_joint'], oracle_gt_scope='context_only',
          target_scope='all6_including_context', windows=1860, scenes=312, parameter_updates=0))
    shutil.copyfile(ROOT / 'docs/fixed_query_oracle_diagnostics_20261023.md', RUN / 'EXPERIMENT.md')
    write(RUN / 'PREFLIGHT.json', dict(status='PASSED_CPU_AND_SOURCE_CHECKS', source_sha256=SOURCE_SHA,
          finite_model_tensors=632, source_progress=saved['progress'], all_reference_artifacts_present=True,
          matched_cohort=True, checked_legacy_python_files=len(source_hashes), gpu_execution_verified=False))
    print(RUN)


if __name__ == '__main__':
    main()
