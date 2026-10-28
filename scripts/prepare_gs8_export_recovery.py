from pathlib import Path
import sys,json,subprocess,shutil,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gs_quality_core import SOURCE_SHA,digest,write_json
BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')
OLD=BASE/'frozen_baseline8_gs_quality_20261025_v3_runtimefix'
RUN=BASE/'frozen_baseline8_gs_quality_20261025_v4_export_recovery'


def main():
    if RUN.exists():raise FileExistsError('Existing recovery; inspect instead')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty snapshot')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    audit=json.loads((OLD/'POSTRUN_CACHE_AUDIT_20261025.json').read_text());fit=json.loads((OLD/'FIT_COMPLETE.json').read_text());cfg=json.loads((OLD/'configuration.json').read_text())
    if audit['status']!='PASSED' or audit['official_forward_hash_mismatches'] or audit['cache_windows']!=10197 or audit['official_windows']!=1860 or audit['frozen_rank_receipts_verified']!=8:raise ValueError('Cannot recover corrupted/incomplete cache')
    if fit['status']!='PASSED' or fit['arms']!=['A','C'] or digest(OLD/'official_predictions.pt')!=fit['official_predictions_sha256']:raise ValueError('Fit changed')
    tests=Path('/tmp/gs8_export_recovery_tests_20261025.log').read_text()
    if 'Ran 17 tests' not in tests or '\nOK\n' not in tests:raise ValueError('Tests incomplete')
    unchanged=['tokengs','scripts/cache_gs_quality.py','scripts/fit_gs_quality.py','scripts/score_gs8_quality.py','scripts/gs_quality_core.py','scripts/reciprocity_panoptic.py','scripts/gs_quality_runtime.py','scripts/run_gs8_quality.sh']
    if subprocess.check_output(['git','diff','--name-only','36b6b6de666399a5e1abd475957316bf924d3076','HEAD','--']+unchanged,cwd=ROOT,text=True).strip():raise ValueError('Scientific pipeline changed')
    if digest(OLD/'source_checkpoint.pt')!=SOURCE_SHA:raise ValueError('Source changed')
    if digest(cfg['renderer_path'])!=cfg['renderer_sha256']:raise ValueError('Renderer changed')
    for name in ['cache_cohort','scene_split']:
        if digest(OLD/(name+'.json'))!=cfg[name+'_sha256']:raise ValueError('Source cohort/split changed')
    if digest(cfg['official_cohort'])!=cfg['official_cohort_sha256']:raise ValueError('Formal cohort changed')
    for arm in ['A','C']:
        if digest(OLD/f'heads/{arm}/head_best.pt')!=fit['head_fits'][arm]['head_sha256']:raise ValueError('Head changed')
    if shutil.disk_usage(BASE).free<30*1024**3:raise ValueError('Need >30GiB for independent exported PNGs')
    RUN.mkdir();(RUN/'cache').symlink_to((OLD/'cache').resolve(),target_is_directory=True);(RUN/'heads').symlink_to((OLD/'heads').resolve(),target_is_directory=True)
    for name in ['source_checkpoint.pt','official_predictions.pt']:os.link(OLD/name,RUN/name)
    for name in ['FIT_COMPLETE.json','QUALITY_REGRESSION.json','CONSISTENCY_DIAGNOSTICS.json','scene_split.json','cache_cohort.json']:shutil.copyfile(OLD/name,RUN/name)
    (RUN/'source_checkpoint.sha256').write_text(SOURCE_SHA+'  '+str(RUN/'source_checkpoint.pt')+'\n');(RUN/'code.sha').write_text(code+'\n')
    cfg.update(code_snapshot=str(ROOT),code_sha=code,cache_origin_run=str(OLD),head_fit_origin_run=str(OLD),retry_of_export_job=71664,
       recovery_stages=['export','score'],retrain=False,regenerate_cache=False,cache_hash_audit=str(OLD/'POSTRUN_CACHE_AUDIT_20261025.json'),
       cache_hash_audit_sha256=digest(OLD/'POSTRUN_CACHE_AUDIT_20261025.json'),hash_read_policy='At most two reads; original frozen hash must match; otherwise fail with path and observed hashes',
       prior_export_failure_cause='Hash check failed during export; all files independently match now. Exact transient cause unresolved.')
    write_json(RUN/'configuration.json',cfg);shutil.copyfile('/tmp/gs8_export_recovery_tests_20261025.log',RUN/'cpu_tests.log')
    write_json(RUN/'PREFLIGHT.json',dict(status='PASSED',code_sha=code,source_sha256=SOURCE_SHA,tests=17,old_cache_and_head_fits_reused=True,
       all_1860_forward_files_match_original_hashes=True,training_updates=0,backbone_updates=0,recovery_GPU_execution_verified=False,
       cache_manifest_sha256=digest(OLD/'cache/export_complete.json'),official_predictions_sha256=fit['official_predictions_sha256']))
    print(json.dumps(dict(run=str(RUN),code_sha=code,status='PASSED',stages=['export','score'],training_repeated=False),indent=2))


if __name__=='__main__':main()
