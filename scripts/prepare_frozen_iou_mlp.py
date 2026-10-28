"""Prepare the one-head experiment without submitting jobs or modifying sources."""
import argparse,hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.frozen_iou_data_plan import SOURCE_SHA,TARGET,scene_split
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/frozen_baseline4_iou_mlp_20261023_v1')
PARENT=RUN.with_name('fixed_query_oracle_diagnostics_20261023_v4_paired')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--test-log',type=Path,required=True);args=parser.parse_args()
    if RUN.exists():raise FileExistsError('Inspect existing IoU MLP run; never duplicate preparation')
    log=args.test_log.read_text()
    assert 'Ran 7 tests' in log and re.search(r'^OK$',log,re.M) and not re.search(r'^FAILED',log,re.M)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert json.loads((PARENT/'COMPLETE.json').read_text())['status']=='FIXED_WEIGHT_DIAGNOSTICS_COMPLETE'
    assert sha(PARENT/'source_checkpoint.pt')==SOURCE_SHA
    source=json.loads((PARENT/'source_configuration.json').read_text());meta=json.loads((PARENT/'source_meta.json').read_text())
    assert meta['epochs']==4 and meta['understanding_weight']==1.0 and meta['training_view_protocol']=='2context+2novel'
    manifest=Path(source['manifest']);assert sha(manifest)==meta['manifest_sha256']
    heldout=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json')
    assert sha(heldout)==meta['heldout_sha256']
    cohort=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json')
    parent_cfg=json.loads((PARENT/'configuration.json').read_text());assert sha(cohort)==parent_cfg['cohort_sha256']
    raw=json.loads(manifest.read_text());windows=raw.get('expanded_train_windows') or raw.get('train_windows') or raw['windows']
    evaluation=json.loads(cohort.read_text())['records'];training_scenes={w['scene'] for w in windows};eval_scenes={w['scene'] for w in evaluation}
    assert len(windows)==8337 and len(training_scenes)==1191 and len(evaluation)==1860 and len(eval_scenes)==312
    split=scene_split(training_scenes,eval_scenes);assert len(split['train'])==1072 and len(split['dev'])==119
    unchanged=subprocess.check_output(['git','diff','--name-only','e7c56e9dd60683ea617184fab612226da966138d','HEAD','--','tokengs','scripts/export_unified_query_checkpoint.py','scripts/export_object_locus_v3_set_official.py','scripts/train_dual_stream_v1.py','scripts/invoke_siu3r_official_evaluator.py'],cwd=ROOT,text=True)
    assert not unchanged,'Frozen model/data/official scoring implementation changed'
    assert shutil.disk_usage(RUN.parent).free>40*1024**3
    RUN.mkdir();os.link(PARENT/'source_checkpoint.pt',RUN/'source_checkpoint.pt')
    (RUN/'source_checkpoint.sha256').write_text(SOURCE_SHA+'  '+str(RUN/'source_checkpoint.pt')+'\n')
    write(RUN/'source_meta.json',meta);write(RUN/'source_configuration.json',source)
    write(RUN/'scene_split.json',split);write(RUN/'training_cohort.json',dict(records=windows,source=str(manifest),source_sha256=sha(manifest)))
    config=dict(recipe='frozen_baseline_epoch4_updated_token_iou_mlp',code_snapshot=str(ROOT),code_sha=code_sha,seed=31415,
        source_checkpoint=str(RUN/'source_checkpoint.pt'),source_sha256=SOURCE_SHA,source_progress=dict(epoch=4,update_in_epoch=0,full_updates=4172),
        manifest=str(manifest),manifest_sha256=sha(manifest),heldout=str(heldout),heldout_sha256=sha(heldout),official_cohort=str(cohort),official_cohort_sha256=sha(cohort),
        training_cohort_sha256=sha(RUN/'training_cohort.json'),scene_split_sha256=sha(RUN/'scene_split.json'),
        train_windows=8337,training_scenes=1191,head_train_scenes=1072,head_dev_scenes=119,official_eval_windows=1860,official_eval_scenes=312,
        target_definition=TARGET,feature_key='unified_query_L12_updated',feature_dim=256,head_architecture='LayerNorm256_Linear256_GELU_Linear1_Sigmoid',
        head_trainable_parameters=66561,head_learning_rate=.001,head_weight_decay=.0001,head_batch_size=1024,head_max_epochs=100,head_patience=12,
        loss='MSE',head_selection='minimum_internal_scene_dev_MSE',training_examples='all_original_NMS_kept_queries_including_empty_and_unmatched_zero_IoU',
        inference_score='predicted_iou_direct',change_NMS=False,change_masks=False,change_classes=False,backbone_updates=0,original_optimizer_not_loaded=True)
    write(RUN/'configuration.json',config)
    shutil.copyfile(args.test_log,RUN/'cpu_tests.log');shutil.copyfile(ROOT/'docs/frozen_baseline_iou_mlp_20261023.md',RUN/'EXPERIMENT.md')
    write(RUN/'CPU_TESTS.json',dict(status='PASSED',tests=7,log=str(RUN/'cpu_tests.log'),gpu_verified=False))
    write(RUN/'PREFLIGHT.json',dict(status='PASSED',source_sha256=SOURCE_SHA,backbone_model_unchanged=True,train_official_scene_overlap=0,
        train_dev_scene_overlap=0,training_scenes=1191,head_train_scenes=1072,head_dev_scenes=119,official_scenes=312,source_progress=config['source_progress'],
        cached_targets='training_Context_GT_only',official_GT_used_for_selection=False,gpu_verified=False))
    print(RUN)


if __name__=='__main__':main()
