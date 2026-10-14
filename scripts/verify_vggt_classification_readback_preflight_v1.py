"""Validate actual1/4-worker updates and native official scoring before training."""
import argparse
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.train_vggt_classification_readback_v1 import ARMS,config


def main():
    from scripts.invoke_siu3r_official_evaluator import evaluate,SIU3R_COMMIT
    from scripts.format_object_locus_posefree_siu3r_table import table_values
    from scripts.eval_vggt_joint_short_controls_v1 import write
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    sha=None;initial_classifier=set();capacity={};answers=[]
    for stage,world in [('single',1),('four',4)]:
        for arm in ARMS:
            root=a.root/stage/arm;done=json.loads((root/'COMPLETE.json').read_text());manifest=json.loads((root/'run_manifest.json').read_text())
            assert done['status']=='CLASSIFICATION_SMOKE_PASSED' and done['formal_updates']==0
            assert done['world_size']==world and manifest['config']==config(arm)
            sha=done['execution_sha'] if sha is None else sha;assert done['execution_sha']==sha
            proof=json.loads((root/'update_proof_0002.json').read_text())
            assert proof['status']=='REAL_UPDATES_VERIFIED' and len(proof['rows'])==world
            assert len({r['trainable_sha256'] for r in proof['rows']})==1
            assert all(r['changed_norm']>0 and r['frozen_core_unchanged'] for r in proof['rows'])
            initial_classifier.update(r['source_classifier_sha256'] for r in manifest['rank_initialization'])
            capacity[arm]=manifest['rank_initialization'][0]['trainable_parameters']
            for rank in range(world):
                initial=json.loads((root/f'initial_rank{rank:02d}/native_reproduction.json').read_text())
                final=json.loads((root/f'final_rank{rank:02d}/native_reproduction.json').read_text())
                assert initial['segmentation_reproduced'] and final['RGB_depth_and_GT_reproduced']
                assert len(list((root/f'final_rank{rank:02d}/all').glob('scene*')))==1
            raw=evaluate(root/'final_rank00/all',device='cuda:0')
            report=dict(scopes={'target-all':{k:raw[k] for k in ('absrel','rmse','psnr','ssim','lpips')}},official_segmentation={'all':raw})
            values=table_values(report)
            answers.append(dict(stage=stage,arm=arm,values=values,official_commit=SIU3R_COMMIT,optimizer_updates=0))
    assert len(initial_classifier)==1 and capacity['object_support']==capacity['scene_support']
    write(a.root/'PREFLIGHT_VERIFIED.json',dict(status='CLASSIFICATION_PREFLIGHT_VERIFIED',execution_sha=sha,
        actual1_and4_rank_updates_verified=True,initial_classifier_identical=True,
        zero_initial_native_segmentation_reproduced=True,post_update_native_RGB_depth_reproduced=True,
        equal_C_D_capacity=True,trainable_parameter_counts=capacity,official_13_column_scores=answers,formal_updates=0))
    print('CLASSIFICATION_PREFLIGHT_VERIFIED',capacity,flush=True)


if __name__=='__main__':main()
