"""Audit existing20-step patches; never step or modify an active training model."""
import argparse
import json
import math
from pathlib import Path
import sys
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.train_vggt_classification_readback_v1 import ARMS,SOURCE_SHA,config,slots,digest,PREFIXES
from scripts.eval_vggt_joint_short_controls_v1 import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    proof=json.loads((a.root/'preflight/PREFLIGHT_VERIFIED.json').read_text())
    assert proof['status']=='CLASSIFICATION_PREFLIGHT_VERIFIED' and proof['actual1_and4_rank_updates_verified']
    assert (a.root/'reference/recovery02/CANONICAL_REFERENCE_COMPLETE.json').exists()
    init=set();results={}
    for arm in ARMS:
        root=a.root/arm/'training';manifest=json.loads((root/'run_manifest.json').read_text())
        assert manifest['mode']=='train' and manifest['config']==config(arm)
        assert manifest['source']['checkpoint_sha256']==SOURCE_SHA
        init.update(x['source_classifier_sha256'] for x in manifest['rank_initialization'])
        updated=json.loads((root/'update_proof_0020.json').read_text())
        assert updated['status']=='REAL_UPDATES_VERIFIED' and updated['formal_updates']==20
        assert len(updated['rows'])==4 and len({x['trainable_sha256'] for x in updated['rows']})==1
        assert all(x['changed_norm']>0 and x['frozen_core_unchanged'] for x in updated['rows'])
        logs=[]
        for rank in range(4):
            rows=[json.loads(x) for x in (root/f'training_rank{rank}.jsonl').read_text().splitlines()][:20]
            assert len(rows)==20
            for i,row in enumerate(rows):
                assert row['completed_updates']==i+1 and row['completed_exposures']==(i+1)*8
                assert row['source_indices']==slots(i,rank)
                assert row['frozen_core_unchanged'] and row['actual_feedback_beta']==.05 and row['understanding_weight']==1
                assert abs(row['lr_multiplier']-(i+1)/200)<1e-14
                assert all(math.isfinite(row[k]) for k in ('loss_recon','loss_understanding','preclip_norm'))
            logs.append(rows)
        path=root/'classification_patch_update0020.pt';identity=json.loads(path.with_suffix('.json').read_text())
        assert identity['step']==20 and digest(path)==identity['sha256']
        blob=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        assert blob['completed_updates']==20 and blob['completed_exposures']==160
        assert blob['source_checkpoint_sha256']==SOURCE_SHA and blob['source_joint_exposure']==33376
        assert blob['frozen_core_unchanged'] and len(blob['rank_rng'])==4
        prefixes=PREFIXES[:2] if arm=='head_only' else PREFIXES
        assert all(n.startswith(prefixes) and torch.isfinite(v).all() for n,v in blob['parameter_patch'].items())
        assert all(int(s['step'])==20 for s in blob['optimizer']['state'].values())
        counts=blob['rank_window_counts'];assert len(counts)==4 and all(len(x)==8337 for x in counts)
        for rank in range(4):
            expected=[0]*8337
            for i in range(20):
                for idx in slots(i,rank):expected[idx]+=1
            assert counts[rank]==expected
        assert abs(logs[0][-1]['lr_multiplier']-.1)<1e-14
        results[arm]=dict(updates=20,exposures=160,checkpoint_patch_sha256=identity['sha256'],
            trainable_parameters=manifest['rank_initialization'][0]['trainable_parameters'],
            all4_rank_parameters_synchronized=True,frozen_core_unchanged=True,exact_sampling_lr_U_beta_verified=True,
            peak_GiB=max(x['peak_allocated'] for x in updated['rows'])/2**30,
            source_joint_exposure=33376,actual_feedback_peak=.05)
    assert len(init)==1
    write(a.root/'startup_validation.json',dict(status='CLASSIFICATION_STARTUP_AUDIT_PASSED',arms=results,
        same_full_source_and_classifier_verified=True,optimizer_calls_per_arm=20,new_exposures_per_arm=160,
        stored_parameter_patches_are_not_full_model_checkpoints=True,new_optimizer_updates_in_audit=0))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
