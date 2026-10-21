"""After parent termination: reuse successful training, or recover only cache EEXIST."""
import json
import os
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt06_adapt2_joint4_node17')
SOURCE_SHA='36085703f1fe6b2426668aead700980a79680af1'


def main():
    completed=ROOT/'TRAINING_COMPLETE.json'
    if completed.exists():
        record=json.loads(completed.read_text())
        if record['full_updates']!=4172 or record.get('adaptation_updates')!=2086:raise ValueError('Incomplete staged result')
        (ROOT/'CACHE_RECOVERY_NOT_NEEDED.json').write_text(json.dumps(dict(training_already_complete=True,parameter_updates=0))+'\n')
        return
    errors=[json.loads(p.read_text()) for p in ROOT.glob('FAILED_rank*.json')]
    if not any(e.get('job_id')=='65236' and 'FileExistsError' in e.get('error','') and 'dual_stream_frozen_cache.py' in e.get('traceback','') and '/frozen_cache/' in e.get('traceback','') for e in errors):
        raise RuntimeError('Parent did not complete and failure is not the known cache publication error; refusing automatic restart')
    if any('FloatingPointError' in e.get('error','') or 'out of memory' in e.get('error','').lower() for e in errors):raise RuntimeError('Unrelated numerical/OOM failure needs inspection')
    import torch
    checkpoint=ROOT/'checkpoint_latest.pt'
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if blob['meta']['execution_git_sha']!=SOURCE_SHA:raise ValueError('Staged source SHA differs')
    archive=ROOT/'cache_failure_recovery_20261015';archive.mkdir(exist_ok=False)
    os.link(checkpoint,archive/'checkpoint_before_recovery.pt')
    for name in ('STARTED.json','configuration.json','JOB_FAILED.json','ADAPTATION_STARTED.json'):
        path=ROOT/name
        if path.exists():shutil.copy2(path,archive/name)
    for path in ROOT.glob('FAILED_rank*.json'):shutil.move(str(path),str(archive/path.name))
    receipt=dict(source_job=65236,recovery_job=os.environ.get('SLURM_JOB_ID'),source_progress=blob['progress'],source_meta=blob['meta'],
        action='resume latest durable checkpoint without resetting stages',model_optimizer_rank_rng_preserved=True)
    (archive/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    del blob
    command=['/space/mawb/anaconda3/envs/tokengs/bin/python','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',
        str(REPO/'scripts/train_dual_stream_v1.py'),'--output-dir',str(ROOT),'--architecture','gfm_v3','--adapt-epochs','2',
        '--resume','--resume-source-sha',SOURCE_SHA,'--resume-reason','Engineering cache publication fix; model/loss/data/stage clocks unchanged',
        '--gate-policy','learning_evidence_v2','--defer-export','--cache-root','/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt01/frozen_cache']
    os.chdir(REPO);os.execv(command[0],command)

if __name__=='__main__':main()
