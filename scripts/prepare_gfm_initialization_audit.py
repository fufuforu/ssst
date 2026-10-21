"""Lock existing checkpoints and pre-existing qualification cohort; no inference/training."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.train_dual_stream_v1 import cohorts,MANIFEST,HELDOUT,digest
BASE=Path('/space/mawb/ssst/group_plus/dual_stream_training_v1/attempt13_geometric_sam_weight005_node18_8gpu')
RUN=Path('/space/mawb/ssst/group_plus/gfm_initialization_audit_v1/attempt01')
REFERENCE=Path('/space/mawb/ssst/group_plus/gfm_instance_diagnostics_v1/attempt02/full')
SOURCES=(('zero','checkpoint_initial.pt',0),('epoch1','checkpoint_epoch_01.pt',1043),('epoch2','checkpoint_epoch_02.pt',2086),('epoch4','checkpoint_epoch_04.pt',4172))


def main():
    if (RUN/'preflight.json').exists():raise ValueError('Preflight already exists; preserve original source identity')
    reference=[];rgb={}
    for path in sorted((REFERENCE/'weight005').glob('rank*/COMPLETE.json')):
        complete=json.loads(path.read_text())
        for idx in complete['records']:
            row=json.loads((path.parent/f'{idx}.json').read_text());rgb[idx]=row['input_rgb_sha256'];n=len(row['gt_ids'])
            if not n:continue
            matrix=torch.tensor(row['mask_IoU']['selected_sam']).reshape(-1,n)
            if not len(matrix):continue
            values,queries=matrix.max(0)
            for k in range(n):
                if values[k]>=.5:reference.append(dict(source_index=idx,gt_id=row['gt_ids'][k],reference_seed_query=int(queries[k]),support_iou=float(values[k])))
    if len(reference)!=152 or len(rgb)!=128:raise ValueError('Pre-existing qualification cohort differs')
    train,indices,validation=cohorts();records=[];config=None
    for stage,name,steps in SOURCES:
        path=BASE/name;h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(16*1024*1024),b''):h.update(block)
        blob=torch.load(path,map_location='cpu',weights_only=False,mmap=True);meta=blob['meta'];progress=blob['progress']
        if progress['full_updates']!=steps or progress['short_updates'] or progress.get('adapt_updates',0):raise ValueError('Checkpoint clock differs')
        if meta['manifest_sha256']!=digest(MANIFEST) or meta['heldout_sha256']!=digest(HELDOUT):raise ValueError('Data identity differs')
        if config is None:config=meta['config']
        if meta['config']!=config or config.get('understanding_loss_scale')!=.05 or config.get('query_pooling')!='geometric_sam_v4':raise ValueError('Model recipe differs')
        if steps==0 and blob['optimizer']['state']:raise ValueError('Zero checkpoint has optimizer history')
        stat=path.stat();records.append(dict(stage=stage,path=str(path),sha256=h.hexdigest(),size=stat.st_size,mtime_ns=stat.st_mtime_ns,full_updates=steps))
        del blob
    RUN.mkdir(parents=True,exist_ok=True)
    plan=dict(status='PASSED',parameter_updates=0,model='completed V4 whole-understanding-weight0.05',sources=records,
        heldout_windows=128,qualified_reference_GT=reference,heldout_rgb=rgb,
        train_probe_windows=[train[i] for i in indices],train_probe_source_indices=indices,
        interventions=['native','embedding_fallback','geometry_global','both_fallback'],
        limitations=['No intermediate step checkpoints before epoch1 survive; no training replay.',
            'Gradient probes on fixed training samples at saved checkpoints are not historical optimizer gradients.',
            'Post-initializer interventions are out-of-distribution diagnostic forwards, not trained ablation performance.',
            'Geometry_global changes only centers/scales used for Anchor reading; geometry encoded in q is retained.',
            'Candidate identities use mask hashes; changed cross-view merges are reported rather than matched by query index.'])
    (RUN/'preflight.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(status='PASSED',checkpoints=[r['stage'] for r in records],qualified_GT=len(reference),validation_windows=128,train_gradient_windows=len(indices)),indent=2))


if __name__=='__main__':main()
