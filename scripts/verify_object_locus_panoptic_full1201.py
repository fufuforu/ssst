"""Read-only completed-training and fixed-split provenance verification."""
import collections, dataclasses, json, subprocess
from pathlib import Path
import numpy as np
import torch
from scripts import object_locus_panoptic_full1201_runtime as rt

TRAIN_SHA = 'b2624d57ea9ad5a73fd6a8375bafbe4e4c263c9c'

def main():
    torch.set_num_threads(4)
    manifest, plan, hashes, provenance = rt.assets()
    assert (manifest['S'], manifest['N'], plan['U'], plan['P']) == (1191, 8337, 1043, 7)
    complete = json.loads((rt.REPORT/'training_complete.json').read_text())
    assert (complete['epochs'], complete['updates'], complete['exposures']) == (8, 8344, 66752)
    assert complete['window_counts'] == plan['actual_expected_counts']
    assert sum(complete['window_counts']) == 66752 and sum(n-8 for n in complete['window_counts']) == 56
    counts = np.zeros(8337, dtype=np.int64)
    for rank in range(8):
        n = 0
        with (rt.REPORT/f'training_rank{rank}.jsonl').open() as f:
            for line in f:
                row = json.loads(line); e, k = divmod(n, 1043)
                assert row['window_id'] == plan['orders'][e][8*k+rank]
                assert np.isfinite([row['loss_recon'], row['loss_understanding']]).all()
                counts[row['window_id']] += 1; n += 1
        assert n == 8344
    assert counts.tolist() == complete['window_counts']
    model, opt = rt.base.build_model('cpu', report=False)
    records = []
    for epoch in (0,1,2,4,6,8):
        path = rt.RUN/f'checkpoint_epoch_{epoch:02}.pt'
        blob = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        assert blob['git_sha'] == TRAIN_SHA and blob['science_sha'] == rt.BASE
        assert blob['manifest_sha256'] == hashes['manifest.json'] and blob['plan_sha256'] == hashes['training_plan.json']
        assert blob['completed_updates'] == epoch*1043 and blob['completed_exposures'] == epoch*8344
        assert blob['config'] == dataclasses.asdict(opt)
        assert all(torch.isfinite(v).all() for v in blob['model'].values())
        model.load_state_dict(blob['model'], strict=True)
        record = dict(epoch=epoch, path=str(path), sha256=rt.sha(path), bytes=path.stat().st_size,
                      updates=blob['completed_updates'], exposures=blob['completed_exposures'], model_tensors=len(blob['model']), strict_load=True, finite=True)
        records.append(record); print(json.dumps(record), flush=True); del blob
    train_frames = collections.defaultdict(set)
    identities = {(w['scene'],tuple(w['context']),tuple(w['novel'])) for w in manifest['windows']}
    for w in manifest['windows']: train_frames[w['scene']].update(w['context']+w['novel'])
    exposure = []
    for split, windows in manifest['monitor_splits'].items():
        for index, w in enumerate(windows):
            intersection = sorted(train_frames[w['scene']] & set(w['context']+w['novel']))
            exposure.append(dict(split=split,monitor_index=index,**w, exact_training_window=(w['scene'],tuple(w['context']),tuple(w['novel'])) in identities,
                                 training_scene=w['scene'] in manifest['actual_train_scenes'], exposed_frames=intersection,
                                 label='历史holdout，本轮存在帧曝光' if split.startswith('same_scene_holdout') and intersection else 'holdout' if split.startswith('same_scene_holdout') else 'fixed monitor'))
    rt.base.write_json(rt.REPORT/'fixed_window_exposure.json', exposure)
    slurm = subprocess.check_output(['sacct','-j','58248','-X','--parsable2','--format=JobID,State,ExitCode,Start,End,Elapsed,NodeList'],text=True)
    rt.base.write_json(rt.REPORT/'training_verification.json',dict(status='PASS',slurm=slurm,training_sha=TRAIN_SHA,science_sha=rt.BASE,
        S=1191,N=8337,U=1043,P=7,epochs=8,updates=8344,exposures=66752,padding_exposures=56,all_rank_logs_verified=True,
        checkpoint_records=records,asset_hashes=hashes,additional_optimizer_updates=0))

if __name__ == '__main__': main()
