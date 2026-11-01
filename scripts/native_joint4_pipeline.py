"""Immutable checkpoints, fixed1860 exports, Native eleven-metric comparisons."""
from pathlib import Path
import sys,os,json,csv,math,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def freeze(run):
    import torch
    cfg=read(run/'configuration.json');train=Path(cfg['training_root']);receipt=read(train/'TRAINING_COMPLETE.json')
    assert receipt['status']=='PASSED' and receipt['full_updates']==4172 and receipt['world_size']==8
    for rank in range(8):
        file=f'UPDATE_AUDIT_rank{rank:02d}.json' if cfg['kind']=='native_joint' else f'UPDATE_AUDIT_4172_rank{rank:02d}.json'
        proof=read(train/file);assert proof['status']=='PASSED'
        if cfg['kind']=='native_joint':
            assert proof['DDP_replicas_equal'] and proof['frozen_versions_equal'] and all(proof['required_changed'].values())
            frozen=read(train/f'FROZEN_ENDPOINT_rank{rank:02d}.json')
            assert frozen['status']=='PASSED' and frozen['contents_equal'] and frozen['versions_equal_since_ddp']
            assert frozen['before_ddp_sha256']==frozen['after_ddp_sha256']==frozen['endpoint_sha256']
            assert read(train/f'ISOLATION_rank{rank:02d}.json')['status']=='PASSED'
        else:assert proof['protected_tensors_equal'] and proof['ddp_replicas_equal']
    source=train/'checkpoint_latest.pt';saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    assert saved['progress']==dict(epoch=4,update_in_epoch=0,full_updates=4172)
    assert saved['meta']['code_sha']==cfg['source_code_sha']
    ev=run/'eval';target=ev/'source_checkpoint.pt';assert not target.exists();os.link(source,target)
    fingerprint=sha(target)
    if cfg['kind']=='baseline4096':assert fingerprint==cfg['source_checkpoint_sha256']
    write(ev/'FINAL_CHECKPOINT_FROZEN.json',dict(status='PASSED',source=str(source),sha256=fingerprint,progress=saved['progress'],parameter_updates_during_export=0))


def make_novel(source,root):
    exported=read(source/'evaluation/export_complete.json');assert len(exported['records'])==1860
    root.mkdir(exist_ok=False)
    for r in exported['records']:
        assert len(r['frame_ids'])==6
        name=r['name'];scene=name.split('_context')[0]
        for sub in ('rgb','rgb_gt','depth','depth_gt'):
            dest=root/name/sub;dest.mkdir(parents=True)
            for frame in r['frame_ids'][2:]:
                file=f'{scene}_{frame}.png';original=source/'evaluation/all'/name/sub/file
                assert original.is_file();(dest/file).symlink_to(os.path.relpath(original,dest))


def novel(run):
    ev=run/'eval';make_novel(ev,ev/'novel')
    cfg=read(run/'configuration.json')
    if cfg['kind']=='baseline4096':make_novel(Path(cfg['baseline4_root'])/'eval',ev/'reference_novel')
    write(ev/'NOVEL_PROTOCOL.json',dict(windows=1860,scenes=312,frames_per_window=4,context_excluded=True,selection='official frame_ids[2:]'))


def metric(raw):
    result={k:raw[k] for k in ['absrel','rmse','psnr','ssim','lpips']}
    for s in ['context','target']:
        for k in ['miou','pq']:result[s+'_'+k]=100*raw[s+'_'+k]
        result[s+'_map']=100*raw[s+'_map']['map']
    assert all(math.isfinite(x) for x in result.values());return result


def finish(run):
    from scripts.olda4_ablation_pipeline import signatures,gt
    cfg=read(run/'configuration.json');ev=run/'eval';reference=Path(cfg['baseline4_root'])/'eval'
    m=read(ev/'evaluation/metrics.json');identity=read(ev/'evaluation/checkpoint_identity.json')
    assert m['status']=='OFFICIAL_VAL_SCORED' and m['windows']==1860 and not m['nonfinite_native_paths']
    assert identity['parameter_updates']==0 and not identity['smoke_only']
    assert identity['meta']['code_sha']==cfg['source_code_sha']
    assert signatures(ev)==signatures(reference) and gt(ev)==gt(reference)
    records=read(ev/'evaluation/export_complete.json')['records'];assert len({r['official_pair']['scene'] for r in records})==312
    checkpoint=read(ev/'FINAL_CHECKPOINT_FROZEN.json')
    assert sha(ev/'source_checkpoint.pt')==checkpoint['sha256']
    current=metric(m['official_native_result']);base=metric(read(reference/'evaluation/metrics.json')['official_native_result'])
    result=dict(status='PASSED',recipe=cfg['recipe'],native=current,baseline1024=base,
        delta_from_baseline1024={k:current[k]-base[k] for k in current},novel=read(ev/'novel_metrics.json'),
        checkpoint_sha256=checkpoint['sha256'],windows=1860,scenes=312,matched_inputs_and_gt=True,
        parameter_updates_during_export=0,quality_head=False,text_language_head=False,
        training=read(Path(cfg['training_root'])/'TRAINING_COMPLETE.json'),evaluation_weights='live final epoch4; EMA separate, no test selection')
    reference_novel=ev/'reference_novel_metrics.json' if cfg['kind']=='baseline4096' else Path(cfg['baseline4096_eval'])/'eval/reference_novel_metrics.json'
    if reference_novel.exists():result['baseline1024_novel']=read(reference_novel)
    write(run/'RESULTS.json',result)
    rows=[dict(model='Baseline4_1024_Native',**base),dict(model=cfg['recipe'],**current),dict(model='delta_from_Baseline4_1024',**result['delta_from_baseline1024'])]
    with (run/'formal_11metrics_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(run/'FORMAL_COMPLETE.json',dict(status='PASSED',windows=1860,scenes=312,matched_inputs_and_gt=True,source_checkpoint_sha256=checkpoint['sha256']))


if __name__=='__main__':
    run=Path(sys.argv[1]);dict(freeze=freeze,novel=novel,finish=finish)[sys.argv[2]](run)
