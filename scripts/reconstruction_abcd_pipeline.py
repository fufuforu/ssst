"""Endpoint gates, pinned checkpoint, fair Native comparison and summary."""
from pathlib import Path
import sys,json,os,csv,math,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.olda4_ablation_pipeline import signatures,gt


def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def freeze(run):
    import torch
    cfg=read(run/'configuration.json');arm=cfg['arm'];complete=read(run/'train/TRAINING_COMPLETE.json')
    end=4172 if arm=='D' else 9387
    assert complete['status']=='PASSED' and complete['full_updates']==end and complete['world_size']==8
    for i in range(8):
        init=read(run/f'train/INITIALIZATION_rank{i:02d}.json');assert init['status']=='PASSED' and init['source_sha256']==cfg['source_sha256']
        proof=read(run/f'train/UPDATE_AUDIT_{4172 if arm=="D" else 1043}_rank{i:02d}.json')
        assert proof['status']=='PASSED' and proof['protected_tensors_equal'] and proof['ddp_replicas_equal']
    checkpoint=run/'train/checkpoint_latest.pt';saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    assert saved['progress']==dict(epoch=4 if arm=='D' else 9,update_in_epoch=0,full_updates=end)
    dest=run/'eval/source_checkpoint.pt'
    if dest.exists():raise ValueError('Refusing duplicate freeze')
    os.link(checkpoint,dest)
    write(run/'eval/FINAL_CHECKPOINT_FROZEN.json',dict(status='PASSED',sha256=sha(dest),progress=saved['progress'],parameter_updates_during_export=0))

def novel(run):
    # Relative links select ONLY the four unseen target frames from each
    # previously exported pair; scoring reuses the unchanged official evaluator.
    export=read(run/'eval/evaluation/export_complete.json');root=run/'eval/novel';root.mkdir(exist_ok=False)
    assert len(export['records'])==1860
    for record in export['records']:
        frames=record['frame_ids'];assert len(frames)==6
        name=record['name'];scene=name.split('_context')[0]
        pair=run/'eval/evaluation/all'/name
        for sub in ('rgb','rgb_gt','depth','depth_gt'):
            target=root/name/sub;target.mkdir(parents=True,exist_ok=True)
            for frame in frames[2:]:
                filename=f'{scene}_{frame}.png';source=pair/sub/filename
                assert source.is_file();(target/filename).symlink_to(os.path.relpath(source,target))
    write(run/'eval/NOVEL_PROTOCOL.json',dict(windows=1860,scenes=312,frames_per_window=4,context_excluded=True,source='fixed official PNG export',selection='frame_ids[2:]'))

def metric(raw):
    out={k:raw[k] for k in ('absrel','rmse','psnr','ssim','lpips')}
    for scope in ('context','target'):
        for key in ('miou','pq'):out[scope+'_'+key]=100*raw[scope+'_'+key]
        out[scope+'_map']=100*raw[scope+'_map']['map']
    assert len(out)==11 and all(math.isfinite(v) for v in out.values())
    return out

def finish(run):
    cfg=read(run/'configuration.json');ev=run/'eval';m=read(ev/'evaluation/metrics.json');identity=read(ev/'evaluation/checkpoint_identity.json')
    assert m['status']=='OFFICIAL_VAL_SCORED' and m['windows']==1860 and not m['nonfinite_native_paths']
    assert identity['parameter_updates']==0 and identity['smoke_only'] is False
    assert identity['meta']['code_sha']==cfg['code_sha'] and sha(ev/'source_checkpoint.pt')==read(ev/'FINAL_CHECKPOINT_FROZEN.json')['sha256']
    reference=Path(cfg['baseline4_root'] if cfg['arm']=='D' else cfg['baseline8_root'])/'eval'
    assert signatures(ev)==signatures(reference) and gt(ev)==gt(reference)
    exported=read(ev/'evaluation/export_complete.json');assert len({r['official_pair']['scene'] for r in exported['records']})==312
    result=dict(status='PASSED',arm=cfg['arm'],native=metric(m['official_native_result']),reference=metric(read(reference/'evaluation/metrics.json')['official_native_result']),novel=read(ev/'novel_metrics.json'),checkpoint_sha256=read(ev/'FINAL_CHECKPOINT_FROZEN.json')['sha256'],matched_inputs_and_gt=True,parameter_updates_during_export=0,windows=1860,scenes=312,training=read(run/'train/TRAINING_COMPLETE.json'))
    result['delta_from_reference']={k:result['native'][k]-result['reference'][k] for k in result['native']}
    write(run/'RESULTS.json',result)

def summarize(root):
    rows=[]
    for arm in 'ABCD':
        result=read(root/arm/'RESULTS.json');assert result['status']=='PASSED'
        rows.append(dict(model=arm,**result['native']))
    by_arm={row['model']:row for row in rows}
    for arm in ('B','C'):
        rows.append(dict(model=arm+'_minus_A',**{k:by_arm[arm][k]-by_arm['A'][k] for k in by_arm['A'] if k!='model'}))
    d=read(root/'D/RESULTS.json')
    rows.append(dict(model='Baseline4_Native',**d['reference']))
    rows.append(dict(model='D_minus_Baseline4_Native',**d['delta_from_reference']))
    with (root/'formal_11metrics_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(root/'COMPLETION_AUDIT.json',dict(status='PASSED',rows=rows,ABC_comparison='B/C versus A, same Baseline8 source and 1043 additional updates',D_comparison='D scratch4172 versus original Baseline4 Native',calibration='Native only; no head fitted',scope='target-all6; pure Novel reconstruction separately'))

if __name__=='__main__':
    path=Path(sys.argv[1]);dict(freeze=freeze,novel=novel,finish=finish,summarize=summarize)[sys.argv[2]](path)
