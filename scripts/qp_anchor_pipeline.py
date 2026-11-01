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
    assert read(train/'OPTIMIZER_ENDPOINT_AUDIT.json')['status']=='PASSED'
    source=train/'checkpoint_final_live.pt';saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
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
    result['model_changes']=cfg['model_changes'];result['native_control_root']=cfg['native_control_root']
    control=Path(cfg['native_control_root'])/'eval'
    if (control/'evaluation/metrics.json').exists():
        assert signatures(ev)==signatures(control) and gt(ev)==gt(control)
        control_metrics=metric(read(control/'evaluation/metrics.json')['official_native_result'])
        result['QPnative_control']=control_metrics;result['delta_from_QPnative']={k:current[k]-control_metrics[k] for k in current}
    else:result['QPnative_control_pending']=True
    write(run/'RESULTS.json',result)
    rows=[dict(model='Baseline4_1024_Native',**base),dict(model=cfg['recipe'],**current),dict(model='delta_from_Baseline4_1024',**result['delta_from_baseline1024'])]
    with (run/'formal_11metrics_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(run/'FORMAL_COMPLETE.json',dict(status='PASSED',windows=1860,scenes=312,matched_inputs_and_gt=True,source_checkpoint_sha256=checkpoint['sha256']))





def pair(base):
    a,b=base/'v1',base/'v2';cfg=read(a/'configuration.json');control=Path(cfg['native_control_root']);reference=Path(cfg['baseline4_root'])
    from scripts.olda4_ablation_pipeline import signatures,gt
    for run in [a,b,control]:assert read(run/'FORMAL_COMPLETE.json')['status']=='PASSED'
    for run in [b,control,reference]:assert signatures(a/'eval')==signatures(run/'eval') and gt(a/'eval')==gt(run/'eval')
    scores={label:metric(read(root/'eval/evaluation/metrics.json')['official_native_result']) for label,root in [('Baseline4',reference),('QPnative',control),('QPanchor1',a),('QPanchor2',b)]}
    rows=[dict(model=k,**v) for k,v in scores.items()]
    rows += [dict(model=label,**{k:scores[left][k]-scores[right][k] for k in scores[left]}) for label,left,right in [('v1_minus_native','QPanchor1','QPnative'),('v2_minus_native','QPanchor2','QPnative'),('v2_minus_v1','QPanchor2','QPanchor1')]]
    with (base/'PAIR_11METRICS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    novel={label:read(root/'eval/novel_metrics.json')['result'] for label,root in [('QPnative',control),('QPanchor1',a),('QPanchor2',b)]}
    write(base/'PAIR_RESULTS.json',dict(status='PASSED',scores=scores,novel4=novel,deltas=rows[4:],matched_inputs_and_GT=True,windows=1860,scenes=312,training_budget='each fresh4epochs /4172updates; same1024tokens /2048width /native final objective',extra_intermediate_supervision=False))
    lines=['# QPnative geometric anchor pair','', '|Model|'+ '|'.join(scores['QPnative'])+'|','|---|'+'---:|'*11]
    lines += ['|'+r['model']+'|'+'|'.join(f'{r[k]:.6f}' for k in scores['QPnative'])+'|' for r in rows]
    lines += ['', 'PSNR/SSIM/mIoU/PQ/mAP: higher is better. LPIPS/AbsRel/RMSE: lower is better.',
        'v1: scene-conditioned tokens and anchor-local XYZ. v2: v1 plus12-layer center/radius updates and anchor-ray attention bias.',
        'Both use the native final-output objective; no intermediate reconstruction loss, no GT geometry in generation, no test-selected weights.',
        'Depth uses SIU3R scale+shift alignment. Semantic mIoU is not text mIoU_t. One seed; no significance claim.']
    (base/'PAIR_RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    run=Path(sys.argv[1]);dict(freeze=freeze,novel=novel,finish=finish,pair=pair)[sys.argv[2]](run)
