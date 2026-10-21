#!/usr/bin/env python3
"""Idempotent comparison and bounded result bundles; never starts training."""
from __future__ import annotations
import collections,csv,json,subprocess,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.official_locusgs_recon_runtime import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

METRICS=('psnr','ssim','lpips','absrel','rmse')

def prepare_val32():
    source=Path('/space/mawb/ssst/group_plus/object_locus_v3_set/data_manifest.json')
    windows=json.loads(source.read_text())['val32']
    if len(windows)!=32:raise RuntimeError('Locked V3 val32 has changed')
    records=[dict(scan=w['scene'],context_ids=w['context'],target_ids=w['target']) for w in windows]
    target=REPORT/'val32_manifest.json'
    if target.exists() and json.loads(target.read_text())!=records:raise RuntimeError('val32 manifest mismatch')
    write_json(target,records)
    write_json(REPORT/'val32_provenance.json',dict(source=str(source),source_sha256=sha256(source),manifest_sha256=sha256(target),selection='Existing locked V3 val32; unchanged frame IDs'))
    return target

def per_scene(directory):
    scopes={}
    for scope in ('context','novel','all'):
        rows=collections.defaultdict(list)
        for p in (directory/scope).glob('*/rgb_scores.json'):
            scene=p.parent.name.split('_context')[0]
            rows[scene].extend(json.loads(p.read_text()))
        scopes[scope]={scene:float(np.mean([r['psnr'] for r in v])) for scene,v in rows.items()}
    return scopes

def main():
    prepare_val32()
    newdir=REPORT/'eval_new_step50000';olddir=REPORT/'eval_legacy_best47500'
    if not (REPORT/'training_complete.json').exists():raise RuntimeError('Training not complete; packaging cannot fabricate outcomes')
    new=json.loads((newdir/'scope_metrics.json').read_text());old=json.loads((olddir/'scope_metrics.json').read_text())
    nk=json.loads((newdir/'view_index.json').read_text())['keys'];ok=json.loads((olddir/'view_index.json').read_text())['keys']
    if nk!=ok:raise RuntimeError('New/legacy evaluator key sets differ')
    delta={scope:{k:float(new[scope][k])-float(old[scope][k]) for k in METRICS} for scope in new}
    rows=[]
    for scope in ('context','novel','all'):
        for k in METRICS:rows.append(dict(scope=scope,metric=k,legacy47500=old[scope][k],official50000=new[scope][k],delta=delta[scope][k]))
    with (REPORT/'comparison_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    ns,olds=per_scene(newdir),per_scene(olddir);scene_rows=[];stats={}
    for scope in ns:
        if set(ns[scope])!=set(olds[scope]) or len(ns[scope])!=312:raise RuntimeError('Expected 312 matching scene summaries')
        for scene in sorted(ns[scope]):scene_rows.append(dict(scope=scope,scene=scene,new_PSNR=ns[scope][scene],legacy_PSNR=olds[scope][scene],delta=ns[scope][scene]-olds[scope][scene]))
        stats[scope]={}
        for label,values in [('new',list(ns[scope].values())),('legacy',list(olds[scope].values()))]:
            stats[scope][label]=dict(mean=float(np.mean(values)),median=float(np.median(values)),p10=float(np.percentile(values,10)),p90=float(np.percentile(values,90)))
    with (REPORT/'per_scene.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=scene_rows[0]);w.writeheader();w.writerows(scene_rows)
    d=delta
    if d['novel']['psnr']>=.3 and d['novel']['ssim']>=0 and d['novel']['lpips']<=0 and d['context']['psnr']>=-.3:
        conclusion='历史对照中重建明显提高'
    elif abs(d['novel']['psnr'])<=.3 and abs(d['context']['psnr'])<=.3:
        conclusion='PSNR 相近；SSIM/LPIPS 方向冲突时为混合结果，不声称全面等价'
    else:conclusion='逐项报告改善/退化，未满足预注册的明显提高或相近条件'
    write_json(REPORT/'comparison.json',dict(new=new,legacy=old,delta=delta,scene_statistics=stats,conclusion=conclusion,understanding_metrics='N/A',key_counts={k:len(v) for k,v in nk.items()}))
    caveats='''只有一个新 run，无 seed 方差估计；官方结构与旧 ScanNet 适配结构有多项差异。
旧 checkpoint 由 monitor 择优，新 step50000 为固定 endpoint，预算多 2500 updates。
共用解析 visibility 涉及历史实现修正；使用 GT poses，不等价于 unposed SIU3R。
这不是严格配对的单变量实验，也没有复现论文 DL3DV 训练；不自动追加实验或迁移理解底座。'''
    text='# 官方源码 LocusGS / ScanNet 历史重建对照\n\n'+conclusion+'\n\n|Scope|Metric|旧47500|新50000|Delta|\n|---|---|---:|---:|---:|\n'
    for r in rows:text+=f"|{r['scope']}|{r['metric']}|{r['legacy47500']:.6f}|{r['official50000']:.6f}|{r['delta']:+.6f}|\n"
    text+='\n'+caveats+'\n\n理解指标 mIoU/PQ/mAP/AP50：N/A。\n'
    (REPORT/'comparison_report.md').write_text(text)
    (REPORT/'README.md').write_text(text+'\n主要数据：comparison.json、comparison_metrics.csv、per_scene.csv、fixed_config.json。包不含权重、数据集或全量 PNG 导出。\n')
    train=[json.loads(line) for line in (REPORT/'training_metrics.jsonl').read_text().splitlines()]
    for name,keys in [('training_curve',['loss','psnr','grad_norm']),('layer_losses',['loss_rgb_layer6','loss_rgb_layer12','loss_gaussian_visibility_layer6','loss_gaussian_visibility_layer12','loss_anchor_visibility_layer6','loss_anchor_visibility_layer12'])]:
        fig,ax=plt.subplots(figsize=(9,4))
        for key in keys:ax.plot([x['step'] for x in train],[x[key] for x in train],label=key)
        ax.set_xlabel('Optimizer updates');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(REPORT/f'{name}.png');plt.close(fig)
    q=REPORT/'qualitative';q.mkdir(exist_ok=True)
    records=json.loads((newdir/'view_index.json').read_text())['records'][:3]
    for record in records:
        pair=record['pair'];folder=pair['scene_id']+'_context'+'_'.join(map(str,pair['context_frame_ids']))
        for scope,frame in [('context',pair['context_frame_ids'][0]),('novel',pair['novel_frame_ids'][0])]:
            file=f"{pair['scene_id']}_{frame}.png"
            fig,axs=plt.subplots(2,3,figsize=(9,6))
            for column,(label,path,sub) in enumerate([('GT',newdir,'rgb_gt'),('旧47500',olddir,'rgb'),('新50000',newdir,'rgb')]):
                rgb=np.asarray(Image.open(path/'all'/folder/sub/file));dep=np.asarray(Image.open(path/'all'/folder/('depth_gt' if column==0 else 'depth')/file))/1000.
                axs[0,column].imshow(rgb);axs[0,column].set_title(label)
                axs[1,column].imshow(dep,vmin=0,vmax=5,cmap='viridis')
            for ax in axs.flat:ax.axis('off')
            fig.suptitle(f"record{record['record']} {scope} frame{frame}; depth 0..5m");fig.tight_layout();fig.savefig(q/f"record{record['record']}_{scope}.png");plt.close(fig)
    (REPORT/'source.patch').write_bytes(subprocess.check_output(['git','diff',BASELINE,source_sha()],cwd=REPO))
    write_json(REPORT/'official_source_manifest.json',verify_vendor())
    # Only task artifacts: never recurse into exports, weights, datasets, evaluator checkout or environments.
    files=[p for p in REPORT.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.jsonl','.txt','.md','.png','.patch','.log') and not p.name.startswith(('cleanup_','tokengs_'))]
    files += list(q.glob('*.png'))
    for name in ('smoke','eval_new_step50000','eval_legacy_best47500','eval_new_step47500_val32','eval_new_step50000_val32','eval_legacy_best47500_val32'):
        base=REPORT/name
        files += [p for p in base.glob('*.json') if p.is_file()]
        files += list(base.glob('smoke_result.json'))
        for scope in ('all','context','novel'):files+=list((base/scope).glob('*/rgb_scores.json'))+list((base/scope).glob('*/depth_scores.json'))
    # Greedy parts use the actual compressed size of each independent member.
    parts=[];current=[];size=0;limit=28*1024**2
    import zlib
    for p in sorted(set(files)):
        compressed=len(zlib.compress(p.read_bytes(),6))+len(str(p))*2+256
        if compressed>limit:raise RuntimeError(f'Artifact too large for a bounded part: {p}')
        if current and size+compressed>limit:parts.append(current);current=[];size=0
        current.append(p);size+=compressed
    if current:parts.append(current)
    bundles=[]
    for i,part in enumerate(parts):
        target=REPORT/('result_bundle.zip' if i==0 else f'result_bundle_part{i+1:02d}.zip')
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in part:z.write(p,p.relative_to(REPORT))
        if target.stat().st_size>limit:raise RuntimeError('ZIP exceeds 28 MiB')
        with zipfile.ZipFile(target) as z:
            if z.testzip():raise RuntimeError('ZIP integrity failure')
        bundles.append(dict(path=str(target),sha256=sha256(target),bytes=target.stat().st_size))
    for p in q.glob('*.png'):
        with Image.open(p) as img:img.verify()
    write_json(REPORT/'bundle_index.json',dict(bundles=bundles,complete=True))
    print(json.dumps(bundles,indent=2))

if __name__=='__main__':main()
