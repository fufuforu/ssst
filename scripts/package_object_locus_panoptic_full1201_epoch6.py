"""User-scoped epoch6 reporting/packaging only; no prediction or metric computation."""
import csv, hashlib, json, os, re, shutil, subprocess, zipfile
from pathlib import Path
from PIL import Image

R=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
REPO=Path(__file__).resolve().parents[1]
D=R/'delivery_epoch6'

def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()
def csvout(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
def copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)

def main():
    a=read(R/'scope_amendment.json');assert a['best_epoch']==6 and a['complete_full_validation_epochs']==[6]
    selection=read(R/'checkpoint_selection.json');assert selection['best_epoch']==6
    root=R/'evaluation/full_epoch06';full=read(root/'metrics.json')
    assert len(full)==6 and {(x['cohort'],x['scope']) for x in full}=={(c,s) for c in ('all','excluding_dev8_scenes') for s in ('context','target_all','novel')}
    assert all(x['epoch']==6 for x in full)
    assert read(root/'official_aggregated.json')['source_count']==1860
    assert sum(read(root/f'official_shard{s:02}_complete.json')['windows'] for s in range(8))==1860
    assert read(R/'training_verification.json')['status']=='PASS'
    D.mkdir(exist_ok=True)
    # Existing epoch8 monitoring remains valid; its full benchmark was cancelled.
    cancelled=[]
    for c in ('all','excluding_dev8_scenes'):
        for s in ('context','target_all','novel'):
            x={k:'本轮取消' for k in full[0]};x.update(epoch=8,split='full_validation',cohort=c,scope=s,status='本轮取消');cancelled.append(x)
    full=[x|{'status':'COMPLETE'} for x in full]+cancelled
    write(R/'full_validation_metrics.json',full);csvout(R/'full_validation_metrics.csv',full)
    names=('metrics_all_nodes.csv','metrics_all_nodes.json','full_validation_metrics.csv','full_validation_metrics.json',
           'checkpoint_selection.json','comparison_128.csv','comparison_128.json','training_verification.json',
           'evaluation_git_provenance.json','evaluation_sources.json','evaluation_jobs.json','evaluation_environment.json',
           'aggregation_pipeline.json','scope_amendment.json','fixed_window_exposure.json','fixed_qualitative_windows.json',
           'manifest.json','training_plan.json','full_validation_manifest.json','run_manifest.json','weights_provenance.json',
           'weights_mapping.json','optimizer_groups.json','single_smoke.json','eight_smoke.json','cleanup_128.json',
           'official_aggregation_contract.json','memoryfix_smoke_contract.json','training_complete.json','git_provenance.json')
    for name in names:copy(R/name,D/name)
    for name in ('full_sources.json','official_aggregated.json','per_scene_local.json'):
        copy(root/name,D/'evaluation_provenance'/name)
    for s in range(8):copy(root/f'per_scene_official_shard{s:02}.json',D/'evaluation_provenance'/f'per_scene_official_shard{s:02}.json')
    # Reuse already-generated statistics only. No full-validation error analysis.
    old=R/'delivery'
    for name in ('error_statistics.csv','error_statistics.json','comparison_sources.json','reference_metrics.json'):
        if (old/name).exists():copy(old/name,D/name)
    for p in (old/'per_gt').glob('*.csv'):copy(p,D/'per_gt'/p.name)
    images=[]
    for epoch in (0,1,2,4,6,8):
        for split in ('dev8','val32'):
            for p in (R/'evaluation'/f'epoch{epoch:02}'/split/f'qualitative/step_{epoch*1043:04d}'/split).glob('*.png'):
                dest=D/'qualitative'/f'epoch{epoch:02}'/split/p.name;copy(p,dest);images.append(dest)
    assert len(images)==24,'reuse all existing fixed monitoring panels'
    rows=read(R/'metrics_all_nodes.json');fmt=lambda v:f'{v:.4f}' if isinstance(v,(float,int)) else str(v)
    text=['# full1201 epoch6：收缩范围收尾','',
          '最佳checkpoint固定epoch6；dev8 true-novel official packed AP50=0.245240718126297。dev8参与选模且属于val32，不能把val32称为完全独立测试集。',
          '', '训练58248 COMPLETED、ExitCode0:0；1191scene、8337独立两context窗口；8epochs/8344updates/66752exposures，含56padding曝光。六份checkpoint严格加载及参数有限性已核验。',
          '', '本轮只完成epoch6完整验证收尾。epoch8完整验证标为“本轮取消”；7个已完成分片及全部已写盘结果保留，不从部分分片伪造完整指标。六节点固定监测结果仍有效并复用。',
          '', '全部比例指标使用0–1原始单位；PSNR为dB、SSIM为0–1、LPIPS为距离。context→all/context；target-all→all/target；true novel→novel/target。candidate与official packed指标分开。AP按原始预测重新聚合，不平均scene AP。SSIM/LPIPS为已有clamped FP32图像结果，非PNG量化图像。',
          '', '## 六节点既有监测曲线','', '|epoch|dev8 novel packed AP50|val32 novel packed AP50|','|---|---:|---:|']
    for e in (0,1,2,4,6,8):
        v=[next(x['official_ap50'] for x in rows if (x['epoch'],x['split'],x['scope'])==(e,s,'novel')) for s in ('dev8','val32')]
        text.append(f'|{e}|{fmt(v[0])}|{fmt(v[1])}|')
    text+=['','dev8在epoch6达到注册峰值，epoch8下降；val32监测峰值epoch4，未用于改选。candidate/raw覆盖改善不等于最终packed实例完全改善。',
           '', '## epoch6完整验证','', '|cohort|scope|windows|mIoU|PQ|mAP|AP50|PSNR|SSIM|LPIPS|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in full[:6]:text.append('|'+x['cohort']+'|'+x['scope']+'|'+str(x['windows'])+'|'+'|'.join(fmt(x[k]) for k in ('official_miou','official_pq','official_map','official_ap50','psnr','ssim','lpips'))+'|')
    text+=['','完整清单1860窗口/312scene；排除dev8全部8scene后1812窗口/304scene。没有训练thing数量/面积过滤，空预测及无thing GT窗口保留。原始预测和来源留在evaluation/full_epoch06；包内包含既有逐scene指标和来源。',
           '', '## 既有比较与限制','', '同窗口比较复用comparison_128.csv/json：epoch6在dev8/val32 novel packed AP50为0.2452/0.4243；128场景epoch8为0.2015/0.3902，epoch16为0.2071/0.3722，epoch64为0.1131/0.3334；Fresh128 epoch64为0.0000/0.0099。属于配方比较，训练scene、曝光分配、更新量及scheduler不同；Fresh128还涉及batch、LR组及预训练路径差异，不能归因于单个模块。',
           '', '历史holdout8的6/8、holdout16的13/16窗口有本轮帧曝光，不能用整组证明未曝光窗口泛化。训练probe/all各窗口是否实际训练已单列。dev8/val32 scene未参与训练。',
           '', '未见scene已有有效实例，仍有漏检、错分和不完整mask；context-Hungarian准确率不是novel-only匹配准确率。既有错误统计仅覆盖固定监测，未追加完整验证错误分析。PSNR/SSIM/LPIPS显示重建总体保持，固定图片仍有薄结构与细节不足。',
           '', '当前为GT camera poses；SIU3R已有unposed出处不能作同条件胜负，尚无同条件比较。旧结果缺SSIM/LPIPS等标MISSING，UNDEFINED与实测0分开；本轮取消不是0或MISSING。不补旧模型实验。',
           '', '## 复用固定图片','', '仅复制原已生成面板，未新增图片或按效果筛选；query分数s是最高thing类别posterior，不是AP排序分数。']
    for s in ('dev8','val32'):text.append(f'![epoch6 {s} fixed pair0](qualitative/epoch06/{s}/pair0.png)')
    text+=['','本轮optimizer updates=0、backward=0；未改模型、指标定义或导出协议，未重训。依赖作业自动收尾后等待用户确认，再执行人工核验与交付；不启动下一轮训练。']
    (D/'report.md').write_text('\n'.join(text)+'\n');copy(D/'report.md',R/'report.md')
    for p in (R/'slurm').iterdir():
        if p.is_file():
            lines=p.read_text(errors='replace').replace('\x00','').splitlines()
            out=D/'logs'/p.name;out.parent.mkdir(exist_ok=True);out.write_text('\n'.join(lines[:8]+['[existing log excerpt; source '+str(p)+']']+lines[-35:])+'\n')
    for name in ('cpu_contracts.log','training_verification.log'):copy(R/name,D/'logs'/name)
    (D/'code').mkdir(exist_ok=True)
    (D/'code/source.diff').write_bytes(subprocess.check_output(['git','diff','b2624d57ea9ad5a73fd6a8375bafbe4e4c263c9c','HEAD'],cwd=REPO))
    copy(Path(__file__),D/'code'/Path(__file__).name)
    status=dict(status='PACKAGED_AWAITING_USER_VERIFICATION',best_epoch=6,epoch6_full='COMPLETE',epoch8_full='本轮取消',
                monitoring='reused complete six-node results',new_analysis=False,new_images=False,optimizer_updates=0,backward=0,
                new_training_launched=False,packaging_job_id=os.environ.get('SLURM_JOB_ID'))
    write(R/'evaluation_status.json',status);write(D/'evaluation_status.json',status)
    checks=[]
    for p in sorted(D.rglob('*')):
        if not p.is_file() or p.name=='SHA256SUMS':continue
        if p.suffix=='.json':read(p)
        elif p.suffix=='.csv':
            with p.open() as f:list(csv.DictReader(f))
        elif p.suffix=='.png':
            with Image.open(p) as im:im.load()
        checks.append(f'{sha(p)}  {p.relative_to(D)}')
    for link in re.findall(r'!\[[^]]*\]\(([^)]+)\)',(D/'report.md').read_text()):assert (D/link).is_file()
    (D/'SHA256SUMS').write_text('\n'.join(checks)+'\n');copy(D/'SHA256SUMS',R/'SHA256SUMS')
    archives=[];limit=28*1024**2
    for label,files in [('results',[p for p in sorted(D.rglob('*')) if p.is_file() and p.suffix!='.png']),('images',sorted(D.rglob('*.png')))]:
        part=1;current=[]
        def pack(paths,index):
            p=R/f'epoch6_{label}_part{index:02}.zip'
            with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for f in paths:z.write(f,str(f.relative_to(D)))
            return p
        def finish(paths,index):
            p=pack(paths,index);assert p.stat().st_size<limit
            with zipfile.ZipFile(p) as z:assert z.testzip() is None
            archives.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
        for p in files:
            trial=pack(current+[p],part)
            if trial.stat().st_size>=limit:
                assert current,'single compressed file exceeds limit: '+str(p)
                finish(current,part);part+=1;current=[p]
            else:current.append(p)
        if current:finish(current,part)
    write(R/'result_archives.json',archives)
    write(R/'epoch6_package_complete.json',dict(status='PASS',zip_crc=True,csv_json_readable=True,png_decoded=True,
          relative_links=True,each_zip_under_28_MiB=True,archives=archives,optimizer_updates=0,backward=0))
    print(json.dumps(archives,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
