"""Reporting, comparisons, exact fixed galleries and verified size-limited ZIPs."""
import collections,csv,hashlib,json,math,re,shutil,subprocess,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.summarize_object_locus_panoptic_full1201 import rows_for,metric
from scripts.eval_object_locus_panoptic_full1201 import EVAL,EPOCHS,SCOPES,csvout,read,write
from scripts import object_locus_panoptic_full1201_runtime as rt

OLD=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_v1_8gpu')
FRESH=Path('/space/mawb/ssst/group_plus/object_locus_v3_set_fresh128')
D=rt.REPORT/'delivery'

def normalized(value):
    if isinstance(value,dict):return {k:normalized(v) for k,v in value.items()}
    if isinstance(value,list):return [normalized(v) for v in value]
    if value is None:return 'MISSING'
    return metric(value)

def compare(rows):
    old_manifest=read(OLD/'data_manifest.json');m=read(rt.REPORT/'manifest.json');references=[];sources=[]
    def identities(ws):return [(w['scene'],w['context'],w['novel']) for w in ws]
    for epoch in (8,16,64):
        p=OLD/f'eval_epoch{epoch:02}.json';node=read(p);sources.append(dict(model='Panoptic128',epoch=epoch,path=str(p),sha256=rt.sha(p)))
        for split,windows in m['monitor_splits'].items():
            assert identities(windows)==identities(old_manifest[split])
            references.extend([r|dict(model='Panoptic128',source_epoch=epoch) for r in rows_for(epoch,split,node['results'][split])])
    fresh=read(FRESH/'eval_node_epoch_64.json')['splits'];fmanifest=read(FRESH/'data_manifest.json')
    for split,windows in m['monitor_splits'].items():
        alias='original_train_all56' if split=='train_all56' else split
        if alias not in fresh:continue
        assert identities(windows)==identities(fmanifest[alias])
        references.extend([r|dict(model='Fresh128',source_epoch=64) for r in rows_for(64,split,fresh[alias])])
    previous=read(OLD/'delivery/fresh128_comparison.json')
    assert previous['source_sha256']==rt.sha(FRESH/'eval_node_epoch_64.json')
    for old in previous['rows']:
        if old['split']!='same_scene_holdout8':continue
        ref={k:'MISSING' for k in rows[0]}
        ref.update(epoch=64,source_epoch=64,model='Fresh128',split=old['split'],scope=old['scope'],windows=8)
        for k,v in old.items():
            if k.startswith('fresh128_'):ref[k[len('fresh128_'):]]=v
        references.append(ref)
    sources.append(dict(model='Fresh128 holdout8 persisted subset',path=str(OLD/'delivery/fresh128_comparison.json'),sha256=rt.sha(OLD/'delivery/fresh128_comparison.json')))
    sources.append(dict(model='Fresh128',epoch=64,path=str(FRESH/'eval_node_epoch_64.json'),sha256=rt.sha(FRESH/'eval_node_epoch_64.json')))
    result=[];keys=('official_miou','official_pq','official_map','official_ap50','candidate_map','candidate_ap50','raw_mask_coverage','matched_class_accuracy','candidate_cw_recall','psnr','ssim','lpips')
    for r in rows:
        for ref in references:
            if (r['split'],r['scope'])!=(ref['split'],ref['scope']):continue
            row=dict(full_epoch=r['epoch'],split=r['split'],scope=r['scope'],reference_model=ref['model'],reference_epoch=ref['source_epoch'])
            for k in keys:
                row['full_'+k]=r[k];row['reference_'+k]=ref[k]
                row['delta_'+k]=r[k]-ref[k] if isinstance(r[k],(int,float)) and isinstance(ref[k],(int,float)) else 'MISSING'
            row['recipe_comparison']=True;result.append(row)
    write(rt.REPORT/'comparison_128.json',result);csvout(rt.REPORT/'comparison_128.csv',result)
    write(D/'comparison_sources.json',sources);write(D/'reference_metrics.json',references)
    return result,references

def gt_tables():
    statistics=[];files=[]
    selected=read(rt.REPORT/'checkpoint_selection.json')['full_evaluation_epochs']
    roots=[p for e in EPOCHS for p in sorted((EVAL/f'epoch{e:02}').iterdir()) if (p/'complete.json').exists()]
    roots += [p for e in selected for p in sorted((EVAL/f'full_epoch{e:02}').glob('shard*')) if (p/'complete.json').exists()]
    for root in roots:
        done=read(root/'complete.json');result=read(root/'result.json')
        with (root/'per_gt.csv').open() as f: gt=list(csv.DictReader(f))
        # Original registered evaluator writes GT rows window->scope->GT. Restore
        # exact window identity from its saved scope counts without re-inference.
        offset=0
        for w in result['windows']:
            for scope in SCOPES:
                count=w['scopes'][scope]['gt_count']
                for row in gt[offset:offset+count]:row.update(context_frame_ids=json.dumps(w['context']),novel_frame_ids=json.dumps(w['novel']))
                offset+=count
        assert offset==len(gt)
        name=root.parent.name+'_'+root.name;dest=D/'per_gt'/f'{name}.csv';csvout(dest,gt);files.append(dest)
        for scope in SCOPES:
            rr=[r for r in gt if r['scope']==scope];n=len(rr)
            number=lambda r,k:float(r[k]) if r.get(k) not in ('','None','UNDEFINED') else None
            stat=dict(node=root.parent.name,split=root.name,scope=scope,GT=n,
                      raw_mask_below_0_5=sum(float(r['best_raw_iou'])<.5 for r in rr),
                      eligible_candidate_miss=sum(float(r['best_eligible_candidate_iou'])<.5 for r in rr),
                      panoptic_miss=sum(float(r['best_panoptic_iou'])<.5 for r in rr),
                      raw_best_class_error=sum(r['best_query_class']!=r['class'] for r in rr),
                      raw_best_class_denominator=n,
                      raw_best_no_object=sum(r['best_query_class_argmax19']=='18' for r in rr),
                      eligible_best_class_error=sum(float(r['best_eligible_candidate_iou'])>=.5 and r['best_eligible_candidate_class']!=r['class'] for r in rr),
                      eligible_best_class_denominator=sum(float(r['best_eligible_candidate_iou'])>=.5 for r in rr),
                      duplicate_GT=sum(int(r['eligible_queries_iou_ge_0_5'])>1 for r in rr),
                      raw_GT_not_fully_covered=sum(number(r,'raw_recall') is not None and number(r,'raw_recall')<1 for r in rr),
                      raw_prediction_outside_GT=sum(number(r,'raw_precision') is not None and number(r,'raw_precision')<1 for r in rr),
                      context_hungarian_matched_no_object=sum(r.get('matched_query') not in ('','None') and r.get('matched_class') in ('','None') for r in rr),
                      matching_source='raw/eligible masks: this render scope; Hungarian: context matching',
                      mask_analysis='area diagnostics, not a new eligibility threshold: raw recall<1 / precision<1 indicate incomplete/outside GT; visual inspection distinguishes crossing objects')
            statistics.append(stat)
    for node in sorted({r['node'] for r in statistics if r['node'].startswith('full_epoch')}):
        for scope in SCOPES:
            rr=[r for r in statistics if r['node']==node and r['scope']==scope]
            total={k:sum(r[k] for r in rr) for k,v in rr[0].items() if isinstance(v,int)}
            statistics.append(dict(node=node,split='full_validation_all',scope=scope,**total,
                                   matching_source=rr[0]['matching_source'],mask_analysis=rr[0]['mask_analysis']))
    write(D/'error_statistics.json',statistics);csvout(D/'error_statistics.csv',statistics)
    return statistics

def build_report(rows,full,refs,errors):
    verification=read(rt.REPORT/'training_verification.json');selection=read(rt.REPORT/'checkpoint_selection.json');best=selection['best_epoch']
    fmt=lambda x:f'{x:.4f}' if isinstance(x,(int,float)) else str(x)
    def row(e,s,scope='novel'):return next(r for r in rows if (r['epoch'],r['split'],r['scope'])==(e,s,scope))
    curves=[row(e,'dev8') for e in EPOCHS];val=[row(e,'val32') for e in EPOCHS]
    lower=row(8,'dev8')['official_ap50']<row(best,'dev8')['official_ap50']
    val_peak=min(val,key=lambda r:(-r['official_ap50'],r['epoch']))
    text=['# Object-Locus Panoptic full1201：核验、注册评测与交付','',
          f'固定选模为epoch{best}：dev8 true-novel official packed AP50={fmt(selection["value"])}。epoch8 endpoint AP50={fmt(row(8,"dev8")["official_ap50"])}。'+('存在endpoint相对所选节点的后期泛化退化。' if lower else 'dev8所选指标未显示endpoint低于最佳节点。'),
          '', '训练job58248 COMPLETED、ExitCode0:0，8epochs/8344 optimizer updates/66752 exposures，1191实际scene、8337独立两context窗口。原窗口各曝光8次，额外padding56次；六份checkpoint严格加载与参数有限性通过，八rank顺序/计数和代码/manifest SHA正确。',
          '',f'训练SHA `{verification["training_sha"]}`；科学基线 `{verification["science_sha"]}`；评测SHA及push收据见evaluation_git_provenance.json，逐评测来源见evaluation_sources.json。只新增核验、执行和汇总封装；模型、loss、provider、renderer、candidate阈值和官方导出/evaluator未改。本轮optimizer updates=0、backward=0，不续训/重训，不启动后续训练。',
          '',f'val32仅作曲线诊断：注册节点峰值epoch{val_peak["epoch"]}，packed AP50={fmt(val_peak["official_ap50"])}，endpoint相对峰值变化{fmt(row(8,"val32")["official_ap50"]-val_peak["official_ap50"])}。未据此改变dev8选模。',
          '', '## 口径与选模','', '所有比例、mIoU/PQ/mAP/AP50使用0–1原始单位；PSNR为dB，SSIM为0–1，LPIPS为距离（越低越好）。PSNR复用原窗口MSE定义；SSIM/LPIPS使用官方相同TorchMetrics实现（VGG、normalize=True），按帧均值。context→all/context，target-all→all/target，true novel→novel/target。独立candidate与竞争后的packed-panoptic指标分开。',
          '', 'dev8参与选模，且属于val32；val32不独立于选模。唯一选择指标为dev8 true-novel official packed AP50，完全相同取更早epoch；未用val32、完整验证集或图片改选。epoch8保留。',
          '', '离线understanding_step显式恢复checkpoint.completed_exposures。完整验证1860窗口/312scene，不应用训练thing数量或面积过滤，不丢空预测/无thing GT窗口，false positives保留。AP从所有原始预测重新聚合，不平均scene AP。',
          '', '分类confusion和matched准确率/no-object的matching来源是原context final_hungarian；各render scope中的重复值不是novel-only分类准确率。逐GT表分别保留raw best query、eligible best candidate及context-Hungarian matched query，附匹配来源、分母和精确窗口身份。',
          '', '0=实测零；UNDEFINED=定义不成立；MISSING=未获得；FAILED=执行失败，均不互相替换。旧结果未存SSIM/LPIPS等项保留MISSING，不启动旧模型评测。',
          '', '## 六节点未见scene曲线','', '|epoch|dev8 packed AP50|val32 packed AP50|dev8 candidate AP50|val32 candidate AP50|dev8 raw覆盖|val32 raw覆盖|val32 PSNR|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for d,v in zip(curves,val):text.append('|'+str(d['epoch'])+'|'+'|'.join(fmt(x) for x in (d['official_ap50'],v['official_ap50'],d['candidate_ap50'],v['candidate_ap50'],d['raw_mask_coverage'],v['raw_mask_coverage'],v['psnr']))+'|')
    text+=['','![固定监测曲线](curves.png)','','## 完整官方验证清单与排除dev8 scene汇总','', '|epoch|cohort|scope|windows|official mIoU|PQ|mAP|AP50|PSNR|SSIM|LPIPS|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in full:text.append('|'+str(r['epoch'])+'|'+r['cohort']+'|'+r['scope']+'|'+str(r['windows'])+'|'+'|'.join(fmt(r[k]) for k in ('official_miou','official_pq','official_map','official_ap50','psnr','ssim','lpips'))+'|')
    if best==8:text+=['','最佳与endpoint是同一checkpoint，完整验证仅运行一次。']
    text+=['','完整清单含选模scene，排除dev8的汇总移除其全部scene窗口；逐scene指标与有效数量见per_scene_metrics.json。','','## 固定训练与历史holdout曝光','']
    roster=read(rt.REPORT/'fixed_window_exposure.json')
    for split in read(rt.REPORT/'manifest.json')['monitor_splits']:
        rr=[r for r in roster if r['split']==split];text.append(f'- {split}: {len(rr)}窗口，实际训练窗口{sum(r["exact_training_window"] for r in rr)}，帧有曝光{sum(bool(r["exposed_frames"]) for r in rr)}。')
    text+=['','历史holdout8的6/8、holdout16的13/16窗口在本轮有帧曝光，必须标为“历史holdout，本轮存在帧曝光”；整组不能证明同场景未见窗口泛化。其余无帧交集的2/3窗口身份已保留，未重新采样或改变评测清单。dev8/val32全部scene未参与训练。',
           '', '## 与128场景结构、Fresh128及SIU3R比较','', '固定同窗口、同scope的完整逐指标比较见comparison_128.csv/json。128新结构epoch8/16/64只复用已有正式结果；Fresh128只复用epoch64正式结果。full对Panoptic128：结构、预训练来源、LR峰值、global batch8相同，训练scene覆盖、窗口曝光分配、总训练量及注册scheduler不同；full为66752曝光/8344更新，128为64512曝光/8064更新，epoch含义不同。full的200-update LR warmup是用户指定配置，原128使用原exposure时钟。Fresh128比较还涉及global batch、LR组、预训练理解及耦合等不同配方，不能归因于单个模块。',
           '', '|model|epoch|split|packed AP50|candidate AP50|raw覆盖|context-Hungarian分类准确率|PSNR|','|---|---:|---|---:|---:|---:|---:|---:|']
    for r in refs+[row(best,'dev8'),row(8,'dev8'),row(best,'val32'),row(8,'val32')]:
        if r['scope']=='novel' and r['split'] in ('dev8','val32'):text.append('|'+r.get('model','full1191')+'|'+str(r['epoch'])+'|'+r['split']+'|'+'|'.join(fmt(r[k]) for k in ('official_ap50','candidate_ap50','raw_mask_coverage','matched_class_accuracy','psnr'))+'|')
    text+=['','SIU3R：尚无同条件比较。当前使用GT camera poses；SIU3R README明确unposed。官方验证清单1860/312、2context、256输入及官方聚合协议可追溯，但相机条件不一致，且本模型encoder将256 crop上采样512，不宣称原生512细节。没有运行新SIU3R模型；未把论文unposed数值列入同条件胜负表。',
           '', '## 实例mask、分类、漏检与重建','']
    for e in sorted({best,8}):
        for split in ('expanded_train_probe32','same_scene_holdout8','dev8','val32'):
            r=row(e,split);text.append(f'epoch{e} {split} true-novel：raw覆盖{fmt(r["raw_mask_coverage"])}，candidate CW recall {fmt(r["candidate_cw_recall"])}，packed AP50 {fmt(r["official_ap50"])}；context-Hungarian分类准确率{fmt(r["matched_class_accuracy"])}（不能解读为novel-only matching）。')
    text+=['','完整验证漏检、eligible错分、重复候选、raw mask面积不足/超出GT及matched no-object统计见error_statistics.csv/json。raw recall<1及precision<1只是完整性/跨GT面积诊断，保持原mask阈值；跨物体最终需结合固定真实图片，未据此调参。注入非零只能证明通路活动，不代替任务成功。',
           '', '重建变化（val32 true-novel，epoch0→所选/endpoint）：'+fmt(row(best,'val32')['psnr']-row(0,'val32')['psnr'])+' / '+fmt(row(8,'val32')['psnr']-row(0,'val32')['psnr'])+' dB。SSIM/LPIPS和所有scope均见指标表；不只凭PSNR判断。',
           '', '## 固定真实图片与缺失项','', '沿用原128实验dev8/val32前两个固定窗口；六个checkpoint均保留，无按效果挑选。列为GT/pred RGB、GT/pred semantic、GT/pred panoptic、独立candidate overlay，以及类别/分数/资格明确的五个最高类分数query。旧128三节点相同固定面板一并保留，未重新评测旧模型。']
    for e in sorted({best,8}):
        for split in ('dev8','val32'):text.append(f'![epoch{e} {split} fixed pair0](qualitative/full_epoch{e:02}/{split}/pair0.png)')
    text+=['','缺失：旧128/Fresh结果未存的SSIM/LPIPS等标MISSING；Fresh128 holdout8缺独立local汇总时不平均AP补值；尚无GT-pose同条件SIU3R正式结果。注册本模型评测完整性见evaluation_status.json。',
           '', '交付ZIP均<28MiB，CRC、CSV/JSON、PNG及相对图片链接验证通过；不含checkpoint/数据集。六份训练checkpoint保留。本轮optimizer updates=0，科学配方未改，未启动后续训练。']
    (D/'report.md').write_text('\n'.join(text)+'\n')
    shutil.copyfile(D/'report.md',rt.REPORT/'report.md')

def main():
    D.mkdir(exist_ok=True);rows=read(rt.REPORT/'metrics_all_nodes.json');selection=read(rt.REPORT/'checkpoint_selection.json');full=[];per_scene=[];confusions=[]
    for e in EPOCHS:
        for split in read(rt.REPORT/'manifest.json')['monitor_splits']:
            r=read(EVAL/f'epoch{e:02}'/split/'result.json')
            for scope in SCOPES:confusions.append(dict(epoch=e,split=split,scope=scope,source='context final_hungarian; not novel-only',matrix=r['local'][scope]['classification_confusion']))
    for e in selection['full_evaluation_epochs']:
        root=EVAL/f'full_epoch{e:02}';full.extend(read(root/'metrics.json'));local={r['scene']:r for r in read(root/'per_scene_local.json')}
        for shard in range(8):
            for r in read(root/f'per_scene_official_shard{shard:02}.json'):per_scene.append(dict(epoch=e,**local[r['scene']],official=r['official']))
        for cohort,r in read(root/'aggregated_local.json').items():
            for scope in SCOPES:confusions.append(dict(epoch=e,split='full_validation',cohort=cohort,scope=scope,source='context final_hungarian; not novel-only',matrix=r['local'][scope]['classification_confusion']))
    write(D/'classification_confusion.json',confusions)
    write(rt.REPORT/'full_validation_metrics.json',full);csvout(rt.REPORT/'full_validation_metrics.csv',full);write(D/'per_scene_metrics.json',normalized(per_scene))
    comparisons,refs=compare(rows);errors=gt_tables()
    for name in ('metrics_all_nodes.csv','metrics_all_nodes.json','full_validation_metrics.csv','full_validation_metrics.json','checkpoint_selection.json','comparison_128.csv','comparison_128.json','training_verification.json','evaluation_git_provenance.json','evaluation_sources.json','evaluation_jobs.json','fixed_window_exposure.json','manifest.json','training_plan.json','asset_hashes.json','weights_provenance.json','weights_mapping.json','optimizer_groups.json','run_manifest.json','deferred_evaluation_plan.json','full_validation_manifest.json','single_smoke.json','eight_smoke.json','cleanup_128.json','official_aggregation_contract.json','pruned_official_exports.json'):
        shutil.copyfile(rt.REPORT/name,D/name)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,key in zip(axes.flat,('official_ap50','candidate_ap50','raw_mask_coverage','matched_class_accuracy','psnr','lpips')):
        for split in ('dev8','val32'):
            rr=[r for r in rows if r['split']==split and r['scope']=='novel'];ax.plot([r['epoch'] for r in rr],[r[key] for r in rr],marker='o',label=split)
        ax.set_title('true novel '+key);ax.set_xlabel('epoch');ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(D/'curves.png',dpi=150);plt.close(fig)
    for epoch in EPOCHS:
        for split in ('dev8','val32'):
            source=EVAL/f'epoch{epoch:02}'/split/f'qualitative/step_{epoch*1043:04d}'/split
            for image in source.glob('*.png'):
                dest=D/f'qualitative/full_epoch{epoch:02}'/split/image.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(image,dest)
    for epoch in (8,16,64):
        for split in ('dev8','val32'):
            for image in (OLD/f'qualitative/step_{epoch*126:04d}'/split).glob('*.png'):
                dest=D/f'qualitative/panoptic128_epoch{epoch:02}'/split/image.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(image,dest)
    status=dict(status='COMPLETE',training_job=58248,training_status='COMPLETED',training_exit_code='0:0',monitor_nodes=6,monitor_splits=6,scope_records=len(rows),
                full_evaluation_epochs=selection['full_evaluation_epochs'],full_records_per_checkpoint=1860,full_scenes=312,per_scene_records=len(per_scene),
                new_jobs=read(rt.REPORT/'evaluation_jobs.json'),missing_registered_evaluations=[],optimizer_updates=0,backward=0,scientific_recipe_changed=False,new_training_launched=False)
    write(rt.REPORT/'evaluation_status.json',status);write(D/'evaluation_status.json',status)
    build_report(rows,full,refs,errors)
    code=Path(__file__).resolve().parents[1];(D/'code').mkdir(exist_ok=True)
    diff=subprocess.check_output(['git','diff','b2624d57ea9ad5a73fd6a8375bafbe4e4c263c9c','HEAD'],cwd=code)
    (D/'code/source.diff').write_bytes(diff)
    for p in sorted(code.glob('scripts/*full1201*')):
        if p.is_file():shutil.copyfile(p,D/'code'/p.name)
    for p in sorted(code.glob('docs/*full1201*')):shutil.copyfile(p,D/'code'/p.name)
    logs=D/'logs';logs.mkdir(exist_ok=True)
    for name in ('cpu_contracts.log','training_verification.log'):
        shutil.copyfile(rt.REPORT/name,logs/name)
    for p in sorted((rt.REPORT/'slurm').glob('*')):
        if p.is_file():
            text=p.read_text(errors='replace').replace('\x00','');lines=text.splitlines();(logs/p.name).write_text('\n'.join(lines[:12]+(['... middle omitted; original path '+str(p)] if len(lines)>72 else [])+lines[-60:])+'\n')
    # Validate all deliverables before archives. Cross-package image links resolve
    # after extracting all archives into the same directory.
    checks=[]
    for p in sorted(D.rglob('*')):
        if not p.is_file():continue
        if p.suffix=='.json':read(p)
        elif p.suffix=='.csv':
            with p.open() as f:list(csv.DictReader(f))
        elif p.suffix=='.png':
            with Image.open(p) as im:im.load()
        checks.append(f'{rt.sha(p)}  {p.relative_to(D)}')
    for link in re.findall(r'!\[[^]]*\]\(([^)]+)\)',(D/'report.md').read_text()):assert (D/link).is_file(),link
    (D/'SHA256SUMS').write_text('\n'.join(checks)+'\n');shutil.copyfile(D/'SHA256SUMS',rt.REPORT/'SHA256SUMS')
    limit=28*1024**2;archives=[]
    groups=[('results',[p for p in sorted(D.rglob('*')) if p.is_file() and p.suffix!='.png']),('images',[p for p in sorted(D.rglob('*.png'))])]
    for label,files in groups:
        part=1;current=[]
        def pack(paths,index):
            path=rt.REPORT/f'{label}_part{index:02}.zip'
            with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for p in paths:z.write(p,str(p.relative_to(D)))
            return path
        for p in files:
            trial=pack(current+[p],part)
            if trial.stat().st_size>=limit:
                assert current,'single file exceeds package limit: '+str(p)
                finished=pack(current,part)
                with zipfile.ZipFile(finished) as z:assert z.testzip() is None
                archives.append(dict(path=str(finished),bytes=finished.stat().st_size,sha256=rt.sha(finished)));part+=1;current=[p]
            else:current.append(p)
        if current:
            finished=pack(current,part);assert finished.stat().st_size<limit
            with zipfile.ZipFile(finished) as z:assert z.testzip() is None
            archives.append(dict(path=str(finished),bytes=finished.stat().st_size,sha256=rt.sha(finished)))
    write(rt.REPORT/'result_archives.json',archives);print(json.dumps(archives,indent=2),flush=True)

if __name__=='__main__':main()
