"""Unchanged evalfix eligibility/IoU protocol applied to the panoptic model."""
from scripts.object_locus_panoptic_v1_runtime import *
from scripts.eval_object_locus_v3_set import evaluate_windows


def evaluate_epoch(model,opt,manifest,splits,epoch,device):
    import csv,time
    start=time.perf_counter();results={};gt=[];queries=[]
    for split in splits:
        result,g,q=evaluate_windows(model,opt,manifest[split],epoch*126,split,REPORTS,device,build_batch,
                                   official=epoch in OFFICIAL,panels=True)
        results[split]=result;gt.extend(g);queries.extend(q)
    for label,rows in [('per_gt',gt),('queries',queries)]:
        path=REPORTS/f'{label}_epoch{epoch:02}.csv'
        keys=sorted({k for row in rows for k in row})
        with path.open('w') as f:
            writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)
    write_json(REPORTS/f'eval_epoch{epoch:02}.json',dict(epoch=epoch,update=epoch*126,exposure=epoch*1008,
        posed_setting=True,results=results,seconds=time.perf_counter()-start))
    return results


def parameter_drift(model,epoch):
    from tokengs.models.object_locus_panoptic_v1_pretrained import MAST,PANOPTIC
    mapping=json.loads((REPORTS/'weights_mapping.json').read_text())['mapping']
    target=model.state_dict();rows={}
    for label,path,container in [('mast3r',MAST,'model'),('panoptic',PANOPTIC,'state_dict')]:
        source=torch.load(path,map_location='cpu',weights_only=False,mmap=True)[container]
        numerator=denominator=0.0
        for row in mapping:
            key=row.get('target')
            if row['status']!='LOADED' or not key or not key.startswith('understanding.'):continue
            if row['source'] not in source:continue
            x=target[key].detach().cpu().float();y=source[row['source']].float()
            numerator+=float((x-y).square().sum());denominator+=float(y.square().sum())
        rows[label]=dict(l2_drift=numerator**.5,relative_l2=(numerator/max(denominator,1e-30))**.5)
        del source
    write_json(REPORTS/f'pretrained_drift_epoch{epoch:02}.json',rows)


def endpoint_report():
    import csv,hashlib,subprocess,zipfile,shutil
    new=json.loads((REPORTS/'eval_epoch64.json').read_text())
    old_path=FRESH/'eval_node_epoch_64.json'
    old=json.loads(old_path.read_text())['splits'] if old_path.is_file() else None
    rows=[];comparison=[]
    for split,result in new['results'].items():
        old_name='original_train_all56' if split=='train_all56' else split
        reference=old.get(old_name) if old else None
        for scope in ('context','target_all','novel'):
            local=result['local'][scope]
            arm,view=('novel','target') if scope=='novel' else ('all','target' if scope=='target_all' else 'context')
            official=result.get('official',{}).get(arm,{})
            omap=official.get(view+'_map',{})
            row=dict(split=split,scope=scope,semantic_miou=local['semantic_miou'],thing_miou=local['mIoU_thing'],stuff_miou=local['mIoU_stuff'],
                local_pq=local['panoptic_pq'],local_panoptic_miou=local['panoptic_semantic_miou'],candidate_map=local.get('candidate_ap',{}).get('map'),candidate_ap50=local.get('candidate_ap',{}).get('map_50'),
                official_miou=official.get(view+'_miou'),official_pq=official.get(view+'_pq'),official_map=omap.get('map'),official_ap50=omap.get('map_50'),psnr=local['psnr'],
                classification_accuracy=local['matched_19_class_accuracy'],candidate_ca=local['candidate_ca'],candidate_cw=local['candidate_cw'])
            rows.append(row)
            oldofficial=reference.get('official',{}).get(arm,{}) if reference else {}
            comparison.append(dict(split=split,scope=scope,new_ap50=row['official_ap50'],fresh128_ap50=oldofficial.get(view+'_map',{}).get('map_50'),
                new_psnr=row['psnr'],fresh128_psnr=reference['local'][scope]['psnr'] if reference else None,
                status='AVAILABLE' if reference else 'PENDING' if old is None else 'UNAVAILABLE: split not evaluated in Fresh128; no extra checkpoint evaluation'))
    ci_path=REPORTS/'paired_bootstrap.json'
    subprocess.run(['/space/mawb/SIU3R/.venv_gpu_v4/bin/python','-m','scripts.export_object_locus_panoptic_v1_official',
        '--bootstrap',str(REPORTS/'official/step_8064/val32/novel'),str(FRESH/'official/step_64512/val32/novel'),'--output',str(ci_path)],cwd=REPO,check=True)
    ci=json.loads(ci_path.read_text())
    def metric(split,scope):return next(r for r in rows if r['split']==split and r['scope']==scope)
    def comp(split,scope):return next(r for r in comparison if r['split']==split and r['scope']==scope)
    vnov=comp('val32','novel');vctx=comp('val32','context');hold=comp('same_scene_holdout8','context')
    origin=json.loads((REPORTS/'eval_epoch00.json').read_text())['results']['val32']['local']
    criteria={
        'val32_true_novel_AP50_gain_at_least_0_01': None if vnov['fresh128_ap50'] is None else vnov['new_ap50']-vnov['fresh128_ap50']>=.01,
        'val32_context_AP50_not_lower':None if vctx['fresh128_ap50'] is None else vctx['new_ap50']>=vctx['fresh128_ap50'],
        'same_scene_holdout8_context_AP50_drop_at_most_0_01':None if hold['fresh128_ap50'] is None else hold['new_ap50']-hold['fresh128_ap50']>=-.01,
        'val32_context_PSNR_drop_at_most_0_5dB':metric('val32','context')['psnr']-origin['context']['psnr']>=-.5,
        'val32_true_novel_PSNR_drop_at_most_0_5dB':metric('val32','novel')['psnr']-origin['novel']['psnr']>=-.5}
    engineer=all(v is True for v in criteria.values()) if all(v is not None for v in criteria.values()) else None
    delta=None if vnov['fresh128_ap50'] is None else vnov['new_ap50']-vnov['fresh128_ap50']
    ci_lower=ci.get('ap50_difference_ci95',[None])[0]
    conclusion='Comparison PENDING/UNAVAILABLE; no unregistered checkpoint reevaluation.' if delta is None else '固定工程判据未全部满足；本轮停止，不启动新实验。'
    if delta is not None and delta>0 and (ci_lower is None or ci_lower<=0):conclusion='点估计提升、证据仍不充分。'
    if engineer and ci_lower is not None and ci_lower>0:conclusion='本联合配方在该固定验证池有提升证据，不能归因到某一个模块，也不等于完整SIU3R条件已对齐。'
    write_json(REPORTS/'task_metrics.json',dict(rows=rows,comparison=comparison,criteria=criteria,engineering_criteria_met=engineer,bootstrap=ci,conclusion=conclusion,
        posed_setting=True,recipe_comparison='Batch, LR groups, pretraining and coupling differ; same 64512 exposures versus Fresh128 64512 optimizer steps and new 8064 optimizer steps.'))
    with (REPORTS/'task_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Object-Locus Panoptic V1 endpoint','',
        'posed setting：使用GT camera poses；256 crop上采样到512，并未获得原生512细节。',
        '本轮为配方比较：batch、LR组、预训练和耦合均变化。曝光64512相同，优化更新次数新配方8064、Fresh128 64512；不能归因于单个结构模块。','',
        '| Split | Scope | semantic mIoU | local PQ | candidate AP50 | official packed AP50 | PSNR |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"|{r['split']}|{r['scope']}|{r['semantic_miou']:.5f}|{r['local_pq']:.5f}|{r['candidate_ap50']}|{r['official_ap50']}|{r['psnr']:.3f}|")
    for split,scope,label in [('expanded_train_probe32','context','训练池完整实例任务'),('same_scene_holdout8','context','同场景新窗口'),('val32','novel','未见场景真实novel')]:
        r=metric(split,scope)
        text.extend(['',f"{label}：candidate AP50={r['candidate_ap50']}，official packed AP50={r['official_ap50']}，CW recall={r['candidate_cw']['recall']:.5f}，matched classification accuracy={r['classification_accuracy']:.5f}。"])
    text.extend(['',f"重建保持：val32 context PSNR变化{metric('val32','context')['psnr']-origin['context']['psnr']:.3f}dB；true novel变化{metric('val32','novel')['psnr']-origin['novel']['psnr']:.3f}dB。"])
    text+=['','## 固定结论','',conclusion,'',json.dumps(criteria,ensure_ascii=False,indent=2),'',
        '训练池、同场景新窗口、未见场景的mask/classification结论需分别依据上表和per_gt/query表，训练成功不能替代泛化结论。AP=-1表示官方指标未定义，不替换为0。',
        '本轮只有已授权计数勘误，未修改科学配方；不承诺8倍加速。']
    (REPORTS/'analysis_report.md').write_text('\n'.join(text)+'\n')
    shutil.copyfile(REPO/'docs/object_locus_panoptic_v1_codex_spec.md',REPORTS/'spec.md')
    # Keep real qualitative cases, splitting image archives without dropping cases.
    images=sorted(p for p in REPORTS.rglob('*.png') if 'qualitative' in p.parts)
    base_files=sorted(p for p in REPORTS.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.md','.log','.out','.err') and 'official' not in p.parts and 'smoke_eval' not in p.parts)
    archives=[];limit=28*1024**2
    def pack(path,files):
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in files:z.write(p,str(p.relative_to(REPORTS)))
        if path.stat().st_size>=limit:raise RuntimeError(f'ZIP exceeds registered size: {path}')
        archives.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pack(REPORTS/'results.zip',base_files+images[:1])
    current=[];size=0;part=1
    for p in images[1:]:
        if current and size+p.stat().st_size>=limit-1024**2:
            pack(REPORTS/f'images_part{part:02}.zip',current);part+=1;current=[];size=0
        current.append(p);size+=p.stat().st_size
    if current:pack(REPORTS/f'images_part{part:02}.zip',current)
    write_json(REPORTS/'result_archives.json',archives)
    return rows,comparison,archives
