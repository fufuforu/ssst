"""Unchanged evalfix eligibility/IoU protocol applied to the panoptic model."""
from scripts.object_locus_panoptic_v1_runtime import *
from scripts.eval_object_locus_v3_set import _run,_candidate_stats,write_panel


def local_ap_metric():
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    return MeanAveragePrecision(iou_type='segm',sync_on_compute=False)


def evaluate_windows(model,opt,windows,step,split,reports,device,batch_builder,*,official=False,panels=False):
    from scripts.object_locus_v3_set_runtime import capture_rng,restore_rng,write_json
    from scripts.export_object_locus_v3_set_official import export_windows
    from scripts.eval_object_locus_v1 import _official_run
    was=model.training;rng=capture_rng();model.eval();rows=[]; per_gt=[];query_rows=[]
    try:
      from torchmetrics.detection.mean_ap import MeanAveragePrecision
      ap_metrics={s:local_ap_metric() for s in ('context','target_all','novel')}
    except Exception as exc:
      ap_metrics={};ap_error=f'{type(exc).__name__}: {exc}'
    try:
      with torch.no_grad():
       for wi,win in enumerate(windows):
        batch,out=_run(model,opt,win,batch_builder,device)
        frame_ids=[int(x) for x in batch['frame_ids'][0].cpu().tolist()]
        ctx=[0,1]; target=list(range(len(frame_ids))); novel=[i for i,x in enumerate(frame_ids) if x in set(map(int,win['novel']))]
        scopes={}
        for name,idx in [('context',ctx),('target_all',target),('novel',novel)]:
         row=_candidate_stats(out,batch,idx)
         if ap_metrics:
          payload=row.pop('_map_payload')
          for branch in ('pred','target'):
           pmask=payload[branch]['masks']
           payload[branch]['masks']=pmask.reshape(pmask.shape[0],-1,pmask.shape[-1]) if pmask.shape[0] else torch.zeros((0,len(idx)*256,256),dtype=torch.bool,device=pmask.device)
          ap_metrics[name].update([payload['pred']],[payload['target']])
         views=len(idx);mse=(out['render']['images_pred'][0,idx]-batch['images_all'][0,idx]).square().mean().clamp_min(1e-12)
         row['psnr']=float((-10*torch.log10(mse)).cpu());row['scope']=name;scopes[name]=row
         per_gt.extend({'split':split,'step':step,'scope':name,'scene':win['scene'],**x} for x in row['per_gt'])
         query_rows.extend({'split':split,'step':step,'scope':name,'scene':win['scene'],**x} for x in row['query_rows'])
        rows.append({'scene':win['scene'],'context':win['context'],'novel':win['novel'],'scopes':scopes})
        if panels and wi<2: write_panel(batch,out,win,Path(reports)/f'qualitative/step_{step:04d}/{split}/pair{wi}.png',f'{step} {split}')
    finally:restore_rng(rng);model.train(was)
    aggregated={}
    for scope in ('context','target_all','novel'):
      rr=[x['scopes'][scope] for x in rows]
      conf=np.sum([np.asarray(x['semantic_confusion']) for x in rr],axis=0)
      panconf=np.sum([np.asarray(x['panoptic_semantic_confusion']) for x in rr],axis=0)
      def _ious(matrix, classes):
        values=[]
        for c in classes:
          tp=matrix[c,c];den=matrix[c,:].sum()+matrix[:,c].sum()-tp
          if den:values.append(float(tp/den))
        return values
      all_iou=_ious(conf,range(20));thing_iou=_ious(conf,range(2,20));stuff_iou=_ious(conf,(0,1))
      pan_all_iou=_ious(panconf,range(20))
      agg={'windows':len(rr),'gt_count':sum(x['gt_count'] for x in rr),'semantic_confusion':conf.tolist(),
        'panoptic_semantic_confusion':panconf.tolist(),
        'semantic_miou':float(np.mean(all_iou)) if all_iou else 0.,'mIoU_thing':float(np.mean(thing_iou)) if thing_iou else 0.,
        'mIoU_stuff':float(np.mean(stuff_iou)) if stuff_iou else 0.,'psnr':float(np.mean([x['psnr'] for x in rr])),
        'candidate_ca':{k:sum(x['candidate_ca'][k] for x in rr) for k in ('tp','fp','fn')},
        'candidate_cw':{k:sum(x['candidate_cw'][k] for x in rr) for k in ('tp','fp','fn')},
        'raw_best_iou_ge_0_5_fraction':sum(sum(x['raw_best_iou_ge_0_5_fraction']*len(x['raw_best_ious']) for x in rr) for x in []) if False else sum(sum(v>=.5 for v in x['raw_best_ious']) for x in rr)/max(1,sum(len(x['raw_best_ious']) for x in rr)),
        'matched_19_class_accuracy':sum(x['matched_19_class_accuracy']*x['matched_gt_count'] for x in rr)/max(1,sum(x['matched_gt_count'] for x in rr)),
        'panoptic_pq':float(np.mean([x['panoptic_pq'] for x in rr])),
        'panoptic_semantic_miou':float(np.mean(pan_all_iou)) if pan_all_iou else 0.,
        'classification_confusion':np.sum([np.asarray(x['classification_confusion']) for x in rr],axis=0).tolist(),
        'matched_gt_count':sum(x['matched_gt_count'] for x in rr),'candidate_count':sum(x['candidate_count'] for x in rr),
        'panoptic_ca':{k:sum(x['panoptic_ca'][k] for x in rr) for k in ('tp','fp','fn')},
        'panoptic_cw':{k:sum(x['panoptic_cw'][k] for x in rr) for k in ('tp','fp','fn')},
        'per_class_iou':{str(c):float(conf[c,c]/max(1,conf[c,:].sum()+conf[:,c].sum()-conf[c,c])) for c in range(20)}}
      for k in ('candidate_ca','candidate_cw','panoptic_ca','panoptic_cw'):
       d=agg[k];d['precision']=d['tp']/max(1,d['tp']+d['fp']);d['recall']=d['tp']/max(1,d['tp']+d['fn'])
      try:
       if ap_metrics:
        m=ap_metrics[scope].compute();agg['candidate_ap']={'map':float(m['map']),'map_50':float(m['map_50'])}
       else:agg['candidate_ap']={'error':ap_error}
      except Exception as exc:agg['candidate_ap']={'error':f'{type(exc).__name__}: {exc}'}
      aggregated[scope]=agg
    result={'step':step,'split':split,'local':aggregated,'windows':[{k:v for k,v in x.items() if k!='scopes'}|{'scopes':{s:{k:v for k,v in x['scopes'][s].items() if k not in ('semantic_confusion','panoptic_semantic_confusion','panoptic_semantic','panoptic_instance','raw_masks','gt_semantic','gt_instance','predictions','per_gt')} for s in x['scopes']}} for x in rows]}
    if official:
      root=Path(reports)/f'official/step_{step:04d}/{split}'
      allx=export_windows(model,opt,windows,root/'all',device=device,batch_builder=batch_builder,target_frames='all')
      nov=export_windows(model,opt,windows,root/'novel',device=device,batch_builder=batch_builder,target_frames='novel')
      all_json=_official_run(root/'all',root/'official_all.json');nov_json=_official_run(root/'novel',root/'official_novel.json')
      result['official']={'all':all_json.get('result'),'novel':nov_json.get('result')}
    write_json(Path(reports)/f'eval_{split}_step{step:04d}.json',result)
    return result,per_gt,query_rows


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
    # Fresh128 registered 16-window holdout contains the required original eight.
    # Reaggregate the existing raw predictions; never load an old checkpoint.
    if old and 'same_scene_holdout8' not in old and 'same_scene_holdout16' in old:
        manifest=json.loads((REPORTS/'data_manifest.json').read_text())
        subset=manifest['same_scene_holdout8']
        source=old['same_scene_holdout16']
        selected=[r for r in source['windows'] if any(r['scene']==w['scene'] and r['context']==w['context'] and r['novel']==w['novel'] for w in subset)]
        if len(selected)==8:
            from scripts.eval_object_locus_v1 import _official_run
            base=REPORTS/'comparison/fresh128_same_scene_holdout8'
            official={}
            complete=True
            for arm in ('all','novel'):
                root=base/arm;root.mkdir(parents=True,exist_ok=True)
                for w in subset:
                    name=w['scene']+'_context'+'_'.join(map(str,w['context']))
                    target=FRESH/'official/step_64512/same_scene_holdout16'/arm/name
                    if not target.is_dir():complete=False;break
                    link=root/name
                    if not link.exists():link.symlink_to(target,target_is_directory=True)
                if complete:official[arm]=_official_run(root,base/f'official_{arm}.json')['result']
            if complete:
                old['same_scene_holdout8']=dict(official=official,local={scope:dict(psnr=float(np.mean([r['scopes'][scope]['psnr'] for r in selected]))) for scope in ('context','target_all','novel')})
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
