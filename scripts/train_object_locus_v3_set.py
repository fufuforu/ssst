"""Fresh 8-scene / 56-window, 64-epoch Object-Locus V3-Set experiment."""
from __future__ import annotations
import argparse,csv,json,math,os,subprocess,zipfile
from pathlib import Path
import sys
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.object_locus_v3_set_runtime import (REPO,PRETRAINED,PRETRAINED_SHA,PRETRAINED_STEP,
    REPORTS_DEFAULT,RUN_ROOT_DEFAULT,build_model,build_optimizer,build_batch,train_one_step,
    trainability_counts,capture_rng,restore_rng,seed_everything,sha256_file,write_json,jsonable,
    GC_ALPHA,OBJECT_PEAK_LR,RECON_PEAK_LR)
from scripts.eval_object_locus_v3_set import evaluate_windows

SOURCE_MANIFEST=Path('/space/mawb/ssst/group_plus/object_locus_v2_1/data_manifest.json')
SOURCE_MANIFEST_SHA='ebed1a133d64ed38ef7afce17aaaf27bbe65c6b0edea950c229b5d1e4ff77bf0'
FIXED_SCENES=('scene0000_00','scene0003_02','scene0009_00','scene0013_01','scene0018_00','scene0024_02','scene0031_00','scene0035_00')
EVAL_EPOCHS=(0,8,16,32,64)

def _key(w):return (w['scene'],tuple(w['context']),tuple(w['novel']))
def _frames(w):return set(map(int,w['context']+w['novel']))
def _sort(w):return (tuple(w['context']),tuple(w['novel']),int(w.get('index',0)))

def build_manifest():
    if sha256_file(SOURCE_MANIFEST)!=SOURCE_MANIFEST_SHA:raise RuntimeError('locked V2.1 data manifest SHA mismatch')
    src=json.loads(SOURCE_MANIFEST.read_text());train=[w for w in src['small_train_windows'] if w['scene'] in FIXED_SCENES]
    hold=[w for w in src['same_scene_holdout16'] if w['scene'] in FIXED_SCENES]
    if len(train)!=56 or len(hold)!=8 or {w['scene'] for w in train}!=set(FIXED_SCENES):raise RuntimeError(f'fixed split mismatch train={len(train)} hold={len(hold)}')
    counts={s:sum(x['scene']==s for x in train) for s in FIXED_SCENES}
    if any(x!=7 for x in counts.values()):raise RuntimeError(f'expected 7 windows per scene: {counts}')
    hb={w['scene']:w for w in hold};leaks=[(_key(w),_key(hb[w['scene']])) for w in train if _frames(w)&_frames(hb[w['scene']])]
    if leaks:raise RuntimeError(f'train/holdout frame overlap {leaks[:3]}')
    pool_scenes={w['scene'] for w in src['expanded_train_windows']}
    dev_scenes={w['scene'] for w in src['dev8']}
    if dev_scenes & pool_scenes:raise RuntimeError(f'dev8 is not cross-scene: {sorted(dev_scenes & pool_scenes)}')
    probe=[sorted((w for w in train if w['scene']==s),key=_sort)[0] for s in FIXED_SCENES]
    return {'source_manifest':str(SOURCE_MANIFEST),'source_sha256':SOURCE_MANIFEST_SHA,'small_stage_scenes':list(FIXED_SCENES),
      'train_all56':train,'train_probe8':probe,'same_scene_holdout8':hold,'dev8':src['dev8'],
      'val8':src['fixed_val8'],'val32':src['fixed_val32'],'legacy_train16':src['legacy_train16'],
      'window_counts':{'train_all56':56,'train_probe8':8,'holdout8':8,'dev8':8,'val32':len(src['fixed_val32'])},
      'scene_window_counts':counts,'frame_disjoint_train_holdout':True,'training_exposures_per_window':64,'epochs':64,'updates':3584}

def build_plan(manifest):
    entries=[]
    for epoch in range(64):
        for pos,wi in enumerate(np.random.default_rng(42+epoch).permutation(56)):
            w=manifest['train_all56'][int(wi)];entries.append({'step':len(entries)+1,'epoch':epoch,'position':pos,'window_index':int(wi),'identity':{'scene':w['scene'],'context':w['context'],'novel':w['novel']},'scene':w['scene'],'context':w['context'],'novel':w['novel']})
    return {'seed_base':42,'epochs':64,'windows_per_epoch':56,'updates':len(entries),'entries':entries}

def lr_mult(t):
    if t<=0:return 0.
    if t<=200:return t/200.
    u=(t-200)/(3584-200);return .1+.9*(1+math.cos(math.pi*u))/2

def _gpu_assert():
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('requires exactly one CUDA GPU')
    if torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 3090':raise RuntimeError('formal V3-Set is fixed to RTX3090')
    if torch.cuda.get_device_properties(0).total_memory<23*1024**3:raise RuntimeError('GPU memory is not 24GB class')
    if not os.uname().nodename.startswith('3dimage-13'):raise RuntimeError('formal V3-Set is fixed to 3dimage-13')
    return {'gpu':torch.cuda.get_device_name(0),'node':os.uname().nodename,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}

def _save(path,model,opt,optimizer,epoch,step,plan_sha,manifest_sha,exposure):
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':capture_rng(),'stage':'V3_SET','epoch':epoch,'stage_step':step,'global_optimizer_step':step,'completed_updates':step,'next_epoch':epoch,'next_position':0,'plan_position':step,'architecture_name':model.architecture_name,'git_sha':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'source_hashes':{'pretrained':PRETRAINED_SHA,'source_manifest':SOURCE_MANIFEST_SHA},'plan_sha256':plan_sha,'manifest_sha256':manifest_sha,'optimizer_config':{'betas':[.9,.95],'eps':1e-8,'gc_alpha':GC_ALPHA},'exposure_stats':exposure,'config':jsonable(vars(opt))},path)

def _eval(model,opt,splits,reports,step,names,official_names,panels):
    result={'step':step,'epoch':step//56,'global_optimizer_step':step,'splits':{}};pergt=[];queryrows=[]
    for name in names:
        x,g,q=evaluate_windows(model,opt,splits[name],step,name,reports,'cuda',build_batch,official=name in official_names,panels=panels);result['splits'][name]=x;pergt+=g;queryrows+=q
    write_json(reports/f'curves_step_{step:04d}.json',result);return result,pergt,queryrows

def _write_outputs(reports,nodes,pergt,queryrows,summary):
    rows=[]
    for node in nodes:
      for split,x in node['splits'].items():
       for scope,m in x['local'].items():
        row={'step':node['step'],'epoch':node['epoch'],'split':split,'scope':scope}
        for k,v in m.items():
         if isinstance(v,(str,int,float)):row[k]=v
         elif k in ('candidate_ca','candidate_cw','panoptic_ca','panoptic_cw','candidate_ap'):
          for a,b in v.items():row[f'{k}_{a}']=b
        official=x.get('official',{})
        for arm in ('all','novel'):
         metrics=official.get(arm,{}) if isinstance(official,dict) else {}
         for mk,mv in metrics.items():
          if isinstance(mv,dict):
           for sk,sv in mv.items():
            if isinstance(sv,(int,float)):row[f'official_{arm}_{mk}_{sk}']=sv
        rows.append(row)
    keys=sorted({k for r in rows for k in r})
    with (reports/'task_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    write_json(reports/'task_metrics.json',rows)
    gkeys=sorted({k for r in pergt for k in r})
    with (reports/'per_gt_candidate_and_panoptic.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=gkeys);w.writeheader();w.writerows(pergt)
    qkeys=sorted({k for r in queryrows for k in r})
    with (reports/'candidate_panoptic_queries.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=qkeys);w.writeheader();w.writerows(queryrows)
    confusion={}
    for n in nodes:
      for split,s in n['splits'].items():
       for scope,m in s['local'].items():confusion[f"{n['step']}:{split}:{scope}"]=m['classification_confusion']
    write_json(reports/'classification_confusion.json',confusion)
    s0=next(x for x in nodes if x['step']==0)['splits']['train_all56']
    s64=next(x for x in nodes if x['step']==3584)['splits']['train_all56']
    e0c=s0['local']['context'];e64c=s64['local']['context']
    e0n=s0['local']['novel'];e64n=s64['local']['novel']
    off=s64.get('official',{}).get('all',{});official_ap50=off.get('context_map',{}).get('map_50') if isinstance(off,dict) else None
    checks={
      'raw_best_mask_iou50_fraction_ge_0_80':e64c['raw_best_iou_ge_0_5_fraction']>=.80,
      'matched_19_class_accuracy_ge_0_85':e64c['matched_19_class_accuracy']>=.85,
      'candidate_ca_recall_ge_0_70':e64c['candidate_ca']['recall']>=.70,
      'candidate_cw_recall_ge_0_60':e64c['candidate_cw']['recall']>=.60,
      'candidate_cw_precision_ge_0_60':e64c['candidate_cw']['precision']>=.60,
      'candidate_ap50_ge_0_60':(e64c.get('candidate_ap',{}).get('map_50') or 0.)>=.60,
      'panoptic_cw_recall_ge_0_50':e64c['panoptic_cw']['recall']>=.50,
      'official_ap50_ge_0_40':(official_ap50 or 0.)>=.40,
      'context_psnr_drop_le_0_5db':e0c['psnr']-e64c['psnr']<=.5,
      'true_novel_psnr_drop_le_0_5db':e0n['psnr']-e64n['psnr']<=.5,
    }
    gate={'passed':all(checks.values()),'checks':checks,'measured':{'train_all56_context':e64c,'train_all56_true_novel_psnr_E0':e0n['psnr'],'train_all56_true_novel_psnr_E64':e64n['psnr'],'official_context_ap50':official_ap50}}
    write_json(reports/'train_set_acceptance.json',gate)
    write_json(reports/'training_summary.json',summary)
    lines=['# Object-Locus V3-Set：小规模完整实例任务验证','','完成3584/3584 optimizer updates。固定8场景训练集验证，不代表跨场景泛化或完整SIU3R benchmark。',f"\ntrain_all56 context endpoint gate: **{'PASS' if gate['passed'] else 'FAIL'}**。",'','| step | split | scope | thing mIoU | candidate AP50 | candidate CW P/R | panoptic CW TP/FP/FN | PSNR | official AP50 |','|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for node in nodes:
      for split,x in node['splits'].items():
       for scope,m in x['local'].items():
        ap=m.get('candidate_ap',{}).get('map_50');cw=m['candidate_cw'];pan=m['panoptic_cw'];off=x.get('official',{}).get('all',{});o=off.get('context_map',{}).get('map_50') if isinstance(off,dict) else None
        lines.append(f"| {node['step']} | {split} | {scope} | {m['mIoU_thing']:.4f} | {ap if ap is not None else 'MISSING'} | {cw['precision']:.3f}/{cw['recall']:.3f} | {pan['tp']}/{pan['fp']}/{pan['fn']} | {m['psnr']:.3f} | {o if o is not None else 'MISSING'} |")
    (reports/'analysis_report.md').write_text('\n'.join(lines)+'\n')

def _write_package_metadata(reports):
    base='5f1bdf74cc988ad9063a2621482de32f581cce7c'
    diff=subprocess.run(['git','-C',str(REPO),'diff',f'{base}..HEAD'],capture_output=True,text=True,check=True).stdout
    (reports/'source.patch').write_text(diff)
    status=subprocess.run(['git','-C',str(REPO),'status','--short'],capture_output=True,text=True,check=True).stdout
    (reports/'git_status.txt').write_text(status or 'clean\n')
    readme='''# Object-Locus V3-Set result bundle\n\n小规模训练结果，不是跨场景泛化或完整 SIU3R benchmark。\n\n- [中文分析报告](analysis_report.md)\n- [任务指标 CSV](task_metrics.csv)\n- [任务指标 JSON](task_metrics.json)\n- [逐 GT 候选与 panoptic 记录](per_gt_candidate_and_panoptic.csv)\n- [分类混淆矩阵](classification_confusion.json)\n- [slot 到 panoptic filtering 记录](candidate_panoptic_queries.csv)\n- [训练曲线](training_metrics.jsonl)\n- [数据清单](data_manifest.json)\n- [训练计划](training_plan.json)\n- [运行配置](run_manifest.json)\n- [源代码差异](source.patch)\n- [Git 状态](git_status.txt)\n- 固定图片位于 `qualitative/`。\n\n压缩包不含 checkpoint、数据集或完整 official prediction exports。\n'''
    (reports/'README.md').write_text(readme)

def _package(reports):
    zpath=reports/'result_bundle.zip';files=[p for p in reports.rglob('*') if p.is_file() and p!=zpath and 'official' not in p.parts and p.suffix not in ('.pt','.pth','.ckpt')]
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(reports))
    with zipfile.ZipFile(zpath) as z:
        if z.testzip():raise RuntimeError('result bundle ZIP integrity error')
    if zpath.stat().st_size>=28*1024*1024:
        zpath.unlink()
        files=[p for p in reports.rglob('*') if p.is_file() and p!=zpath and 'official' not in p.parts and 'qualitative' not in p.parts and p.suffix not in ('.pt','.pth','.ckpt')]
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in files:z.write(p,p.relative_to(reports))
        qzip=reports/'result_bundle_qualitative.zip'
        with zipfile.ZipFile(qzip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in (reports/'qualitative').rglob('*'):
                if p.is_file():z.write(p,p.relative_to(reports))
        if zpath.stat().st_size>=28*1024*1024 or qzip.stat().st_size>=28*1024*1024:raise RuntimeError('split result package still exceeds 28 MiB')

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train'],default='train');p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.device!='cuda':raise RuntimeError('formal V3-Set requires cuda')
    gpu=_gpu_assert();reports=REPORTS_DEFAULT;run=RUN_ROOT_DEFAULT
    if reports.exists() and {x.name for x in reports.iterdir()}-{'smoke_validation','slurm'}:raise RuntimeError(f'reports directory has unregistered contents: {reports}')
    if run.exists() and any(run.iterdir()):raise RuntimeError(f'run directory is not empty: {run}')
    reports.mkdir(parents=True,exist_ok=True);run.mkdir(parents=True,exist_ok=True)
    manifest=build_manifest();plan=build_plan(manifest);write_json(reports/'data_manifest.json',manifest);write_json(reports/'training_plan.json',plan)
    msha=sha256_file(reports/'data_manifest.json');psha=sha256_file(reports/'training_plan.json')
    seed_everything(42);model,opt,transfer=build_model('cuda');optimizer,opt_audit=build_optimizer(model)
    counts=trainability_counts(model);commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    run_manifest={'architecture':'LOCUSGS_OBJECT_LOCUS_V3_SET','recipe':'OBJECT_LOCUS_V3_SET_8SCENE_56WINDOW_64EPOCH','git_sha':commit,'pretrained_sha256':PRETRAINED_SHA,'pretrained_step':PRETRAINED_STEP,'transfer':transfer,'source_manifest_sha256':SOURCE_MANIFEST_SHA,'manifest_sha256':msha,'plan_sha256':psha,'global_seed':42,'object_seed':31415,'epochs':64,'updates':3584,'optimizer':opt_audit,'trainability':counts,'gc_alpha':GC_ALPHA,'gpu':gpu,'fixed_eval_epochs':EVAL_EPOCHS,'model_loss_unchanged_from_spec':True}
    write_json(reports/'run_manifest.json',run_manifest)
    splits={k:manifest[k] for k in ('train_probe8','train_all56','same_scene_holdout8','dev8','val8','val32')}
    nodes=[];pergt=[];rng=capture_rng();model.train()
    offnames=('same_scene_holdout8','dev8','train_all56','val32')
    n,g,q=_eval(model,opt,splits,reports,0,('train_probe8','same_scene_holdout8','dev8','train_all56','val32'),offnames,True);nodes.append(n);pergt+=g;queryrows=q;restore_rng(rng);model.train()
    _save(run/'checkpoint_epoch_00.pt',model,opt,optimizer,0,0,psha,msha,{'total_exposures':0})
    exposure={_key(w):0 for w in splits['train_all56']};log=reports/'training_metrics.jsonl'
    for epoch in range(64):
      perm=np.random.default_rng(42+epoch).permutation(56)
      for pos,wi in enumerate(perm):
        step=epoch*56+pos+1;win=splits['train_all56'][int(wi)];batch=build_batch(opt,win,'cuda');mult=lr_mult(step)
        out,metrics=train_one_step(model,optimizer,batch,step,understanding_weight_value=min(step/200,1.),lr_values=(OBJECT_PEAK_LR*mult,RECON_PEAK_LR*mult),failure_capture_dir=run/'failures',failure_context={'epoch':epoch,'position':pos,'window':win})
        exposure[_key(win)]+=1
        if step%20==0 or step==3584:
          row={k:v for k,v in metrics.items() if k!='classification_confusion'};row.update({'epoch':epoch+1,'position':pos,'global_step':step,'scene':win['scene'],'context':win['context'],'novel':win['novel'],'gc_alpha':GC_ALPHA,'allocated_gib':torch.cuda.memory_allocated()/1024**3,'reserved_gib':torch.cuda.memory_reserved()/1024**3})
          with log.open('a') as f:f.write(json.dumps(jsonable(row),allow_nan=False)+'\n')
        del out,metrics,batch
      e=epoch+1;step=e*56;_save(run/f'checkpoint_epoch_{e:02d}.pt',model,opt,optimizer,e,step,psha,msha,{'unique_windows':56,'min_exposures':min(exposure.values()),'max_exposures':max(exposure.values()),'total_exposures':sum(exposure.values())})
      if e in EVAL_EPOCHS[1:]:
        names=('train_probe8','same_scene_holdout8','dev8')
        official=('same_scene_holdout8','dev8','train_all56','val32') if e in (32,64) else ()
        if official:names+=('train_all56','val32')
        rng=capture_rng();n,g,q=_eval(model,opt,splits,reports,step,names,official,e in (32,64));nodes.append(n);pergt+=g;queryrows+=q;restore_rng(rng);model.train()
    summary={'completed_updates':3584,'epochs':64,'scene_count':8,'train_windows':56,'exposures_each':64,'gpu':gpu,'source_checkpoint':str(PRETRAINED),'source_checkpoint_sha256':PRETRAINED_SHA,'source_step':PRETRAINED_STEP,'manifest_sha256':msha,'plan_sha256':psha}
    _write_outputs(reports,nodes,pergt,queryrows,summary);_write_package_metadata(reports);_package(reports)
    print(json.dumps({'complete':True,'updates':3584,'bundle':str(reports/'result_bundle.zip'),'bytes':(reports/'result_bundle.zip').stat().st_size}),flush=True)

if __name__=='__main__':main()
