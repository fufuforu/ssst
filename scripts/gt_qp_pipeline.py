"""One fresh four-epoch GT-camera experiment, immutable export, official scoring."""
from pathlib import Path
import os,sys,json,subprocess,shutil,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.verified_runtime_stage import load_run_config,sha,copy_verified
from scripts.b_continuation_pipeline import pack,extract,publish_bytes
from scripts.evaluation_workspace import local_root
from scripts.qp_anchor_pipeline import make_novel,metric
PYTRAIN='/space/mawb/anaconda3/envs/tokengs/bin/python'
PYSCORE='/space/mawb/SIU3R/.venv_gpu_v4/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):publish_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode(),p)

def validate(run,cfg):
    receipt=read(run/'train/TRAINING_COMPLETE.json');assert receipt['status']=='PASSED' and receipt['full_updates']==4172 and receipt['camera_protocol']==cfg['camera_protocol']
    for rank in range(8):
        a=read(run/f'train/UPDATE_AUDIT_rank{rank:02d}.json');f=read(run/f'train/FROZEN_ENDPOINT_rank{rank:02d}.json')
        assert a['DDP_replicas_equal'] and all(a['required_changed'].values()) and f['contents_equal'] and f['status']=='PASSED'
        assert read(run/f'train/ISOLATION_rank{rank:02d}.json')['status']=='PASSED'
        c=read(run/f'train/GT_CAMERA_CONTRACT_rank{rank:02d}.json');assert c['protocol']==cfg['camera_protocol'] and not c['target_RGB_used_for_camera'] and c['Context_only'] and c['Omega_depth_reprojected'] and not c['Target_cameras_used_for_generation']
    assert read(run/'train/OPTIMIZER_ENDPOINT_AUDIT.json')['reconstruction_steps']==4172

def export_run(run,cfg):
    validate(run,cfg);local=local_root();lr=local/'GT';ev=lr/'eval';(ev/'evaluation').mkdir(parents=True)
    (lr/'configuration.json').write_text(json.dumps(cfg));subprocess.run([PYTRAIN,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4','--','scripts/export_qp_anchor_gt_camera.py',str(run),str(ev/'evaluation')],cwd=ROOT,check=True)
    exp=read(ev/'evaluation/export_complete.json');assert exp['camera_protocol']==cfg['camera_protocol'] and len(exp['records'])==1860
    archive=local/'evaluation.tar';files=pack(lr,archive);h=sha(archive);name='GT_epoch4_'+h+'.tar';candidates=[str(Path(cfg['artifact_root'])/name),str(run/'archives'/name)]
    for dest in candidates:
        p=Path(dest);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();tmp=p.with_suffix('.partial');shutil.copyfile(archive,tmp);assert sha(tmp)==h;tmp.rename(p);p.chmod(0o444)
    for rel in ['evaluation/export_complete.json','evaluation/checkpoint_identity.json','evaluation/gt_reference_manifest.jsonl','FINAL_CHECKPOINT_FROZEN.json']:publish_bytes((ev/rel).read_bytes(),run/'eval'/rel)
    write(run/'EXPORT_COMPLETE.json',dict(status='PASSED',sha256=h,candidates=candidates,windows=1860,scenes=312,files=len(files),camera_protocol=cfg['camera_protocol'],empty_depth_views=sum(len(r['empty_depth_views']) for r in exp['records']),checkpoint_sha256=read(ev/'FINAL_CHECKPOINT_FROZEN.json')['sha256']))
    shutil.rmtree(local)

def score_run(run,cfg):
    pin=read(run/'EXPORT_COMPLETE.json');assert pin['status']=='PASSED';local=local_root();archive=local/'evaluation.tar';copy_verified(pin['candidates'],archive,pin['sha256']);lr=local/'GT';lr.mkdir();extract(archive,lr);ev=lr/'eval'
    (ev/'configuration.json').write_text(json.dumps(cfg));env={k:v for k,v in os.environ.items() if not k.startswith('SSST_RUNTIME_') and k!='SSST_LOCAL_CODE'}
    p=subprocess.run([PYSCORE,'scripts/score_dual_stream_v1.py','--run-root',str(ev)],cwd=ROOT,env=env)
    if p.returncode:
        for rel in ['evaluation/metrics_invalid.json','evaluation/siu3r_table1_invalid.csv','evaluation/all/results.json']:
            if (ev/rel).exists():publish_bytes((ev/rel).read_bytes(),run/'eval'/rel)
        write(run/'SCORING_FAILED.json',dict(status='FAILED',returncode=p.returncode,empty_depth_views=pin['empty_depth_views'],predictions_kept=True,no_samples_filtered=True,official_evaluator_unchanged=True))
        raise RuntimeError('Official scoring failed; available invalid metrics have been preserved')
    make_novel(ev,ev/'novel');subprocess.run([PYSCORE,'scripts/invoke_siu3r_official_evaluator.py','--eval-path',str(ev/'novel'),'--output',str(ev/'novel_metrics.json'),'--recon-only','--device','cuda:0'],cwd=ROOT,env=env,check=True)
    m=read(ev/'evaluation/metrics.json');assert m['status']=='OFFICIAL_VAL_SCORED' and m['windows']==1860 and not m['nonfinite_native_paths']
    current=metric(m['official_native_result']);old=cfg['reference_metrics'];result=dict(status='PASSED',recipe=cfg['recipe'],native=current,novel=read(ev/'novel_metrics.json'),reference_predicted_camera=old,delta_from_original_QPanchor1_epoch4={k:current[k]-old[k] for k in current},source_initialization_sha256=cfg['source_sha256'],checkpoint_sha256=pin['checkpoint_sha256'],camera_protocol=cfg['camera_protocol'],GT_provider_units_per_meter=.15,scene_units_per_meter='per-window .15*a',Context_GT_camera_conditioned=True,training_updates=4172,training_epochs=4,token_count=1024,context_views=2,windows=1860,scenes=312,quality_head=False,parameter_updates_during_export=0,caveat='Context geometry is reprojected with GT Context cameras in one bounded Context-only gauge; Target GT cameras use that same gauge. Changes include generation camera conditioning, supervision and evaluation versus the old predicted-camera baseline; this is not a training-only ablation. Original background/clipping/PNG writer retained; no GT depth in generation or alignment.')
    for rel in ['evaluation/metrics.json','evaluation/siu3r_table1.csv','evaluation/siu3r_table1.md','evaluation/comparison_to_old_C.csv','novel_metrics.json']:publish_bytes((ev/rel).read_bytes(),run/'eval'/rel)
    write(run/'RESULTS.json',result);write(run/'FORMAL_COMPLETE.json',dict(status='PASSED',windows=1860,scenes=312,archive_sha256=pin['sha256'],checkpoint_sha256=pin['checkpoint_sha256'],metrics_sha256=sha(run/'eval/evaluation/metrics.json')));shutil.rmtree(local)

def summary(run,cfg):
    r=read(run/'RESULTS.json');assert read(run/'FORMAL_COMPLETE.json')['status']=='PASSED';keys=list(r['native']);old=r['reference_predicted_camera'];new=r['native']
    with (run/'COMPARISON.csv').open('x') as f:
        w=csv.writer(f);w.writerow(['metric','original_predicted_camera','GT_train_and_eval','delta'])
        for k in keys:w.writerow([k,old[k],new[k],new[k]-old[k]])
    rows=['# QPanchor1 epoch4: GT camera training and evaluation','','|metric|original predicted camera|GT train/eval|delta|','|---|---:|---:|---:|']+['|'+k+'|'+f'{old[k]:.6f}|{new[k]:.6f}|{new[k]-old[k]:+.6f}|' for k in keys]+['',r['caveat']]
    publish_bytes(('\n'.join(rows)+'\n').encode(),run/'COMPARISON.md');write(run/'SUMMARY_COMPLETE.json',dict(status='PASSED',camera_protocol=cfg['camera_protocol']))
if __name__=='__main__':
    run=Path(sys.argv[1]);mode=sys.argv[2];cfg=load_run_config(run)
    if mode=='export':export_run(run,cfg)
    elif mode=='score':score_run(run,cfg)
    elif mode=='summary':summary(run,cfg)
    else:raise ValueError(mode)
