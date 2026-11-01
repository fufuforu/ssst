"""Three frozen epoch evaluations; local export/score and verified archive publication."""
from pathlib import Path
import sys,os,json,hashlib,shutil,subprocess,tempfile,tarfile,io,csv,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.verified_runtime_stage import sha,copy_verified,load_run_config
from scripts.continuation_export_validation import expected,png
from scripts.olda4_ablation_pipeline import signatures,gt
from scripts.qp_anchor_pipeline import make_novel,metric
PYTRAIN='/space/mawb/anaconda3/envs/tokengs/bin/python'
PYSCORE='/space/mawb/SIU3R/.venv_gpu_v4/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();publish_bytes(data,p)
def publish_bytes(data,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256(data).hexdigest()
 if p.exists():assert sha(p)==digest;return
 tmp=p.with_name(p.name+'.partial_'+os.environ.get('SLURM_JOB_ID','cpu'))
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 assert sha(tmp)==digest;tmp.rename(p);assert sha(p)==digest

def epoch_run(run,epoch):return run/f'epoch{epoch:02d}'
def local_root():
 parent=Path('/tmp/mawb_B_continue3_eval');parent.mkdir(exist_ok=True)
 st=os.statvfs(parent)
 assert shutil.disk_usage(parent).free>24*1024**3 and st.f_favail>150000,'Insufficient node-local evaluation capacity'
 return Path(tempfile.mkdtemp(prefix=os.environ.get('SLURM_JOB_ID','cpu')+'_',dir=parent))
def comparison_identity(ev,reference):
 assert signatures(ev)==signatures(reference/'eval') and gt(ev)==gt(reference/'eval')
 exp=read(ev/'evaluation/export_complete.json');ref=read(reference/'eval/evaluation/export_complete.json')
 generation=lambda x:{r['name']:(r['generation_frame_ids'],r['generation_input_rgb_sha256']) for r in x['records']}
 assert generation(exp)==generation(ref)
 assert len({r['name'].split('_context')[0] for r in exp['records']})==312
 for row in exp['records']:assert row['generation_frame_ids']==row['frame_ids'][:2]
 return dict(same_evaluation_frames_and_GT=True,same_generation_inputs=True,windows=1860,scenes=312)
def pack(local_run,archive):
 ev=local_run/'eval';exp=read(ev/'evaluation/export_complete.json');assert exp['status']=='EXPORTED' and len(exp['records'])==1860
 paths=[local_run/'configuration.json']
 for row in exp['records']:
  for rel in expected(row):
   p=ev/'evaluation/all'/row['name']/rel;data=p.read_bytes()
   png(data) if rel.endswith('.png') else json.loads(data)
   paths.append(p)
 paths += [ev/'evaluation'/n for n in ('export_complete.json','checkpoint_identity.json','gt_reference_manifest.jsonl')]
 paths += [ev/'FINAL_CHECKPOINT_FROZEN.json']
 files={}
 with tarfile.open(archive,'w') as tf:
  for p in paths:
   data=p.read_bytes();name=str(p.relative_to(local_run));member=tarfile.TarInfo(name);member.size=len(data);member.mode=0o644;tf.addfile(member,io.BytesIO(data));files[name]=hashlib.sha256(data).hexdigest()
  data=json.dumps(files,sort_keys=True).encode();member=tarfile.TarInfo('FILES.json');member.size=len(data);tf.addfile(member,io.BytesIO(data))
 with tarfile.open(archive) as tf:
  for name,digest in files.items():assert hashlib.sha256(tf.extractfile(name).read()).hexdigest()==digest,name
 return files

def export_epoch(run,epoch,cfg):
 local=local_root();local_run=local/'B';ev=local_run/'eval';(ev/'evaluation').mkdir(parents=True)
 (local_run/'configuration.json').write_text(json.dumps(cfg))
 env=dict(os.environ,SSST_EVAL_LOCAL_RUN=str(local_run))
 subprocess.run([PYTRAIN,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4','--','scripts/export_b_continue3.py','--run-root',str(run),'--epoch',str(epoch)],cwd=ROOT,env=env,check=True)
 comparison_identity(ev,Path(cfg['parent_run']))
 archive=local/'evaluation.tar';files=pack(local_run,archive);digest=sha(archive)
 name=f'epoch{epoch:02d}_{digest}.tar';candidates=[str(Path(cfg['artifact_root'])/name),str(run/'archives'/name)]
 for destination in candidates:
  target=Path(destination);target.parent.mkdir(parents=True,exist_ok=True)
  assert not target.exists();temporary=target.with_suffix('.partial');shutil.copyfile(archive,temporary);assert sha(temporary)==digest
  temporary.rename(target);assert sha(target)==digest;target.chmod(0o444)
 er=epoch_run(run,epoch)
 # Only a handful of readable metadata files are published separately; PNGs stay archived.
 for relative in ('evaluation/export_complete.json','evaluation/checkpoint_identity.json','evaluation/gt_reference_manifest.jsonl','FINAL_CHECKPOINT_FROZEN.json'):
  publish_bytes((ev/relative).read_bytes(),er/'eval'/relative)
 write(er/'EXPORT_COMPLETE.json',dict(status='PASSED',cumulative_epochs=epoch,files=len(files),bytes=archive.stat().st_size,sha256=digest,candidates=candidates,checkpoint_sha256=read(ev/'FINAL_CHECKPOINT_FROZEN.json')['sha256'],local_files_all_verified=True))
 shutil.rmtree(local)

def extract(archive,dest):
 with tarfile.open(archive) as tf:
  for m in tf.getmembers():assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  tf.extractall(dest)
 files=read(dest/'FILES.json')
 for name,digest in files.items():assert sha(dest/name)==digest,name
 return files

def score_epoch(run,epoch,cfg):
 er=epoch_run(run,epoch);pin=read(er/'EXPORT_COMPLETE.json');assert pin['status']=='PASSED'
 local=local_root();archive=local/'evaluation.tar';copy_verified(pin['candidates'],archive,pin['sha256'])
 lr=local/'B';lr.mkdir();files=extract(archive,lr);ev=lr/'eval'
 assert read(ev/'FINAL_CHECKPOINT_FROZEN.json')['sha256']==pin['checkpoint_sha256']
 env={k:v for k,v in os.environ.items() if not k.startswith('SSST_RUNTIME_') and k not in ('SSST_LOCAL_CODE','SSST_EVAL_LOCAL_RUN')}
 subprocess.run([PYSCORE,'scripts/score_dual_stream_v1.py','--run-root',str(ev)],cwd=ROOT,env=env,check=True)
 make_novel(ev,ev/'novel')
 subprocess.run([PYSCORE,'scripts/invoke_siu3r_official_evaluator.py','--eval-path',str(ev/'novel'),'--output',str(ev/'novel_metrics.json'),'--recon-only','--device','cuda:0'],cwd=ROOT,env=env,check=True)
 m=read(ev/'evaluation/metrics.json');novel=read(ev/'novel_metrics.json')
 assert m['status']=='OFFICIAL_VAL_SCORED' and m['windows']==1860 and not m['nonfinite_native_paths']
 assert all(math.isfinite(novel['result'][k]) for k in ('absrel','rmse','psnr','ssim','lpips'))
 proof=comparison_identity(ev,Path(cfg['parent_run']))
 identity=read(ev/'evaluation/checkpoint_identity.json');meta=identity['meta']
 assert meta['code_sha']==cfg['code_sha'] and meta['cumulative_epochs']==epoch and meta['cumulative_updates']==epoch*1043
 assert meta['source_sha256']==cfg['source_sha256'] and meta['token_count']==4096 and meta['context_views']==2
 assert identity['parameter_updates']==0 and not identity['smoke_only']
 checkpoint=run/f'train/epoch{epoch:02d}';storage=read(checkpoint/'FINAL_CHECKPOINT_STORAGE.json')
 assert storage['status']=='PASSED' and storage['sha256']==pin['checkpoint_sha256']==sha(checkpoint/'checkpoint_live.pt')
 assert read(checkpoint/'EPOCH_COMPLETE.json')['progress']==dict(epoch=epoch-5,update_in_epoch=0,full_updates=(epoch-5)*1043)
 for rank in range(4):
  audit=read(checkpoint/f'UPDATE_AUDIT_rank{rank:02d}.json');frozen=read(checkpoint/f'FROZEN_ENDPOINT_rank{rank:02d}.json')
  assert audit['status']=='PASSED' and audit['DDP_replicas_equal'] and all(audit['required_changed'].values())
  assert frozen['contents_equal'] and frozen['parameter_updates']==0
  assert read(checkpoint/f'OPTIMIZER_ENDPOINT_rank{rank:02d}.json')['step']==(epoch-5)*1043
  assert read(run/f'train/INITIALIZATION_rank{rank:02d}.json')['all_shapes_and_values_exact']
  assert read(run/f'train/SOURCE_RNG_RESTORED_rank{rank:02d}.json')['status']=='PASSED'
  assert read(run/f'train/ISOLATION_rank{rank:02d}.json')['status']=='PASSED'
 baseline=read(Path(cfg['parent_run'])/'RESULTS.json');native=metric(m['official_native_result'])
 result=dict(status='PASSED',cumulative_epochs=epoch,local_epochs=epoch-5,token_count=4096,context_views=2,native=native,novel=novel,source_checkpoint_sha256=cfg['source_sha256'],checkpoint_sha256=pin['checkpoint_sha256'],delta_from_B_epoch5={k:native[k]-baseline['native'][k] for k in native},optimizer_reset_at_epoch5=True,optimizer_continuous_across_added_epochs=True,learning_rate=1e-5,**proof)
 for relative in ('evaluation/metrics.json','evaluation/siu3r_table1.csv','evaluation/siu3r_table1.md','evaluation/comparison_to_old_C.csv','novel_metrics.json'):
  publish_bytes((ev/relative).read_bytes(),er/'eval'/relative)
 write(er/'RESULTS.json',result)
 write(er/'FORMAL_COMPLETE.json',dict(status='PASSED',**proof,checkpoint_sha256=pin['checkpoint_sha256'],metrics_sha256=sha(er/'eval/evaluation/metrics.json'),archive_sha256=pin['sha256']))
 shutil.rmtree(local)

def summarize(run,cfg):
 assert read(run/'train/TRAINING_COMPLETE.json')['full_updates']==3129
 rows=[];baseline=read(Path(cfg['parent_run'])/'RESULTS.json');assert baseline['status']=='PASSED' and baseline['checkpoint_sha256']==cfg['source_sha256']
 results={5:baseline}
 for epoch in (6,7,8):
  er=epoch_run(run,epoch);assert read(er/'FORMAL_COMPLETE.json')['status']=='PASSED';r=read(er/'RESULTS.json');assert r['source_checkpoint_sha256']==cfg['source_sha256'] and r['cumulative_epochs']==epoch;results[epoch]=r
 for epoch,r in results.items():rows.append(dict(cumulative_epoch=epoch,**r['native']))
 write(run/'CONTINUATION_COMPARISON.json',dict(status='PASSED',primary='reconstruction',native=rows,novel4={str(e):r['novel']['result'] for e,r in results.items()},delta_from_B_epoch5={str(e):r.get('delta_from_B_epoch5',{}) for e,r in results.items()},caveat='B epoch5 optimizer was not retained: fresh Adam at start of extension, continuous thereafter; single seed'))
 with (run/'CONTINUATION_COMPARISON.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 keys=['absrel','rmse','psnr','ssim','lpips'];lines=['# B epoch5–8 continuation','','Same4096tokens/2views,constant1e-5,four GPUs/globalbatch8. Adam reset once after epoch5.','', '|epoch|'+'|'.join(keys)+'|','|---|'+'---:|'*5]
 lines+=['|'+str(row['cumulative_epoch'])+'|'+'|'.join(f'{row[k]:.6f}' for k in keys)+'|' for row in rows]
 (run/'CONTINUATION_COMPARISON.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 run=Path(sys.argv[1]);mode=sys.argv[2];cfg=load_run_config(run)
 if mode.startswith('export'):export_epoch(run,int(mode[6:]),cfg)
 elif mode.startswith('score'):score_epoch(run,int(mode[5:]),cfg)
 elif mode=='summary':summarize(run,cfg)
 else:raise ValueError(mode)
