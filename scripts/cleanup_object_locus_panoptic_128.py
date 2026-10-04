"""Authorized exact-run cleanup after independently verified model-only exports."""
from pathlib import Path
import dataclasses,hashlib,json,os,shutil,subprocess
import torch
from scripts import object_locus_panoptic_v1_runtime as runtime
OLD_REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_v1_8gpu')
OLD_RUN=Path('/space/mawb/ssst/workspace_group_plus/object_locus_panoptic_v1_8gpu')
NEW_REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
BASE='7300b6ff6ae963ea7228ae7bc0c7d0e0446eefa3'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 NEW_REPORT.mkdir(parents=True,exist_ok=True);before=shutil.disk_usage(OLD_RUN).free
 if (NEW_REPORT/'cleanup_128.json').exists():raise RuntimeError('cleanup already completed; do not repeat')
 torch.set_num_threads(4);model,opt=runtime.build_model('cpu',report=False);kept=[]
 outdir=OLD_RUN/'retained_model_only';outdir.mkdir(exist_ok=True)
 for epoch in (8,16,64):
  source=OLD_RUN/f'checkpoint_epoch_{epoch:02}.pt';blob=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
  assert blob['git_sha']==BASE;source_sha=sha(source)
  payload={k:blob[k] for k in ('epoch','completed_updates','completed_exposures','git_sha','world_size')}
  payload.update(model=blob['model'],config=dataclasses.asdict(opt),source_path=str(source),source_sha256=source_sha,format='complete_model_only_no_optimizer_scheduler_rng')
  output=outdir/f'epoch{epoch:02}_model_only.pt';tmp=output.with_suffix('.tmp');torch.save(payload,tmp)
  check=torch.load(tmp,map_location='cpu',weights_only=False,mmap=True)
  assert not {'optimizer','scheduler','rank_rng','rng'}&set(check);assert set(check['model'])==set(blob['model'])
  for name,tensor in blob['model'].items():
   other=check['model'][name];assert tensor.shape==other.shape and tensor.dtype==other.dtype and torch.equal(tensor,other),name
  model.load_state_dict(check['model'],strict=True)
  for name,tensor in model.state_dict().items():assert torch.equal(tensor,check['model'][name]),name
  os.replace(tmp,output);kept.append(dict(epoch=epoch,path=str(output),bytes=output.stat().st_size,sha256=sha(output),source=str(source),source_sha256=source_sha,tensors=len(check['model']),exact_tensor_equality=True,strict_model_reload=True,config=payload['config'],code_sha=BASE));del blob,payload,check
  print('model-only validated',epoch,flush=True)
 del model
 # Exact paths only. Final metrics/manifests/logs and three ZIPs are untouched.
 targets=[OLD_RUN/f'checkpoint_epoch_{e:02}.pt' for e in (0,2,4,8,16,32,64)]+[OLD_RUN/'resume_latest.pt',OLD_RUN/'resume_previous.pt',OLD_RUN/'old_startup_checkpoints',OLD_REPORT/'pre_level_embedding_fix_smoke',OLD_REPORT/'pre_raw24_combined_fix',OLD_REPORT/'results_oversized_original.zip']
 deleted=[]
 for p in targets:
  if not p.exists():continue
  files=sorted(x for x in p.rglob('*') if x.is_file() and not x.is_symlink()) if p.is_dir() else [p]
  entries=[dict(path=str(x),bytes=x.stat().st_size) for x in files];deleted.extend(entries)
  if p.is_dir():shutil.rmtree(p)
  else:p.unlink()
 archive_names=set()
 import zipfile
 for name in ('results.zip','images_part01.zip','images_part02.zip'):
  with zipfile.ZipFile(OLD_REPORT/name) as z:archive_names.update(z.namelist())
 # Only delete official PNGs demonstrably included in final delivery archives.
 for p in sorted((OLD_REPORT/'official').rglob('*.png')):
  if str(p.relative_to(OLD_REPORT)) in archive_names:
   deleted.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
 after=shutil.disk_usage(OLD_RUN).free
 report=dict(retained_checkpoints=kept,deleted=deleted,deleted_bytes=sum(x['bytes'] for x in deleted),free_before=before,free_after=after,free_delta=after-before,official_note='Current final ZIPs do not contain bulk official raw PNG exports; not covered by conditional deletion authorization, so retained.',retained_reports=str(OLD_REPORT),other_experiments_untouched=True)
 (NEW_REPORT/'cleanup_128.json').write_text(json.dumps(report,indent=2)+'\n');print('cleanup bytes',report['deleted_bytes'],'free delta',after-before,flush=True)
if __name__=='__main__':main()
