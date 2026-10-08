#!/usr/bin/env python3
"""Frozen, strict R3D epoch8 inference on the registered 32 windows only."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.extract_object_locus_frozen_probe import extract_one,sha,write_json,state_sha,raw_iou
from scripts.object_locus_r3d_registration import normalize_r3d_registration

CKPT=Path('/space/mawb/ssst/workspace_group_plus/object_locus_output_refine_gc001_v1/attempts/attempt02/checkpoint_epoch8.pt')
R3D_ROOT=Path('/space/mawb/ssst/group_plus/object_locus_output_refine_gc001_v1/attempts/attempt02')
EXPECTED_CODE='d4c096b80c4a28e93e095abac7dd777ed1af5f3a'
EXPECTED_CKPT_SHA='9f06d78c58bd5e00b84ed712840e767c3151db1fe170f3408c6079fca263db2c'
EXPECTED_SOURCE='68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a'
EXPECTED_PLAN='0a7c8173b3dcc187d7ce3074e69d9c8649204a933262ded3a773b08898650ea8'
R3D_SOURCE_FILES={
 'tokengs/models/object_locus_output_refine_gc001.py':{'blob':'16219514ee7daeb83335a061a7295e4c7d8ecf6f','content':'01feb24de4daf00f183ef21476f1c2e4badc0b4a18299b114faa0009d9eafe68'},
 'tokengs/models/object_locus_output_refine_v1.py':{'blob':'c0f98010dfff77bedfa92bb784fc28707b63a62e','content':'f187e62c8b2989a7f8245bc4f5d0b68cfa8073ead697f10fcd8841dcacb29e18'},
 'scripts/object_locus_output_refine_gc001_runtime.py':{'blob':'8fa4f4c9d51d315a80cd5b308d2a57e12c1fe874','content':'9826398d9734ddce527f8ef8ca4186ee5dac77c5de5d165a8c48dbd57a12bdf0'}}
def state_digest(model):
 h=hashlib.sha256()
 for n,v in sorted(model.state_dict().items()):
  a=v.detach().contiguous().cpu().numpy();h.update(n.encode());h.update(str(a.dtype).encode());h.update(np.asarray(a.shape,np.int64).tobytes());h.update(a.tobytes())
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=Path,required=True);a=ap.parse_args();root=a.attempt
 ckpt_sha=sha(CKPT)
 if ckpt_sha!=EXPECTED_CKPT_SHA:raise RuntimeError(f'R3D epoch8 checkpoint SHA mismatch: {ckpt_sha}')
 blob_identity=torch.load(CKPT,map_location='cpu',weights_only=False,mmap=True)
 checkpoint_metadata={k:blob_identity[k] for k in ('epoch','completed_updates','new_exposures','source_exposure','model_exposure','alpha','code_sha','source_sha256','plan_sha256')}
 checkpoint_metadata['checkpoint_path']=str(CKPT);del blob_identity
 normalized_registration=normalize_r3d_registration(R3D_ROOT,checkpoint_metadata=checkpoint_metadata,checkpoint_sha256=ckpt_sha)
 write_json(root/'r3d_registration_normalized.json',normalized_registration)
 identity=json.loads((root/'cohort_identity.json').read_text())
 if identity.get('source_manifest_sha256')!='a03e6842e9657d512a0de2b9dd20ed73aa9bcc61ec8f4fd8d33ed07e8cc11249' or identity.get('dev_test_identity_matches_registered_four_arm') is not True:raise RuntimeError('prelocked public cohort identity receipt invalid')
 dev,test=identity['dev'],identity['test']
 windows=[('test',i,w) for i,w in enumerate(test)]+[('dev',i,w) for i,w in enumerate(json.loads(Path('/space/mawb/ssst/group_plus/object_locus_panoptic_v1_8gpu/data_manifest.json').read_text())['dev8'])]
 if len(windows)!=32:raise RuntimeError('R3D requires fixed dev8 + test24 window cohort')
 for rel,info in R3D_SOURCE_FILES.items():
  if sha(ROOT/rel)!=info['content']:raise RuntimeError(f'copied R3D source content hash mismatch: {rel}')
 (root/'r3d_endpoint_manifest.json').write_text(json.dumps({'status':'LOCKED','checkpoint_path':str(CKPT),'checkpoint_sha256':ckpt_sha,'checkpoint_bytes':CKPT.stat().st_size,
  'training_attempt':str(R3D_ROOT),'training_code_sha':EXPECTED_CODE,'source_checkpoint_sha256':EXPECTED_SOURCE,'plan_sha256':EXPECTED_PLAN,
  'completed_updates':1008,'new_exposures':8064,'source_exposure':50064,'model_exposure':58128,'epoch':8,'alpha':.01,
  'world_size':4,'logical_global_slots':8,'gradient_accumulation_steps':2,'registered_refiner_parameters':527616,
  'code_source_commit':EXPECTED_CODE,'copied_source_files':R3D_SOURCE_FILES,'source_registration':json.loads((R3D_ROOT/'evaluation_registration.json').read_text()),
  'checkpoint_metadata':checkpoint_metadata,'normalized_registration_sha256':normalized_registration['registration_sha256']},indent=2)+'\n')
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError(f'R3D inference needs exactly one visible assigned GPU, got {torch.cuda.device_count()}')
 if not str(os.environ.get('SLURMD_NODENAME','')).startswith('3dimage-11') or torch.cuda.get_device_name(0)!='NVIDIA GeForce RTX 3090':raise RuntimeError('R3D inference requires one assigned RTX3090 on 3dimage-11')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_grad_enabled(False)
 from scripts.object_locus_output_refine_gc001_runtime import load_checkpoint
 device=torch.device('cuda:0');model,opt,meta=load_checkpoint(str(CKPT),device)
 if meta.get('code_sha')!=EXPECTED_CODE or meta.get('completed_updates')!=1008 or meta.get('model_exposure')!=58128 or meta.get('epoch')!=8 or meta.get('alpha')!=.01:raise RuntimeError('loaded R3D checkpoint metadata does not meet registered endpoint')
 for p in model.parameters():p.requires_grad_(False);p.grad=None
 refiner_params=sum(p.numel() for n,p in model.named_parameters() if n.startswith('panoptic.output_3d_refine.'))
 if refiner_params!=527616:raise RuntimeError(f'R3D refiner parameter count {refiner_params} != 527616')
 model.eval();before=state_digest(model)
 from scripts.object_locus_v3_set_runtime import build_batch
 from scripts.export_object_locus_v3_set_official import write_official_pair
 refs=root/'r3d'/'reference_gpu';cache=root/'r3d'/'cache';features=root/'r3d'/'features';reconroot=root/'reconstruction_cache'/'R3D'
 for d in (refs,cache,features,reconroot):d.mkdir(parents=True,exist_ok=True)
 records=[];torch.cuda.reset_peak_memory_stats()
 for n,(split,i,w) in enumerate(windows):
  batch,out,data,gtrows,iou,inter,union,hpairs,stats=extract_one(model,opt,w,split,i,build_batch,device)
  wid=f'{split}_{i:04d}_{w["scene"]}_c{"_".join(map(str,w["context"]))}';np.savez(cache/f'{wid}.npz',region=data['region'],alpha=data['alpha'],sem=data['sem'],ins=data['ins'],frame_ids=np.asarray(stats['frame_ids'],np.int64))
  ioufiles={}
  for scope,ids in [('context',[0,1]),('true-novel',[j for j,f in enumerate(stats['frame_ids']) if int(f) in set(map(int,w['novel']))])]:
   rr,ii,iv,uv=raw_iou(torch.from_numpy(data['region'][ids]),torch.from_numpy(data['alpha'][ids]),torch.from_numpy(data['sem'][ids]),torch.from_numpy(data['ins'][ids]))
   ip=cache/f'{wid}_{scope}_iou.npz';np.savez(ip,iou=ii,intersections=iv,unions=uv,gt_ids=np.asarray([x[0] for x in rr],np.int64),gt_classes=np.asarray([x[1] for x in rr],np.int64),
    window_id=np.asarray(wid),scope=np.asarray(scope),frame_ids=np.asarray([stats['frame_ids'][j] for j in ids],np.int64))
   ioufiles[scope]={'path':str(ip),'sha256':sha(ip),'size':ip.stat().st_size}
  np.savez(features/f'{wid}.npz',q=data['q'],z=data['z'],logits=data['logits'],pclass=data['pclass'],mass=data['mass'],frame_ids=np.asarray(stats['frame_ids'],np.int64))
  write_official_pair(out,batch,w,refs,target_frames='novel')
  depth=out['render']['depths_pred'];depth=depth[:,:,0] if depth.ndim==5 else depth
  rp=reconroot/split/f'{w["scene"]}_context{"_".join(map(str,w["context"]))}.npz';rp.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(rp,frame_ids=np.asarray(stats['frame_ids'],np.int64),context_ids=np.asarray(w['context'],np.int64),novel_ids=np.asarray(w['novel'],np.int64),
   pred_rgb=out['render']['images_pred'][0].detach().float().clamp(0,1).cpu().numpy(),gt_rgb=batch['images_all'][0].detach().float().clamp(0,1).cpu().numpy(),
   pred_depth=depth[0].detach().float().cpu().numpy(),gt_depth_m=batch['depth_gt_m_all'][0,:,0].detach().float().cpu().numpy(),depth_valid=batch['depth_gt_valid_all'][0,:,0].detach().bool().cpu().numpy())
  if any(not np.isfinite(data[k]).all() for k in ('q','z','region','alpha','pclass')):raise FloatingPointError(f'R3D nonfinite output {wid}')
  records.append({'window_id':wid,'split':split,'scene':w['scene'],'context':w['context'],'novel':w['novel'],'frame_ids':stats['frame_ids'],
   'cache':str(cache/f'{wid}.npz'),'cache_sha256':sha(cache/f'{wid}.npz'),'features':str(features/f'{wid}.npz'),'features_sha256':sha(features/f'{wid}.npz'),
   'scope_iou':ioufiles,'reconstruction_cache':str(rp),'reconstruction_sha256':sha(rp),'state_sha256':before,'finite':True})
  if n==0:write_json(root/'r3d_startup_confirmation.json',{'status':'PASS','first_window':wid,'finite':True,'state_sha256':before,'peak_allocated_bytes':int(torch.cuda.max_memory_allocated())})
  print(f'R3D cached {split} {i+1} / 32 {wid}',flush=True);del batch,out,data;torch.cuda.empty_cache()
 after=state_digest(model)
 if before!=after or any(p.grad is not None for p in model.parameters()):raise RuntimeError('R3D freeze/state/gradient check failed')
 write_json(root/'r3d_frozen_inference_receipt.json',{'status':'PASS','windows':32,'dev':8,'test':24,'endpoint_sha256':ckpt_sha,'state_sha256_before':before,'state_sha256_after':after,'requires_grad_parameters':0,'non_null_grads':0,'all_finite':True,'peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
  'gpu_name':torch.cuda.get_device_name(0),'visible_devices':torch.cuda.device_count(),'slurm_job_id':os.getenv('SLURM_JOB_ID'),'slurm_node':os.getenv('SLURMD_NODENAME'),'slurm_partition':os.getenv('SLURM_JOB_PARTITION'),'slurm_gres':'gpu:1','forward_step':58128,'beta':.1,'records':records})
 write_json(root/'r3d_cache_manifest.json',{'schema':'GC001 shared dev/test cache plus R3D q/z/region/alpha/pclass/GT/frame identity','records':records,'window_count':32})
 write_json(root/'r3d_h0_cache_pair_identity.json',{'same_manifest_and_window_identity':True,'same_GT_frame_ids':True,'probe_readout_does_not_reuse_R3D_mask':True,'windows':records})
if __name__=='__main__':main()
