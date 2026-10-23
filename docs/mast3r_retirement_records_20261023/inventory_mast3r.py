"""Inventory model archives by actual MASt3R state names, never unpickle tensors."""
from pathlib import Path
import os,json,zipfile,pickletools,collections,datetime
BASE=Path('/space/mawb/ssst');OUT=BASE/'group_plus/mast3r_checkpoint_retirement_20261023_v1'
ROOTS=[BASE/'workspace_group_plus',BASE/'group_plus']
PRUNE={'snapshots','.git','evaluation','all','images','images_all','rgb','rgb_gt','depth','depth_gt','target_seg_gt','target_seg_pred','context_seg_gt','context_seg_pred','exports','export','delivery','delivery_epoch6','tensorboard','slurm','visualizations','__pycache__','smoke_eval','smoke_runtime','contracts'}
SKIP_TOP={'semantic_resolution_v1','unified_query_training_v1','dual_stream_training_v1','dual_stream_v1_smoke','spatial_pair_20261023_v1','mast3r_checkpoint_retirement_20261023_v1','obsolete_experiment_retirement_20261023_v1'}
MARKERS=('understanding.encoder.enc_blocks.','understanding.mask2former.','pretrained_understanding.encoder.','pretrained_understanding.mask2former.')
# These patch-only families have explicit MASt3R provenance in their project specs.
DOC_FAMILIES={'vggt_classification_readback_v1':'docs/vggt_classification_readback_v1.md','vggt_classification_readback_full312_v1':'docs/vggt_classification_readback_v1.md','vggt_readout_probe_v1':'docs/vggt_readout_probe_v1.md','vggt_understanding_probes_v1':'docs/vggt_understanding_probes_v1.md','vggt_reading_controls_v1':'docs/vggt_reading_controls_v1.md','vggt_joint_followups_v1':'docs/vggt_joint_followups_v1.md','vggt_feedback_factorial_v1':'docs/vggt_feedback_factorial_v1.md','vggt_interaction_controls_v1':'docs/vggt_interaction_controls_v1.md','vggt_quality_full312_v1':'docs/vggt_quality_full312_v1.md','vggt_feature_reading_v1':'docs/vggt_feature_reading_v1.md','vggt_joint_short_controls_v1':'docs/vggt_joint_short_controls_v1.md'}
files=[];unknown=[];seen={};errors=[]
for root in ROOTS:
 for path,dirs,names in os.walk(root):
  rel=Path(path).relative_to(root);depth=len(rel.parts)
  dirs[:]=[d for d in dirs if d not in PRUNE and not d.startswith(('evaluation','eval_','official','export','aggregate','delivery','scene')) and (depth!=0 or d not in SKIP_TOP) and depth<9]
  for name in names:
   if not name.endswith(('.pt','.pth','.ckpt','.safetensors')):continue
   if 'official' in name.lower() or name.lower().startswith('cache_'):continue
   if not ('checkpoint' in name.lower() or name.lower().startswith(('latest_','best_')) or name.lower() in ['model.pt','model.pth','best.pt','last.pt','best.ckpt','last.ckpt'] or rel.parts and rel.parts[0] in DOC_FAMILIES):continue
   p=Path(path)/name
   if p.is_symlink():continue
   s=p.stat()
   if s.st_size<1024:continue
   family=rel.parts[0] if rel.parts else ''
   key=(s.st_dev,s.st_ino)
   if key not in seen:
    evidence=[]
    try:
     with zipfile.ZipFile(p) as z:
      member=next((m for m in z.infolist() if m.filename.endswith('/data.pkl') or m.filename=='data.pkl'),None)
      if member and member.file_size<32*2**20:
       data=z.read(member)
       for op,arg,pos in pickletools.genops(data):
        if op.name in ['BINUNICODE','SHORT_BINUNICODE','UNICODE'] and isinstance(arg,str) and arg.startswith(MARKERS):
         evidence.append(arg)
         if len(evidence)>=3:break
    except (OSError,zipfile.BadZipFile,ValueError) as e:errors.append(dict(path=str(p),error=str(e)))
    seen[key]=evidence
   evidence=seen[key]
   row=dict(path=str(p),family=root.name+'/'+family,device=s.st_dev,inode=s.st_ino,size=s.st_size,blocks=s.st_blocks,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns)
   if evidence:row.update(evidence='actual MASt3R/panoptic understanding state keys',evidence_keys=evidence);files.append(row)
   elif family in DOC_FAMILIES:row.update(evidence='documented MASt3R lineage/patch family',evidence_document=DOC_FAMILIES[family]);files.append(row)
   else:unknown.append(row)
by_inode=collections.defaultdict(list)
for row in files:by_inode[(row['device'],row['inode'])].append(row)
summary=collections.defaultdict(lambda:dict(paths=0,logical_GiB=0))
for row in files:summary[row['family']]['paths']+=1;summary[row['family']]['logical_GiB']+=row['size']/2**30
result=dict(at=datetime.datetime.now().isoformat(),status='INVENTORIED_NOT_DELETED',roots=[str(p) for p in ROOTS],candidates=files,unknown_not_selected=unknown,errors=errors,families=dict(summary),candidate_paths=len(files),unique_candidate_inodes=len(by_inode),allocated_GiB=sum(x[0]['blocks']*512 for x in by_inode.values())/2**30,pretrained_assets_datasets_code_and_raw_exports_excluded=True)
(OUT/'INVENTORY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['candidates','unknown_not_selected','errors']},indent=2),flush=True)
print('UNKNOWN',len(unknown),'ERRORS',len(errors),flush=True)
