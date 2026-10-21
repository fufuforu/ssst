"""Register immutable deferred benchmark assets; never execute evaluation."""
from pathlib import Path
import hashlib,json,shutil
REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
def main():
 manifest=json.loads((REPORT/'manifest.json').read_text());plan=json.loads((REPORT/'training_plan.json').read_text());source=Path(manifest['full_validation_source']);target=REPORT/'full_validation_manifest.json'
 if target.exists() and target.read_bytes()!=source.read_bytes():raise RuntimeError('registered full validation changed')
 shutil.copyfile(source,target)
 payload=json.loads(target.read_text());records=payload.get('records',[]) if isinstance(payload,dict) else payload;scenes={r.get('scan',r.get('scene')) for r in records};assert not scenes&set(manifest['actual_train_scenes'])
 dev={w['scene'] for w in manifest['monitor_splits']['dev8']};val={w['scene'] for w in manifest['monitor_splits']['val32']};assert dev<=val
 report=dict(execution='WAIT_USER_INSTRUCTION; not launched',checkpoint_epochs=[0,1,2,4,6,8],checkpoint_updates=[e*plan['U'] for e in (0,1,2,4,6,8)],monitor_splits=manifest['monitor_splits'],selection='dev8 true-novel official packed AP50; exact ties earlier epoch; no val32 selection',dev8_subset_val32=True,full_manifest_path=str(target),full_manifest_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),full_windows=len(records),full_scenes=len(scenes),full_validation_evaluation='selected best and epoch8, same checkpoint only once; report aggregate and aggregate excluding dev8 scenes',scopes=['context:all/context','target-all:all/target','true novel:novel/target'],metrics=['mIoU','PQ','mAP','AP50','PSNR','SSIM','LPIPS'],posed_setting=True,restore_exposure='checkpoint.completed_exposures; fallback8*completed_updates')
 (REPORT/'deferred_evaluation_plan.json').write_text(json.dumps(report,indent=2)+'\n');print('registered only:',len(records),'full validation windows,',len(scenes),'scenes')
if __name__=='__main__':main()
