"""Read source identities and create a unique, checkpoint-free diagnostic run directory."""
from pathlib import Path
import hashlib,json,subprocess
import torch
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/space/mawb/ssst/group_plus/anchor_child_boundary_diagnostics_20261021_v1')
COHORT=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json')
MANIFEST=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json')
SOURCES={
 'A4':dict(checkpoint='/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_epoch4.pt',sha256='9d3f3f788de59927073cf6d93335623ff37653025258e8e53a0a232f2e01a534',epoch=4,updates=4172),
 'A4plus4':dict(checkpoint='/space/mawb/ssst/group_plus/unified_query_training_v1/attempt08_legacy_dual_superpoint_extend8_node13_8gpu/checkpoint_latest.pt',sha256='0ed6cd8f4fb740698c01ca345a0bbad9a2b1b96302acdbf106864874e5e282c9',epoch=8,updates=8344)}

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 assert not RUN.exists(),'Refuse duplicate run'
 blobs={}
 for label,s in SOURCES.items():
  assert digest(s['checkpoint'])==s['sha256']
  b=torch.load(s['checkpoint'],weights_only=False,map_location='cpu',mmap=True);blobs[label]=b
  assert b['progress']==dict(epoch=s['epoch'],update_in_epoch=0,full_updates=s['updates'])
  assert b['meta']['architecture']=='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1'
  assert not any('iou_head.' in k for k in b['model'])
 a,b=blobs.values()
 for key in ('architecture','config','manifest_sha256','heldout_sha256','omega_identity','sam_identity','global_batch','understanding_weight','understanding_warmup'):
  assert a['meta'][key]==b['meta'][key],key
 assert a['model'].keys()==b['model'].keys()
 assert all(a['model'][k].shape==b['model'][k].shape for k in a['model'])
 assert b['meta']['continuation_sha256']==SOURCES['A4']['sha256']
 cohort_sha=digest(COHORT);manifest_sha=digest(MANIFEST)
 assert cohort_sha==a['meta']['heldout_sha256'] and manifest_sha==a['meta']['manifest_sha256']
 records=json.loads(COHORT.read_text())['records']
 assert len(records)==128 and len({r['official_pair']['scene'] for r in records})==128
 RUN.mkdir(parents=True)
 config=dict(sources=SOURCES,cohort=str(COHORT),cohort_sha256=cohort_sha,manifest_sha256=manifest_sha,
             code_root=str(ROOT),mask_resolution=128,output_resolution=256,supervised_layers=[6,8,10,12],
             contribution_chunk=64,context_views=2,training=False,backward_calls=0,parameter_updates=0,
             ownership='full-scene occlusion-aware alpha contributions × GT masks; no center projection',
             attribution=dict(min_visible_alpha_pixels=1.,min_thing_coverage=.5,purity=[.5,.7,.9],multi_instance_min_fraction=.1,multi_instance_min_alpha_pixels=1.),
             boundary=dict(raw='L2 query-space forward differences / sqrt(2), eligible thing queries after existing NMS',
                           soft='1-exp(-GaussianBlur(raw,sigma=1)/0.1)',threshold=.5,tolerances_pixels=[1,2,3],interior_exclusion_pixels=3,
                           pilot_gate=dict(f1_2px=.7,f1_lower95=.65,ap_2px=.5,max_interior_response_fraction=.2,max_interior_activation=.1)),
             smoke_windows=[0,1],full_windows=128,bootstrap_scene_resamples=4000,preview_windows=[0,1,2,3])
 (RUN/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
 (RUN/'PREFLIGHT.json').write_text(json.dumps(dict(status='PASSED',checkpoint_progress={k:v['progress'] for k,v in blobs.items()},
    checkpoint_meta={k:v['meta'] for k,v in blobs.items()},exact_continuation=True,same_keys_and_shapes=True,cohort_scenes=128,config_sha256=digest(RUN/'configuration.json')),indent=2)+'\n')
 print(json.dumps(dict(status='PASSED',run=str(RUN),sources=SOURCES)),flush=True)

if __name__=='__main__':main()
