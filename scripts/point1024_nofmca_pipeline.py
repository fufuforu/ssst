"""C/D gates and result comparisons. Never submits jobs or waits for results."""
from pathlib import Path
import sys,json,hashlib,os,csv
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
BASE=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1')

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def train_gate(root):
 from scripts.verify_point1024_fmquery_smoke import verify
 cfg=read(root/'configuration.json');smoke=Path(cfg['smoke_root'])
 receipt=verify(smoke);assert receipt==read(smoke/'SMOKE_PASSED.json')
 assert receipt['budget']['architecture']==cfg['architecture'] and receipt['budget']['code_sha']==cfg['code_sha']
 assert not (root/'STARTED.json').exists() and not (root/'checkpoint_latest.pt').exists()
 assert digest(cfg['training_window_manifest'])==cfg['training_window_manifest_sha256']
 assert digest(root/'checkpoint_untrained_initial.pt')==cfg['initial_checkpoint_sha256']
 write(root/'SMOKE_GATE_VERIFIED.json',dict(status='PASSED',smoke=str(smoke),code_sha=cfg['code_sha'],architecture=cfg['architecture']))


def eval_gate(root):
 cfg=read(root/'configuration.json');train=Path(cfg['training_root'])
 r=read(train/'TRAINING_COMPLETE.json');assert r['architecture']==cfg['architecture'] and r['progress']==dict(epoch=4,update_in_epoch=0,full_updates=4172)
 first=read(train/'FIRST_UPDATE_VERIFIED.json');assert first['status']=='PASSED' and len(first['ranks'])==8 and len({x['sha256'] for x in first['ranks']})==1 and all(x['changed'] for x in first['ranks'])
 meta=read(train/'STARTED.json')
 for k in ('architecture','code_sha','training_view_protocol','training_window_manifest_sha256','gradient_isolation','supervised_layers','layer_weights','object_anchor_attention','object_fm_attention','geometry_routing'):
  assert meta[k]==cfg[k],k
 assert meta['global_batch']==8 and meta['epochs']==4
 assert digest(cfg['cohort'])==cfg['cohort_sha256']
 target=root/'source_checkpoint.pt';assert not target.exists(),'Refuse duplicate evaluation'
 os.link(train/'checkpoint_latest.pt',target);cfg['checkpoint_sha256']=digest(target);write(root/'configuration.json',cfg)
 write(root/'FINAL_CHECKPOINT_FROZEN.json',dict(status='PASSED',sha256=cfg['checkpoint_sha256'],training_progress=r['progress'],source=str(train/'checkpoint_latest.pt'),hardlink=True))


def signatures(p):
 d=read(p/'evaluation/export_complete.json');assert d['windows']==1860 and d['parameter_updates']==0
 return {r['name']:(r['source_index'],r['frame_ids'],r['input_rgb_sha256']) for r in d['records']}
def gt(p):return {x['name']:x for x in map(json.loads,(p/'evaluation/gt_reference_manifest.jsonl').read_text().splitlines())}

def finish(root):
 cfg=read(root/'configuration.json');m=read(root/'evaluation/metrics.json');identity=read(root/'evaluation/checkpoint_identity.json')
 assert m['status']=='OFFICIAL_VAL_SCORED' and m['windows']==1860 and not m['nonfinite_native_paths'] and identity['parameter_updates']==0
 meta=identity['meta']
 for k in ('architecture','code_sha','training_view_protocol','training_window_manifest_sha256','gradient_isolation','supervised_layers','layer_weights','object_anchor_attention','object_fm_attention','geometry_routing'):
  assert meta[k]==cfg[k],k
 assert meta['epochs']==4 and meta['global_batch']==8 and not identity['smoke_only'] and digest(root/'source_checkpoint.pt')==cfg['checkpoint_sha256']
 refs={'A':BASE/'point1024_fmquery_isoadapter_official1860_after70026_20261021_v1','B':BASE/'point1024_isoadapter_noca_official1860_after70162_20261021_v1'}
 sig=signatures(root);gold=gt(root)
 for label,p in refs.items():assert signatures(p)==sig and gt(p)==gold
 write(root/'EVALUATION_VERIFIED.json',dict(status='PASSED',windows=1860,scenes=312,parameter_updates=0,matched_inputs=True,matched_gt=True,compared_with=list(refs),architecture=cfg['architecture']))
 parent=root.parents[1]
 refs.update({k:parent/k/'eval' for k in ('C','D')})
 if not all((p/'EVALUATION_VERIFIED.json').exists() for p in refs.values()):return
 rows=[]
 for label,p in refs.items():
  assert signatures(p)==sig and gt(p)==gold
  data=read(p/'evaluation/metrics.json')['official_segmentation']['all']
  row=dict(group=label,anchor_ca=label in ('A','C'),fm_ca=label in ('A','B'),psnr=data['psnr'],absrel=data['absrel'],rmse=data['rmse'],ssim=data['ssim'],lpips=data['lpips'])
  for scope in ('context','target'):
   row[scope+'_miou']=100*data[scope+'_miou'];row[scope+'_pq']=100*data[scope+'_pq'];row[scope+'_map']=100*data[scope+'_map']['map']
  rows.append(row)
 # Separate per-finisher comparison avoids concurrent writes to the same receipt.
 folder=root/'comparison';folder.mkdir(exist_ok=True)
 with (folder/'factorial.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 fields=list(rows[0]);lines=['# A/B/C/D CA factorial: official1860','',
  'Same 2+4 data, zero source, 4172 updates, LR/optimizer/clip, L6/L12. Single seed; inspect per-class and TP/FP/FN before capability conclusions. Target uses existing target-all six-view protocol.','',
  '|'+ '|'.join(fields)+'|','|'+'---|'*len(fields)]
 for row in rows:lines.append('|'+ '|'.join(str(row[k]) if not isinstance(row[k],float) else f'{row[k]:.5f}' for k in fields)+'|')
 (folder/'factorial.md').write_text('\n'.join(lines)+'\n')
 write(folder/'COMPLETE.json',dict(status='FACTORIAL_COMPLETE',groups=list(refs),windows=1860,scenes=312,matched_inputs_and_GT=True))
 print('\n'.join(lines),flush=True)

if __name__=='__main__':
 root=Path(sys.argv[1]);{'train_gate':train_gate,'eval_gate':eval_gate,'finish':finish}[sys.argv[2]](root)
