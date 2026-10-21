"""Original legality, bounded local search; immutable manifest determines all counts."""
from pathlib import Path
import ast,collections,hashlib,json,subprocess,random
import numpy as np
BASE='7300b6ff6ae963ea7228ae7bc0c7d0e0446eefa3'
REPORT=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
OLD=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_v1_8gpu')
TRAIN_ROOT=Path('/space/mawb/SIU3R/data/scannet/train')
SPLIT=Path('/space/mawb/ssst/group_plus/implementation_audit_v1/full_split.json')
ORIGINAL=Path('/space/mawb/ssst/group_plus/instance_state_v1_generalization/train128_windows1024.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def order(epoch,n):
 p=np.random.default_rng(42+epoch).permutation(n);padding=(-n)%8;return np.concatenate((p,p[:padding]))
def identity(w):return (tuple(map(int,w['context'])),tuple(map(int,w.get('target',w['context']+w['novel']))))
def instrumented_sampler():
 from scripts.object_locus_panoptic_v1_runtime import build_options
 from tokengs.data.siu3r_processed import SIU3RProcessedProvider
 import torch
 source=subprocess.check_output(['git','show',BASE+':scripts/instance_state_generalization.py'],text=True)
 fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='sample_scene_windows')
 # Only diagnostics, duplicate accounting, inherited candidates and exception
 # transparency change. Provider and every original legality expression stay intact.
 fn.args.defaults[-1]=ast.Constant(128)
 for node in ast.walk(fn):
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Tuple) for t in node.targets) and isinstance(node.value,ast.Tuple):
   if any(isinstance(x,ast.List) for x in node.value.elts):node.value.elts[0]=ast.Call(func=ast.Name(id='list',ctx=ast.Load()),args=[ast.Name(id='existing',ctx=ast.Load())],keywords=[])
  if isinstance(node,ast.ExceptHandler):
   node.name='error';node.body=ast.parse("rejects[type(error).__name__+': '+str(error)] += 1\nif isinstance(error, (FileNotFoundError,OSError)) or (isinstance(error,RuntimeError) and str(error).startswith('no official pair found for ')):\n continue\nraise").body
  if isinstance(node,ast.If) and ast.unparse(node.test)=='len(good) < 2':node.body.insert(0,ast.parse("rejects['fewer_than_2_context_instances_area_ge100'] += 1").body[0])
  if isinstance(node,ast.While):
   # Filter repeated frame identities before append without changing legality.
   for j,statement in enumerate(node.body):
    if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and ast.unparse(statement.value.func)=='windows.append':
     node.body[j:j]=ast.parse("if any(tuple(w['context'])==tuple(pair['context_frame_ids']) and tuple(w['novel'])==tuple(pair['novel_frame_ids']) for w in windows):\n rejects['duplicate_candidate'] += 1\n continue").body;break
 ast.fix_missing_locations(fn)
 env=dict(SIU3RProcessedProvider=SIU3RProcessedProvider,TRAIN_ROOT=TRAIN_ROOT,torch=torch,existing=[],rejects=collections.Counter())
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original legality instrumented>','exec'),env)
 def sample(scene,seed,need,tries,existing=()):
  state=(random.getstate(),np.random.get_state(),torch.get_rng_state());env['existing']=list(existing);env['rejects']=collections.Counter()
  try:windows,attempts=env['sample_scene_windows'](build_options(),scene,seed,need,tries);return windows,attempts,dict(env['rejects'])
  finally:random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
 return sample

def diagnose_scene(sample):
 from PIL import Image
 import torch
 root=TRAIN_ROOT/'scene0044_00';frames=sorted(int(p.stem) for p in (root/'depth').glob('*.png'));bad=[]
 for f in frames:
  try:
   with Image.open(root/'color'/f'{f}.jpg') as im:im.verify()
   pose=np.loadtxt(root/'extrinsic'/f'{f}.txt');assert pose.shape==(4,4)
  except Exception as e:bad.append(dict(frame=f,error=type(e).__name__+': '+str(e)))
 intrinsic=np.loadtxt(root/'intrinsic.txt');iou=torch.load(root/'iou.pt',map_location='cpu',weights_only=True)
 scene_dirs=sorted(p.name for p in TRAIN_ROOT.iterdir() if p.is_dir() and p.name.startswith('scene'))
 candidates,attempts,rejections=sample(root.name,42000+scene_dirs.index(root.name),8,128)
 write(REPORT/'scene0044_data_check.json',dict(available_frames=len(frames),rgb_camera_files_checked=len(frames),unreadable=bad,intrinsic_shape=list(intrinsic.shape),iou_shape=list(iou.shape),original_budget_attempts=attempts,original_legal_candidates=len(candidates),rejections=rejections,conclusion='Fixed128-attempt original random search did not find legal windows; not proof no legal windows exist.'))
 return candidates,attempts,rejections

def main():
 import torch
 torch.set_num_threads(4);REPORT.mkdir(parents=True,exist_ok=True)
 if (REPORT/'manifest.json').exists():raise RuntimeError('fixed manifest already exists; do not regenerate')
 old=json.loads((OLD/'data_manifest.json').read_text());split=json.loads(SPLIT.read_text());scenes=sorted(split['train_scenes']);assert len(scenes)==len(set(scenes))==1201
 assert not set(scenes)&{w['scene'] for name in ('dev8','val32') for w in old[name]}
 sample=instrumented_sampler();cache_path=REPORT/'candidate_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
 if not (REPORT/'scene0044_data_check.json').exists():
  candidates,a,r=diagnose_scene(sample);cache['scene0044_00']=dict(windows=candidates,initial_attempts=a,initial_rejections=r,initial_complete=True);write(cache_path,cache)
 # Recover already generated successful windows from the original immutable
 # candidate asset; earlier failed planner kept them only in process memory.
 original=json.loads(ORIGINAL.read_text());source_candidates=collections.defaultdict(list)
 for w in original['windows']:source_candidates[w['scene']].append(w)
 scene_dirs=sorted(p.name for p in TRAIN_ROOT.iterdir() if p.is_dir() and p.name.startswith('scene'));windows=[];coverage=[]
 for i,scene in enumerate(scenes):
  item=cache.get(scene)
  if item is None:
   if scene in source_candidates:
    found=source_candidates[scene];item=dict(windows=found,initial_attempts=max(w['attempt'] for w in found),initial_complete=True,source=str(ORIGINAL),source_sha256=sha(ORIGINAL),initial_rejections={'not_retained_in_original_asset':'MISSING'},recovered_without_resampling=True)
   else:
    found,a,r=sample(scene,42000+scene_dirs.index(scene),8,128);item=dict(windows=found,initial_attempts=a,initial_rejections=r,initial_complete=True)
   cache[scene]=item;write(cache_path,cache)
  unique={identity(w):w for w in item['windows']};found=list(unique.values())
  if len(found)<7 and not item.get('extra_complete'):
   found,a,r=sample(scene,42+i,7,4096,found);item.update(windows=found,extra_attempts=a,extra_rejections=r,extra_seed=42+i,extra_complete=True);write(cache_path,cache)
  found=item['windows'];found=list({identity(w):w for w in found}.values());n=len(found)
  indices=[k*(n-1)//6 for k in range(7)] if n>=7 else list(range(n))
  chosen=[found[j] for j in indices];assert len(chosen)==min(n,7) and len({identity(w) for w in chosen})==len(chosen)
  for j,w in zip(indices,chosen):windows.append(dict(w,window_index=int(w.get('index',j)),target=w['context']+w['novel'],camera_source='SIU3R processed ScanNet GT pose/intrinsics; unchanged provider'))
  coverage.append(dict(scene=scene,candidates=n,selected=len(chosen),selected_indices=indices,initial_attempts=item['initial_attempts'],extra_attempts=item.get('extra_attempts',0),extra_seed=item.get('extra_seed'),rejections=[item.get('initial_rejections'),item.get('extra_rejections')],status='included' if n else 'fixed search budget did not find legal windows'))
  if (i+1)%25==0:print(f'scenes {i+1}/1201; windows {len(windows)}',flush=True)
 n=len(windows);u=(n+7)//8;p=8*u-n;s=len({w['scene'] for w in windows})
 manifest=dict(name='object_locus_panoptic_full1201_8gpu',science_sha=BASE,candidate_train_scenes=scenes,actual_train_scenes=sorted({w['scene'] for w in windows}),scene_coverage=coverage,zero_candidate_scenes=[x['scene'] for x in coverage if not x['selected']],windows=windows,S=s,N=n,U=u,P=p,split_source=str(SPLIT),split_sha256=sha(SPLIT),candidate_rule='Original legality; original candidates retained; at most4096 extra attempts with local seed42+sorted1201index; distinct1..7 or zero excluded. No proof of global absence.',monitor_splits={name:old[name] for name in ('expanded_train_probe32','train_all56','same_scene_holdout8','same_scene_holdout16','dev8','val32')},full_validation_source='/space/mawb/SIU3R/data/scannet/val_pair.json')
 path=REPORT/'manifest.json';write(path,manifest);orders=[order(e,n).tolist() for e in range(8)];counts=np.bincount(np.concatenate(orders),minlength=n)
 plan=dict(manifest_sha256=sha(path),epochs=8,N=n,S=s,U=u,P=p,updates_per_epoch=u,exposures_per_epoch=8*u,total_updates=8*u,total_exposures=64*u,padding_window_ids=[x[n:] for x in orders],orders=orders,actual_expected_counts=counts.tolist());write(REPORT/'training_plan.json',plan)
 provenance=json.loads((OLD/'preflight_audit.json').read_text())['weights'];write(REPORT/'weights_provenance.json',dict(source=str(OLD/'preflight_audit.json'),note='Reuse exact previously verified128 run preflight paths/SHA; no weights_provenance filename in old run',weights=provenance))
 write(REPORT/'asset_hashes.json',{name:sha(REPORT/name) for name in ('manifest.json','training_plan.json','weights_provenance.json')});print('FIXED_MANIFEST',s,n,u,p,flush=True)
if __name__=='__main__':main()
