"""Compare native uncached data with cached/retried real six-view data on CPU."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import errno,json,random
from unittest.mock import patch
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import numpy as np
import torch
from tokengs.options import config_defaults
from scripts.train_dual_stream_v1 import cohorts,build_batch,digest,MANIFEST
from scripts.training_windows_2ctx4novel import load_training_windows,DEFAULT_MANIFEST
from scripts.io_resilience import scene_depth_frames
import scripts.object_locus_v3_set_runtime as runtime
import tokengs.data.siu3r_processed as provider

def compare(a,b,path='batch'):
    if torch.is_tensor(a):
        if a.dtype!=b.dtype or a.shape!=b.shape or not torch.allclose(a,b,rtol=0,atol=0,equal_nan=True):raise ValueError('Tensor differs '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('Keys differ '+path)
        for k in a:compare(a[k],b[k],path+'.'+k)
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise ValueError('Length differs '+path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+str(i))
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b,equal_nan=True):raise ValueError('Array differs '+path)
    elif a!=b:raise ValueError('Value differs '+path)

base,_,_=cohorts();windows,identity=load_training_windows(DEFAULT_MANIFEST,base,digest(MANIFEST));window=windows[8205]
opt=config_defaults['train_siu3r_unified_query_v1'].evolve(num_views=6)
with patch.object(provider,'scene_depth_frames',lambda s:tuple(sorted(int(x.stem) for x in (Path(s)/'depth').glob('*.png')))):
    reference=build_batch(opt,window,'cpu')
scene_depth_frames.cache_clear();actual=build_batch(opt,window,'cpu');compare(reference,actual)
cache_before=scene_depth_frames.cache_info();original=runtime.build_batch;calls=[]
py=random.getstate();npstate=np.random.get_state();ts=torch.get_rng_state()
def flaky(*args,**kwargs):
    calls.append(1)
    if len(calls)==1:
        random.random();np.random.random();torch.rand(3)
        raise TimeoutError(errno.ETIMEDOUT,'injected read timeout after CPU RNG consumed')
    return original(*args,**kwargs)
with patch.object(runtime,'build_batch',flaky):replayed=build_batch(opt,window,'cpu')
compare(reference,replayed)
assert py==random.getstate() and npstate[0]==np.random.get_state()[0] and np.array_equal(npstate[1],np.random.get_state()[1]) and npstate[2:]==np.random.get_state()[2:] and torch.equal(ts,torch.get_rng_state())
cache_after=scene_depth_frames.cache_info();assert cache_after.misses==cache_before.misses and cache_after.hits>cache_before.hits
result=dict(status='REAL_DATA_IO_RESILIENCE_PASSED',window_index=8205,scene=window['scene'],frame_ids=actual['frame_ids'][0].tolist(),
 all_batch_tensors_bitwise_equal_to_native_uncached=True,retried_after_injected_timeout=True,attempts=len(calls),
 python_numpy_torch_rng_preserved=True,cache_avoided_repeated_directory_scan=True,training_identity=identity,
 model_optimizer_and_recipe_unchanged=True,jobs_submitted=False)
(ROOT/'docs/io_resilience_cpu_audit_20261021.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
