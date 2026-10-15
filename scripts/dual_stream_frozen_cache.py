"""Exact, RGB-keyed frozen inference caches; no learned features/proposal matches.

Disk writes use unique temporary paths and atomic replacement across ranks.
Namespace includes frozen identities plus cache algorithm version.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import torch
import numpy as np


def cpu_tree(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x,dict): return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list): return [cpu_tree(v) for v in x]
    if isinstance(x,tuple): return tuple(cpu_tree(v) for v in x)
    return x

def device_tree(x,device):
    if torch.is_tensor(x): return x.to(device)
    if isinstance(x,dict): return {k:device_tree(v,device) for k,v in x.items()}
    if isinstance(x,list): return [device_tree(v,device) for v in x]
    if isinstance(x,tuple): return tuple(device_tree(v,device) for v in x)
    return x

def rgb_key(x):
    x=x.detach().float().cpu().contiguous()
    return hashlib.sha256(str(tuple(x.shape)).encode()+x.numpy().tobytes()).hexdigest()

def atomic_torch_save(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    torch.save(payload,temp);temp.replace(path)


def install_frozen_caches(model,root):
    identity=dict(version='rgb_exact_cache_v1',omega=model.frozen_omega.source_identity,sam=model.sam_proposals.source_identity,
        geometry='first_context_depth_median_.25_grid256_pixel_center_.5')
    namespace=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    root=Path(root)/namespace;root.mkdir(parents=True,exist_ok=True)
    omega_forward=model.frozen_omega.forward;sam_forward=model.sam_proposals.forward
    calibrate=model.calibrate_targets
    omega_lru=OrderedDict()
    counts=dict(omega_hits=0,sam_hits=0,calibration_hits=0)

    def omega_cached(rgb):
        if rgb.shape[1]!=2: return omega_forward(rgb)
        key=rgb_key(rgb)
        if key not in omega_lru:
            value=omega_forward(rgb)
            omega_lru[key]=cpu_tree(value)
            if len(omega_lru)>8: omega_lru.popitem(last=False)
            return value
        counts['omega_hits']+=1
        omega_lru.move_to_end(key)
        return device_tree(omega_lru[key],rgb.device)

    def sam_cached(rgb):
        # Cache each view, so repeated view in a different pair is reusable.
        result=[]
        for scene in rgb:
            views=[]
            for view in scene:
                key=rgb_key(view);path=root/'sam'/key[:2]/(key+'.pt')
                if path.exists():
                    blob=torch.load(path,map_location='cpu',weights_only=True)
                    proposals=[dict(mask=np.unpackbits(row['bits'].numpy(),count=row['height']*row['width']).reshape(row['height'],row['width']).astype(bool),score=row['score']) for row in blob]
                    counts['sam_hits']+=1
                else:
                    proposals=sam_forward(view[None,None])[0][0]
                    blob=[dict(bits=torch.from_numpy(np.packbits(p['mask'])),height=p['mask'].shape[0],width=p['mask'].shape[1],score=float(p['score'])) for p in proposals]
                    atomic_torch_save(blob,path)
                views.append(proposals)
            result.append(views)
        return result

    def calibration_cached(rgb,generated):
        # Include generated point map: cached alignment cannot silently refer to a different geometry.
        key=hashlib.sha256((rgb_key(rgb)+rgb_key(generated['predicted_points'])).encode()).hexdigest()
        path=root/'calibration'/key[:2]/(key+'.pt')
        if path.exists():
            counts['calibration_hits']+=1
            return device_tree(torch.load(path,map_location='cpu',weights_only=True),rgb.device)
        value=calibrate(rgb,generated)
        atomic_torch_save(cpu_tree(value),path)
        return value

    model.frozen_omega.forward=omega_cached
    model.sam_proposals.forward=sam_cached
    model.calibrate_targets=calibration_cached
    return dict(root=str(root),identity=identity,counts=counts)
