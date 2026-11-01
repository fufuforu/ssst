"""Read-only SAM acceleration: invalid cache entries fall back to the same frozen SAM.

Never change, delete, or publish shared cache entries. Errors from online SAM are
not caught. Tiny bounded per-process CPU LRU avoids repeated broken-path reads.
"""
from pathlib import Path
from collections import OrderedDict
import os,stat,json,pickle,errno,random
import numpy as np
import torch
from scripts.dual_stream_frozen_cache import rgb_key


def decode(blob,height,width):
    if not isinstance(blob,list):raise ValueError('cache must be a proposal list')
    masks=[]
    for row in blob:
        if not isinstance(row,dict):raise ValueError('cache row must be a mapping')
        if row['height']!=height or row['width']!=width:raise ValueError('cache dimensions differ from RGB')
        bits=row['bits']
        if not torch.is_tensor(bits) or bits.dtype!=torch.uint8 or bits.device.type!='cpu' or bits.ndim!=1 or bits.numel()!=(height*width+7)//8:
            raise ValueError('invalid packed mask')
        mask=np.unpackbits(bits.numpy(),count=height*width).reshape(height,width).astype(bool)
        area=row['area']
        if isinstance(area,bool) or not isinstance(area,(int,np.integer)) or int(mask.sum())!=int(area):raise ValueError('mask area mismatch')
        masks.append(dict(mask=mask,area=int(area)))
    return masks


def read_masks(path,height,width):
    # O_NONBLOCK also avoids blocking if the shared filesystem exposes a FIFO.
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):raise OSError(errno.EINVAL,'SAM cache is not a regular file',str(path))
        with os.fdopen(fd,'rb') as stream:
            fd=None
            blob=torch.load(stream,map_location='cpu',weights_only=True)
        return decode(blob,height,width)
    finally:
        if fd is not None:os.close(fd)


def same_masks(left,right):
    return len(left)==len(right) and all(a['area']==b['area'] and np.array_equal(a['mask'],b['mask']) for a,b in zip(left,right))


def install_caches(model,root):
    root=Path(root);sam=model.sam_proposals.forward
    counts=dict(sam_hits=0,sam_computed=0,omega_hits=0,calibration_hits=0,sam_cache_errors=0,sam_local_hits=0,sam_parity_verified=0,sam_online_determinism_verified=0)
    lru=OrderedDict();smoke=bool(getattr(model,'_anchor_smoke',False));parity_done=False
    def compute(view):
        # Frozen SAM is deterministic. Preserve Python/NumPy/Torch RNGs so the
        # acceleration hit/miss/error cannot alter subsequent training randomness.
        py=random.getstate();np_state=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state(view.device) if view.is_cuda else None
        try:return sam(view[None,None])[0][0]
        finally:
            random.setstate(py);np.random.set_state(np_state);torch.set_rng_state(cpu)
            if cuda is not None:torch.cuda.set_rng_state(cuda,view.device)
    def frozen_sam(rgb):
        nonlocal parity_done
        result=[]
        for scene in rgb:
            views=[]
            for view in scene:
                key=rgb_key(view);path=root/'esam'/key[:2]/(key+'.pt')
                if key in lru:
                    lru.move_to_end(key);masks=lru[key];counts['sam_local_hits']+=1
                else:
                    try:masks=read_masks(path,*view.shape[-2:])
                    except (OSError,EOFError,RuntimeError,ValueError,TypeError,KeyError,AttributeError,IndexError,pickle.UnpicklingError) as error:
                        if isinstance(error,torch.OutOfMemoryError):raise
                        missing=isinstance(error,FileNotFoundError)
                        if not missing:
                            counts['sam_cache_errors']+=1
                            print(json.dumps(dict(event='SAM_CACHE_FALLBACK',rank=os.environ.get('RANK'),path=str(path),error=repr(error),action='same_frozen_SAM_online_no_shared_write')),flush=True)
                        masks=compute(view);counts['sam_computed']+=1
                        if smoke and counts['sam_online_determinism_verified']==0:
                            if not same_masks(masks,compute(view)):raise ValueError('Frozen SAM online repeat is not deterministic')
                            counts['sam_online_determinism_verified']=1
                    else:
                        counts['sam_hits']+=1
                        if smoke and not parity_done:
                            online=compute(view)
                            if not same_masks(masks,online):raise ValueError('Frozen SAM cache/online mask parity failed')
                            counts['sam_parity_verified']+=1;parity_done=True
                            print(json.dumps(dict(event='SAM_CACHE_ONLINE_PARITY_PASSED',rank=os.environ.get('RANK'),key=key)),flush=True)
                    lru[key]=masks
                    if len(lru)>8:lru.popitem(last=False)
                views.append(masks)
            result.append(views)
        return result
    model.sam_proposals.forward=frozen_sam
    return counts
