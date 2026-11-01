"""Final model only: direct shared write, full payload readback, immutable publish."""
from pathlib import Path
import os,shutil,traceback,uuid
import numpy as np
import torch
from scripts.scannet_data_snapshot import sha_file

def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return v

class FinalCheckpointWriter:
    @staticmethod
    def equal_payload(left,right):
        if torch.is_tensor(left):return torch.is_tensor(right) and left.shape==right.shape and left.dtype==right.dtype and torch.equal(left,right)
        if isinstance(left,np.ndarray):return isinstance(right,np.ndarray) and left.dtype==right.dtype and np.array_equal(left,right)
        if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(FinalCheckpointWriter.equal_payload(v,right[k]) for k,v in left.items())
        if isinstance(left,(tuple,list)):return type(left)==type(right) and len(left)==len(right) and all(FinalCheckpointWriter.equal_payload(a,b) for a,b in zip(left,right))
        return left==right

    def save_file(self,value,path):
        # The final name is published only after complete write and payload verification.
        tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.partial')
        payload=cpu(value)
        try:
            with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            expected=sha_file(tmp);size=tmp.stat().st_size
            loaded=torch.load(tmp,map_location='cpu',mmap=True,weights_only=False)
            if not self.equal_payload(payload,loaded):raise IOError('Shared checkpoint payload differs from live CPU state')
            del loaded
            assert not path.exists(), 'final model cannot be overwritten'
            os.replace(tmp,path)
            if sha_file(path)!=expected or path.stat().st_size!=size:raise IOError('Published checkpoint readback differs')
            return dict(name=path.name,sha256=expected,size=size)
        except Exception as error:
            free=shutil.disk_usage(path.parent).free
            raise RuntimeError('Shared checkpoint write failed: path='+str(path)+' free_bytes='+str(free)+' original_traceback='+traceback.format_exc()) from error
        finally:tmp.unlink(missing_ok=True)

def save_final(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def size(v):
        if torch.is_tensor(v):return v.numel()*v.element_size()
        if isinstance(v,dict):return sum(size(x) for x in v.values())
        if isinstance(v,(list,tuple)):return sum(size(x) for x in v)
        return 0
    if 'optimizer' in payload:raise ValueError('Final checkpoint must not contain full optimizer state')
    if shutil.disk_usage(path.parent).free<size(payload)+2*1024**3:raise OSError('Insufficient shared disk space for final model')
    return FinalCheckpointWriter().save_file(payload,path)
