"""Immutable, SHA verified rank-sharded ZeRO recovery; same run/world only."""
from pathlib import Path
import hashlib,json,os,random,shutil,tempfile,uuid
import numpy as np
import torch
import torch.distributed as dist
from scripts.scannet_data_snapshot import sha_file,verified_copy,local_filesystem


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s['cuda'] is not None:torch.cuda.set_rng_state(s['cuda'])
def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return v

def signature(model,adam):
    names={id(p):n for n,p in model.named_parameters()}
    return [[names[id(p)] for p in g['params']] for g in adam.optim.param_groups]

def task_state(model):return {n:v for n,v in model.state_dict().items() if not n.startswith(('frozen_omega.','sam_proposals.'))}

def all_ok(error):
    errors=[None]*dist.get_world_size();dist.all_gather_object(errors,error)
    if any(errors):raise RuntimeError('checkpoint operation failed: '+repr(errors))

class Recovery:
    def __init__(self,root,cfg,mode,model,adam):
        self.root=Path(root);self.cfg=cfg;self.mode=mode;self.model=model;self.adam=adam
        self.rank=dist.get_rank();self.world=dist.get_world_size();self.sig=signature(model,adam)
        self.identity=hashlib.sha256(json.dumps(dict(config=cfg,world=self.world,mode=mode),sort_keys=True).encode()).hexdigest()
        self.root.mkdir(parents=True,exist_ok=True)
        self.local=Path(tempfile.mkdtemp(prefix='qpanchor_checkpoint_',dir='/tmp'))
        assert local_filesystem(self.local), 'checkpoint staging must be node local'
    def save_file(self,value,path):
        tmp=self.local/(path.name+'.'+uuid.uuid4().hex)
        try:
            with tmp.open('wb') as f:torch.save(cpu(value),f);f.flush();os.fsync(f.fileno())
            # Complete deserialization before publication; immutable destination.
            loaded=torch.load(tmp,map_location='cpu',weights_only=False);del loaded
            record=dict(name=path.name,sha256=sha_file(tmp),size=tmp.stat().st_size)
            verified_copy(tmp,path,record['sha256'],record['size'])
            return record
        finally:tmp.unlink(missing_ok=True)
    def save(self,step,active):
        state=rng();identifier=[f'step{step:06d}_{uuid.uuid4().hex}' if self.rank==0 else None];dist.broadcast_object_list(identifier)
        folder=self.root/identifier[0]
        if self.rank==0:folder.mkdir()
        dist.barrier();records=[];error=None
        try:
            if self.rank==0:records.append(self.save_file(dict(model=task_state(self.model),step=step,identity=self.identity),folder/'model.pt'))
            records.append(self.save_file(dict(optimizer=self.adam.optim.state_dict(),signature=self.sig,rng=state,active=sorted(active),step=step,identity=self.identity,rank=self.rank),folder/f'rank{self.rank:02d}.pt'))
        except Exception as e:error=repr(e)
        all_ok(error)
        gathered=[None]*self.world;dist.all_gather_object(gathered,records)
        error=None
        if self.rank==0:
            try:
                receipt=dict(status='COMPLETE',step=step,identity=self.identity,world=self.world,files=sum(gathered,[]))
                assert len(receipt['files'])==self.world+1
                (folder/'COMPLETE.json').write_text(json.dumps(receipt,sort_keys=True)+'\n')
                assert json.loads((folder/'COMPLETE.json').read_text())==receipt
                complete=sorted(self.root.glob('step*/COMPLETE.json'),key=lambda p:json.loads(p.read_text())['step'])
                for old in complete[:-2]:
                    assert old.parent.parent==self.root and old.parent.name.startswith('step')
                    shutil.rmtree(old.parent)
            except Exception as e:error=repr(e)
        all_ok(error);dist.barrier();restore_rng(state)
        return folder
    def latest(self):
        result=[None];error=None
        if self.rank==0:
            try:
                candidates=[]
                for p in self.root.glob('step*/COMPLETE.json'):
                    data=json.loads(p.read_text())
                    assert data['identity']==self.identity and data['world']==self.world
                    candidates.append((data['step'],str(p.parent)))
                if candidates:result[0]=max(candidates)[1]
            except Exception as e:error=repr(e)
        all_ok(error);dist.broadcast_object_list(result)
        return Path(result[0]) if result[0] else None
    def load(self,folder):
        error=None;model_state=rank_state=None
        try:
            receipt=json.loads((folder/'COMPLETE.json').read_text())
            assert receipt['identity']==self.identity and receipt['world']==self.world
            selected={r['name']:r for r in receipt['files']}
            paths=[]
            for name in ('model.pt',f'rank{self.rank:02d}.pt'):
                rec=selected[name];dest=self.local/(folder.name+'_'+name)
                verified_copy(folder/name,dest,rec['sha256'],rec['size']);paths.append(dest)
            model_state=torch.load(paths[0],map_location='cpu',mmap=True,weights_only=False)
            rank_state=torch.load(paths[1],map_location='cpu',weights_only=False)
            assert model_state['identity']==rank_state['identity']==self.identity
            assert model_state['step']==rank_state['step']==receipt['step']
            assert rank_state['rank']==self.rank and rank_state['signature']==self.sig
            current=task_state(self.model);assert set(current)==set(model_state['model'])
            assert all(current[n].shape==v.shape and current[n].dtype==v.dtype for n,v in model_state['model'].items())
        except Exception as e:error=repr(e)
        all_ok(error)
        with torch.no_grad():
            for n,v in model_state['model'].items():
                parameter=dict(self.model.named_parameters()).get(n)
                if parameter is not None and not parameter.requires_grad:
                    assert torch.equal(current[n].detach().cpu(),v), 'frozen checkpoint tensor differs: '+n
                else:current[n].copy_(v)
        self.adam.optim.load_state_dict(rank_state['optimizer'])
        # ZeRO owns non-state group metadata; LR is constant and both must agree.
        assert all(g['lr']==1e-5 for g in self.adam.optim.param_groups)
        restore_rng(rank_state['rng']);step=rank_state['step'];active=set(rank_state['active'])
        del model_state,rank_state
        for p in paths:p.unlink()
        dist.barrier()
        return step,active
    def close(self):shutil.rmtree(self.local)
