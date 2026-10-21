"""One real RTX3090 batch/optimizer smoke for V3-Set."""
from __future__ import annotations
import json,os,tempfile,zipfile
from pathlib import Path
import sys
import torch
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.object_locus_v3_set_runtime import (build_model,build_optimizer,build_batch,train_one_step,
    write_json,REPORTS_DEFAULT,SEED,seed_everything)
from scripts.train_object_locus_v3_set import _gpu_assert
from scripts.eval_object_locus_v3_set import evaluate_windows

WINDOW={'scene':'scene0016_00','context':[1506,1517],'novel':[1509,1516]}

def main():
    gpu=_gpu_assert();seed_everything(42);model,opt,transfer=build_model('cuda');optimizer,opta=build_optimizer(model)
    batch=build_batch(opt,WINDOW,'cuda')
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    smoke_root=REPORTS_DEFAULT/'smoke_validation';out,metrics=train_one_step(model,optimizer,batch,1,understanding_weight_value=1.,lr_values=(1e-4,1e-6),failure_capture_dir=smoke_root/'failure',failure_context={'window':WINDOW,'smoke':True})
    names=dict(model.named_parameters());required=('object_locus_v3_set.class_head.weight','object_locus_v3_set.cls_fuse.weight','object_locus_v3_set.mask_q_mlp.2.weight','object_locus_v3_set.child_mlp.2.weight','anchor_decoder.mu','activation_head.deconv.weight')
    grads={n:{'present':names[n].grad is not None,'finite':bool(torch.isfinite(names[n].grad).all()) if names[n].grad is not None else False,'norm':float(names[n].grad.float().norm()) if names[n].grad is not None else 0.} for n in required}
    if not all(x['present'] and x['finite'] and x['norm']>0 for x in grads.values()):raise RuntimeError(f'V3-Set smoke required gradients missing/zero: {grads}')
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('nonfinite parameter after V3-Set smoke optimizer step')
    reports=smoke_root/'current';result,_,_=evaluate_windows(model,opt,[WINDOW],1,'smoke',reports,'cuda',build_batch,official=True,panels=True)
    torch.cuda.synchronize()
    audit={'gpu':gpu,'node':os.uname().nodename,'window':WINDOW,'losses':{k:(float(v.detach()) if torch.is_tensor(v) else float(v)) for k,v in metrics.items() if (torch.is_tensor(v) and v.ndim==0) or isinstance(v,(int,float))},
      'gradient_report':grads,'optimizer':opta,'transfer':transfer,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
      'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'local_eval':result['local']['context'],'official_single_pair_status':'completed; AP may be undefined by evaluator'}
    reports.mkdir(parents=True,exist_ok=True);write_json(smoke_root/'smoke_audit.json',audit)
    # Exercise heterogeneous JSON/CSV/image packaging before the formal run.
    with tempfile.TemporaryDirectory(prefix='v3set_bundle_') as td:
        root=Path(td);(root/'x.csv').write_text('step,name,value\n1,a,0.2\n');(root/'x.json').write_text('{"finite":true}\n')
        panel=next((reports/'qualitative').rglob('*.png'));(root/'panel.png').write_bytes(panel.read_bytes())
        z=root/'smoke.zip'
        with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as archive:
            for f in root.iterdir():
                if f!=z:archive.write(f,f.name)
        with zipfile.ZipFile(z) as archive:
            if archive.testzip():raise RuntimeError('smoke package integrity failed')
            if not {'x.csv','x.json','panel.png'}.issubset(set(archive.namelist())):raise RuntimeError('smoke package lacks heterogeneous artifacts')
    print(json.dumps(audit,allow_nan=False),flush=True)

if __name__=='__main__':main()
