#!/usr/bin/env python3
"""GPU inference/loss regression for historical S0 and S1 presets; no optimizer."""
from __future__ import annotations
import gc,json,sys
from pathlib import Path
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from tokengs.models import model_registry
from scripts.run_instance_state_v1 import PRETRAINED,PRESET_C,SEED,build_options,transfer_reconstruction_weights
from scripts.instance_state_s1_local3d import S1_PRESET,K_LOCAL
from scripts.instance_state_generalization import _batch_for

OUT=REPO/"group_plus/anchor_group_v1/legacy_forward_regression.json"

def finite(x): return bool(torch.isfinite(x).all())

def main():
    if not torch.cuda.is_available(): raise RuntimeError("legacy full-forward regression requires CUDA")
    device=torch.device("cuda"); man=json.loads((REPO/"group_plus/instance_state_v1_generalization/train128_windows1024.json").read_text());window=man["windows"][0]
    source_obj=torch.load(PRETRAINED,map_location="cpu",weights_only=False);source=source_obj["model"]
    records={"gpu":torch.cuda.get_device_name(0),"cuda_available":True,"torch":torch.__version__,"torch_cuda":torch.version.cuda,"scene":window["scene"],"window_index":0,"models":{}}
    for name,preset,local,k in (("S0",PRESET_C,False,None),("S1",S1_PRESET,True,K_LOCAL)):
        torch.manual_seed(SEED);opt=build_options(preset,num_views=4);model=model_registry[opt.model_type](opt)
        transfer_reconstruction_weights(model,source,opt);model=model.to(device).eval()
        batch=_batch_for(opt,window,device)
        with torch.no_grad(): output,metrics=model.step_loss(batch,step=0,phase="eval",coupled=False)
        pred=output["prediction"]
        finite_tensors={key:finite(pred[key]) for key in ("gaussians","region_mass","semantic_scores")}
        finite_loss=finite(metrics["loss"])
        init=model.anchor_decoder.last_state_init
        init_ok=(bool(opt.instance_state_local3d)==local and (init.get("fps_index") is not None))
        if local:
            init_ok=init_ok and "neighbour_index" in init and tuple(init["neighbour_index"].shape)==(1,100,8)
        else:
            init_ok=init_ok and "neighbour_index" not in init and not bool(opt.instance_state_local3d)
        records["models"][name]={"architecture_name":model.architecture_name,"instance_state_local3d":bool(opt.instance_state_local3d),"local3d_k":int(getattr(opt,"instance_state_local_k",8)) if local else None,"forward":"pass" if all(finite_tensors.values()) else "fail","finite":{"gaussians":finite_tensors["gaussians"],"region_mass":finite_tensors["region_mass"],"semantic_scores":finite_tensors["semantic_scores"],"loss":finite_loss},"loss":"pass" if finite_loss else "fail","initialization_path":"local8" if local and init_ok else "single-anchor" if init_ok else "fail","loss_value":float(metrics["loss"])}
        del pred,output,metrics,batch,model;gc.collect();torch.cuda.empty_cache()
    records["status"]="pass" if all(r["forward"]=="pass" and r["loss"]=="pass" and r["initialization_path"]!="fail" for r in records["models"].values()) else "fail"
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(records,indent=2)+"\n");print(json.dumps(records,indent=2));return int(records["status"]!="pass")
if __name__=="__main__":raise SystemExit(main())
