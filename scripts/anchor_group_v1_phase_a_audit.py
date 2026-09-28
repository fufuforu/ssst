#!/usr/bin/env python3
"""Run Phase-A real-window target audit, reconstruction parity and legacy contracts.

This is inference/contract auditing only. It does not construct an optimizer.
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import torch
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(REPO)
from tokengs.options import config_defaults
from tokengs.models import model_registry
from tokengs.models.input_types import ModelInput, ModelInputDecoder, split_data
from tokengs.models.anchor_group_loss import build_anchor_targets
from scripts.instance_state_generalization import _batch_for, TRAIN_ROOT, VAL_ROOT, move
from scripts.run_instance_state_v1 import PRETRAINED, BASE_PRESET

ROOT=REPO/"group_plus/anchor_group_v1"
MANIFEST=REPO/"group_plus/instance_state_v1_generalization/train128_windows1024.json"

def state_from_checkpoint(path):
    obj=torch.load(path,map_location="cpu",weights_only=False)
    return obj.get("model",obj)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--device",default="cpu"); ap.add_argument("--checkpoint",default=str(PRETRAINED)); args=ap.parse_args()
    device=torch.device(args.device); ROOT.mkdir(parents=True,exist_ok=True)
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("Phase-A parity audit requires a working CUDA device and gsplat renderer")
    manifest=json.loads(MANIFEST.read_text()); window=manifest["windows"][0]
    opt=config_defaults["train_siu3r_anchor_group_v1"].evolve(dataset_kwargs={"data_root":"/space/mawb/SIU3R/data/scannet"},batch_size=1,num_workers=0,seed=42,num_input_views=2,num_views=4)
    batch=_batch_for(opt,window,device); source=state_from_checkpoint(args.checkpoint)
    # Target audit uses a real beta=0 anchor forward from the pretrained weights.
    model=model_registry[opt.model_type](opt).to(device)
    missing,unexpected=model.load_state_dict(source,strict=False)
    allowed=[k for k in missing if k.startswith("anchor_group.")]
    if unexpected or len(allowed)!=len(missing): raise RuntimeError(f"checkpoint transfer mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    model.eval(); mi,_=split_data(batch,opt); dec=ModelInputDecoder(cam_view=batch["cam_view_all"],intrinsics=batch["intrinsics_all"])
    with torch.no_grad():
        ag_runs=[model.forward_anchor_group(ModelInput(mi.encoder,dec),render_decoder_input=dec,coupled=False,step=0) for _ in range(2)]
    states=ag_runs[0]["states"]; final=states[-1]; gs_new=ag_runs[0]["gaussians"]
    targets=build_anchor_targets(final["mu"],batch["semantic_label_all"],batch["instance_label_all"],batch["cam_view_all"],batch["intrinsics_all"])
    kinds=targets["anchor_kind"][0]; counts=targets["Y_anchor"][0].sum(-1).long().tolist(); supported=sum(x>0 for x in counts); K=len(targets["gt_classes"][0])
    sorted_counts=sorted(counts)
    def quantile(q): return sorted_counts[min(len(sorted_counts)-1,round(q*(len(sorted_counts)-1)))] if sorted_counts else 0
    audit={"manifest":str(MANIFEST),"window_index":0,"scene":window["scene"],"context_frames":window["context"],"total_anchors":int(kinds.numel()),"thing_anchors":int((kinds==0).sum()),"wall_anchors":int((kinds==1).sum()),"floor_anchors":int((kinds==2).sum()),"ignore_anchors":int((kinds==-1).sum()),"gt_thing_count":K,"gt_with_anchor_support":int(supported),"gt_without_anchor_support":int(K-supported),"per_gt_anchor_counts":counts,"min":min(counts) if counts else 0,"median":quantile(.5),"p90":quantile(.9),"max":max(counts) if counts else 0}
    (ROOT/"real_batch_anchor_target_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    # Baseline has identical checkpoint and same reconstruction implementation. Save current outputs then instantiate baseline.
    gs_new_cpu=gs_new.detach().float().cpu()
    rgb_ag=[x["render"]["images_pred"].detach().float().cpu() for x in ag_runs]
    ag_gs_repeat=float((ag_runs[0]["gaussians"].detach().float().cpu()-ag_runs[1]["gaussians"].detach().float().cpu()).abs().max())
    del states,final,gs_new,ag_runs,model; gc.collect(); torch.cuda.empty_cache()
    base=model_registry["siu3r_locusgs_recon"](opt).to(device); miss,unexp=base.load_state_dict(source,strict=False)
    if miss or unexp: raise RuntimeError(f"baseline checkpoint not strict: missing={miss[:8]} unexpected={unexp[:8]}")
    base.eval(); mi,_=split_data(batch,opt); dec=ModelInputDecoder(cam_view=batch["cam_view_all"],intrinsics=batch["intrinsics_all"])
    with torch.no_grad(): old_runs=[base.forward_reconstruction_only(ModelInput(mi.encoder,dec),render_decoder_input=dec) for _ in range(2)]
    gs_old=old_runs[0]["gaussians"]
    gs_old=gs_old.detach().float().cpu()
    gd=float((gs_new_cpu-gs_old).abs().max())
    rgb_base=[x["render"]["images_pred"].detach().float().cpu() for x in old_runs]
    gt=batch["images_all"].detach().float().cpu()
    cross=(rgb_base[0]-rgb_ag[0]).abs(); base_repeat=(rgb_base[0]-rgb_base[1]).abs(); ag_repeat=(rgb_ag[0]-rgb_ag[1]).abs()
    def psnr(x): return float((-10*torch.log10(((x-gt)**2).mean(dim=(-1,-2,-3)))).mean())
    pb,pa=psnr(rgb_base[0]),psnr(rgb_ag[0]); pd=abs(pb-pa)
    gd_repeat=float((old_runs[0]["gaussians"].detach().float().cpu()-old_runs[1]["gaussians"].detach().float().cpu()).abs().max())
    parity={"gpu":torch.cuda.get_device_name(device),"cuda_available":torch.cuda.is_available(),"cuda_visible_devices":__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),"torch_version":torch.__version__,"torch_cuda_version":torch.version.cuda,"gsplat_forward":"pass","checkpoint":str(args.checkpoint),"scene":window["scene"],"context_frames":window["context"],"coupled":False,"beta":0.0,"gaussian_max_abs_diff":gd,"gaussian_repeat_baseline_max_abs_diff":gd_repeat,"rgb_max_abs_diff":float(cross.max()),"rgb_mean_abs_diff":float(cross.mean()),"baseline_psnr":pb,"anchor_group_psnr":pa,"psnr_abs_diff":pd,"baseline_repeat_rgb_diff":float(base_repeat.max()),"anchor_group_repeat_rgb_diff":float(ag_repeat.max()),"anchor_group_repeat_gaussian_max_abs_diff":ag_gs_repeat,"status":"pass" if gd<=1e-6 and cross.max()<=1e-6 and pd<=1e-5 else "fail","blocker":bool(gd>1e-5 or cross.max()>1e-5 or pd>1e-5),"pass_target":bool(gd<=1e-6 and cross.max()<=1e-6 and pd<=1e-5)}
    (ROOT/"reconstruction_parity.json").write_text(json.dumps(parity,indent=2)+"\n")
    print(json.dumps({"real_batch_anchor_target_audit":audit,"reconstruction_parity":parity},indent=2))
    del old_runs,base,mi,dec,batch; gc.collect(); torch.cuda.empty_cache()
    return int(not parity["pass_target"])
if __name__=="__main__": raise SystemExit(main())
