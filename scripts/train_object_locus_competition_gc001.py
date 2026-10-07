"""Run the preregistered competition smoke and fresh comp_gc001 training."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.object_locus_competition_gc001_runtime import *
from scripts.check_object_locus_competition_gc001 import run_cpu_contracts


ARM = "comp_gc001"
ALPHA = 0.01


def _tensor_bytes(state):
    return sum(v.numel() * v.element_size() for v in state.values()
               if torch.is_tensor(v))


def save_checkpoint(model, optimizer, updates, *, path, source, plan_sha, code_sha,
                    counts, smoke=False, smoke_indices=None):
    rank, world = rank_world()
    rng = capture_rng()
    rng["window_exposure_counts"] = counts.tolist()
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(rng, gathered, dst=0)
    if rank == 0:
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "rank_rng": gathered, "completed_updates": updates,
            "new_exposures": updates * 8, "source_exposure": SOURCE_EXPOSURES,
            "model_exposure": SOURCE_EXPOSURES + (
                8 * max(smoke_indices) if smoke and smoke_indices else updates * 8
            ),
            "epoch": updates // UPDATES_PER_EPOCH, "alpha": ALPHA,
            "arm": ARM, "competition_lambda": COMPETITION_LAMBDA,
            "config": dataclasses.asdict(model.opt), "plan_sha256": plan_sha,
            "source_checkpoint": source, "code_sha": code_sha,
            "world_size": world, "smoke": smoke,
            "smoke_update_indices": smoke_indices,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        check = torch.load(tmp, map_location="cpu", weights_only=False, mmap=True)
        expected = (ARM, ALPHA, COMPETITION_LAMBDA, updates, plan_sha, world)
        actual = (check["arm"], check["alpha"], check["competition_lambda"],
                  check["completed_updates"], check["plan_sha256"], check["world_size"])
        if actual != expected:
            raise RuntimeError(f"checkpoint metadata verification mismatch: {actual}")
        del check
        os.replace(tmp, path)
    dist.barrier()


def compare_tree(a, b, path="root"):
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"tensor contract shape/dtype mismatch at {path}")
        if a.is_floating_point() or a.is_complex():
            if not torch.isclose(a, b, atol=1e-6, rtol=1e-6).all():
                raise AssertionError(f"tensor contract value mismatch at {path}")
        elif not torch.equal(a, b):
            raise AssertionError(f"tensor contract value mismatch at {path}")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"output keys differ at {path}")
        for k in a:
            compare_tree(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError(f"output lengths differ at {path}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare_tree(x, y, f"{path}[{i}]")


def model_method_contract(model, opt, window, device):
    batch = build_batch(opt, window, device)
    original_keys = tuple(model.state_dict().keys())
    model.eval()
    with torch.no_grad():
        model.competition_lambda = 0.0
        base_out, base_metrics = model.step_loss(
            batch, step=SOURCE_EXPOSURES + 8 * 25, understanding_weight=1.0)
        model.competition_lambda = COMPETITION_LAMBDA
        comp_out, comp_metrics = model.step_loss(
            batch, step=SOURCE_EXPOSURES + 8 * 25, understanding_weight=1.0)
    compare_tree(base_out["prediction"], comp_out["prediction"], "prediction")
    from scripts.export_object_locus_v3_set_official import assemble_panoptic
    base_packed = assemble_panoptic(base_out["prediction"])
    comp_packed = assemble_panoptic(comp_out["prediction"])
    for left, right in zip(base_packed, comp_packed):
        if torch.is_tensor(left) and torch.is_tensor(right):
            if not torch.equal(left, right):
                raise AssertionError("lambda changed packed official output")
    for key in ("loss_recon", "loss_understanding", "loss", "loss_total"):
        if key not in base_metrics or key not in comp_metrics:
            raise AssertionError(f"missing lambda contract metric {key}")
    if not torch.isclose(base_metrics["loss_understanding"], comp_metrics["loss_under_old"], atol=1e-6, rtol=1e-6):
        raise AssertionError("lambda=2 changed the original understanding loss")
    if not torch.isclose(comp_metrics["loss_understanding"],
                         comp_metrics["loss_under_old"] + .2 * comp_metrics["loss_competition"],
                         atol=1e-6, rtol=1e-6):
        raise AssertionError("competition external/internal coefficient mismatch")
    if tuple(model.state_dict().keys()) != original_keys:
        raise AssertionError("method-only model changed state_dict keys")
    model.competition_lambda = COMPETITION_LAMBDA
    model.train()
    del batch, base_out, comp_out, base_metrics, comp_metrics


def run_eight_card_smoke(manifest, plan, plan_sha, source, code_sha, device):
    rank, world = rank_world()
    if world != 8:
        raise RuntimeError("required competition smoke must use eight ranks")
    model, opt, _ = build_comp_model(device)
    optimizer = build_optimizer(model)
    model_method_contract(model, opt, manifest["expanded_train_windows"][
        plan["entries"][0]["rank_windows"][rank]], device)
    counts = np.zeros(WINDOWS, dtype=np.int64)
    details = []
    for smoke_i, update in enumerate((25, 26)):
        # The registered data identities are the first two global plan rows;
        # only the optimizer/LR/warm-up position is moved to u=25/26 so that
        # the understanding and competition gradients are active.
        entry = plan["entries"][smoke_i]
        wi = int(entry["rank_windows"][rank])
        batch = build_batch(opt, manifest["expanded_train_windows"][wi], device)
        _, row = train_step(model, opt, optimizer, batch, update, ALPHA,
                            source_exposure=SOURCE_EXPOSURES, diagnose=False)
        counts[wi] += 1
        if not all(np.isfinite(float(row[k])) for k in (
            "loss_recon", "loss_understanding", "loss_under_old",
            "loss_competition", "weighted_loss_competition",
            "preclip_global_grad_norm")):
            raise FloatingPointError(f"nonfinite competition smoke update {update}")
        if update == 25 and row["understanding_weight"] != 1.0:
            raise AssertionError("u=25 must have understanding weight 1")
        details.append({k: row[k] for k in (
            "local_update", "loss_recon", "loss_understanding", "loss_under_old",
            "loss_competition", "weighted_loss_competition", "understanding_weight",
            "preclip_global_grad_norm", "clipped", "allocated", "reserved",
            "peak_allocated", "peak_reserved")})
        del batch
    smoke_counts = torch.as_tensor(counts, device=device, dtype=torch.int64)
    dist.all_reduce(smoke_counts, op=dist.ReduceOp.SUM)
    expected_smoke_counts = np.zeros(WINDOWS, dtype=np.int64)
    for entry in plan["entries"][:2]:
        expected_smoke_counts[np.asarray(entry["rank_windows"], dtype=np.int64)] += 1
    if smoke_counts.cpu().tolist() != expected_smoke_counts.tolist():
        raise AssertionError("smoke exposures differ from the first two registered plan rows")
    smoke_path = REPORT_ROOT / "smoke" / "comp_gc001_smoke_checkpoint.pt"
    save_checkpoint(model, optimizer, 2, path=smoke_path, source=source,
                    plan_sha=plan_sha, code_sha=code_sha,
                    counts=smoke_counts.cpu().numpy(),
                    smoke=True, smoke_indices=[25, 26])
    # Read the written payload back into the live smoke model/optimizer. This is
    # an isolated smoke resume; formal weights are built fresh below.
    blob = torch.load(smoke_path, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(blob["model"], strict=True)
    optimizer.load_state_dict(blob["optimizer"])
    restore_rng(blob["rank_rng"][rank])
    if blob["smoke_update_indices"] != [25, 26] or blob["completed_updates"] != 2:
        raise RuntimeError("smoke checkpoint resume metadata mismatch")
    fingerprint = torch.zeros(6, device=device, dtype=torch.float64)
    with torch.no_grad():
        for parameter in model.parameters():
            value = parameter.detach().double()
            fingerprint[0] += value.sum()
            fingerprint[1] += value.square().sum()
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    if value.is_floating_point():
                        item = value.detach().double()
                        fingerprint[2] += item.sum()
                        fingerprint[3] += item.square().sum()
                    else:
                        fingerprint[4] += value.detach().double().sum()
    fingerprint[5] = len(optimizer.state)
    gathered = [torch.empty_like(fingerprint) for _ in range(world)]
    dist.all_gather(gathered, fingerprint)
    if any(not torch.equal(gathered[0], item) for item in gathered[1:]):
        raise RuntimeError("smoke model or optimizer state differs across ranks after resume")
    if rank == 0:
        details.append({"rank_state_fingerprint": gathered[0].cpu().tolist()})
    del blob, optimizer, model
    torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        write_json(REPORT_ROOT / "eight_smoke.json", {
            "status": "PASS", "world_size": 8, "arm": ARM,
            "alpha": ALPHA, "competition_lambda": COMPETITION_LAMBDA,
            "smoke_update_indices": [25, 26],
            "plan_entry_indices": [0, 1],
            "windows_per_rank": 2, "actual_optimizer_updates_per_rank": 2,
            "global_optimizer_updates": 2, "global_exposures": 16,
            "resume_checkpoint": str(smoke_path),
            "resume_readback": True, "rank_model_optimizer_state_sync": True,
            "model_method_contract": "PASS",
            "updates": details,
        })
    dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=(ARM,), required=True)
    args = ap.parse_args()
    os.environ["TASK_ARM"] = ARM
    device = init_distributed()
    rank, world = rank_world()
    if world != 8:
        raise RuntimeError("formal training requires eight ranks")
    if rank == 0:
        result = run_cpu_contracts()
        write_json(REPORT_ROOT / "cpu_contracts.json", result)
    dist.barrier()
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise RuntimeError("formal competition worktree must be clean")
    receipt_path = REPORT_ROOT / "git_provenance.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("training_sha") != code or not receipt.get("clean"):
        raise RuntimeError("pushed provenance does not match running source")
    plan_path = REPORT_ROOT / "training_plan.json"
    plan_sha = sha256(plan_path)
    if plan_sha != receipt.get("plan_sha256"):
        raise RuntimeError("registered copied plan SHA mismatch")
    if sha256(SOURCE_MANIFEST) != receipt.get("source_manifest_sha256"):
        raise RuntimeError("registered source manifest SHA mismatch")
    if rank == 0:
        report_arm = REPORT_ROOT / ARM
        run_arm = RUN_ROOT / ARM
        if report_arm.exists() and any(report_arm.iterdir()):
            raise RuntimeError("competition report output already exists; refusing overwrite")
        if run_arm.exists() and any(run_arm.iterdir()):
            raise RuntimeError("competition run output already exists; refusing overwrite")
        report_arm.mkdir(parents=True, exist_ok=True)
        run_arm.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    plan = json.loads(plan_path.read_text())
    windows = manifest.get("expanded_train_windows", [])
    if len(windows) != WINDOWS or len({str(w["scene"]) for w in windows}) != 128:
        raise RuntimeError("registered source manifest is not 128 scenes/1008 windows")
    expected_plan_meta = {"seed": 42, "epochs": 8, "windows": 1008,
                          "world_size": 8, "updates": 1008, "exposures": 8064}
    if any(plan.get(k) != v for k, v in expected_plan_meta.items()):
        raise RuntimeError("registered GC plan metadata mismatch")
    if len(plan.get("entries", [])) != TOTAL_UPDATES:
        raise RuntimeError("registered GC plan entry count mismatch")
    observed_counts = np.zeros(WINDOWS, dtype=np.int64)
    for update, entry in enumerate(plan["entries"]):
        if entry.get("update") != update or entry.get("epoch") != update // UPDATES_PER_EPOCH:
            raise RuntimeError(f"registered plan update/epoch identity mismatch at {update}")
        ids = np.asarray(entry.get("rank_windows", []), dtype=np.int64)
        if ids.shape != (WORLD,) or np.any(ids < 0) or np.any(ids >= WINDOWS):
            raise RuntimeError(f"invalid registered rank window allocation at update {update}")
        observed_counts[ids] += 1
    if observed_counts.tolist() != plan.get("window_exposure_counts") or not np.all(observed_counts == 8):
        raise RuntimeError("registered plan does not expose each window exactly eight times")
    for epoch in range(EPOCHS):
        ids = [w for entry in plan["entries"][epoch * UPDATES_PER_EPOCH:(epoch + 1) * UPDATES_PER_EPOCH]
               for w in entry["rank_windows"]]
        if sorted(ids) != list(range(WINDOWS)):
            raise RuntimeError(f"registered plan epoch {epoch} is not a fixed-window permutation")
    train_scenes = {str(w["scene"]) for w in windows}
    for split in SPLITS:
        if split not in manifest:
            raise RuntimeError(f"registered manifest missing fixed split {split}")
    if (train_scenes & {str(w["scene"]) for w in manifest["dev8"]} or
            train_scenes & {str(w["scene"]) for w in manifest["val32"]}):
        raise RuntimeError("dev8/val32 are not scene independent from registered training scenes")
    source_blob = load_source_blob()
    source = {"path": str(SOURCE_CHECKPOINT), "sha256": SOURCE_SHA256,
              "epoch": source_blob["epoch"],
              "completed_updates": source_blob["completed_updates"],
              "completed_exposures": source_blob["completed_exposures"],
              "strict_load": True, "state_tensors": len(source_blob["model"])}
    del source_blob

    # Run isolated smoke first; build_comp_model resets the registered rank RNG.
    smoke_manifest = json.loads(SOURCE_MANIFEST.read_text())
    run_eight_card_smoke(smoke_manifest, plan, plan_sha, source, code_sha=code,
                         device=device)
    if rank == 0:
        smoke = json.loads((REPORT_ROOT / "eight_smoke.json").read_text())
        if smoke.get("status") != "PASS":
            raise RuntimeError("eight-card smoke did not pass")
    dist.barrier()

    model, opt, loaded_source = build_comp_model(device)
    if loaded_source != source:
        raise RuntimeError("fresh formal model source metadata mismatch")
    optimizer = build_optimizer(model)
    # Budget estimate from actual FP32 model and AdamW moment tensor sizes.
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    checkpoint_bytes = model_bytes * 3  # model + AdamW exp_avg + exp_avg_sq
    peak_bytes = checkpoint_bytes * 5 + 7 * 1024**3  # four saved + atomic temporary + report margin
    if rank == 0:
        write_json(REPORT_ROOT / "disk_budget_estimate.json", {
            "model_tensor_bytes": model_bytes,
            "adamw_moment_tensor_bytes_estimate": model_bytes * 2,
            "single_full_checkpoint_bytes_estimate": checkpoint_bytes,
            "peak_bytes_estimate": peak_bytes,
            "checkpoint_count_at_peak": 5,
            "report_and_cache_margin_bytes": 7 * 1024**3,
            "available_bytes": int(os.statvfs(REPORT_ROOT).f_bavail * os.statvfs(REPORT_ROOT).f_frsize),
        })
        if int(os.statvfs(REPORT_ROOT).f_bavail * os.statvfs(REPORT_ROOT).f_frsize) < peak_bytes:
            raise RuntimeError("insufficient disk space for registered checkpoint peak")
        with (REPORT_ROOT / "runtime_versions.json").open("w") as f:
            import torchvision
            import gsplat
            import torchmetrics
            json.dump({"python": __import__("sys").version,
                       "torch": torch.__version__, "torchvision": torchvision.__version__,
                       "gsplat": getattr(gsplat, "__version__", "unknown"),
                       "torchmetrics": torchmetrics.__version__,
                       "python_executable": __import__("sys").executable}, f, indent=2)
    dist.barrier()
    counts = np.zeros(WINDOWS, dtype=np.int64)
    u = 0
    arm_report = REPORT_ROOT / ARM
    arm_run = RUN_ROOT / ARM
    if rank == 0:
        write_json(arm_report / "run_manifest.json", {
            "arm": ARM, "alpha": ALPHA, "competition_lambda": COMPETITION_LAMBDA,
            "competition_external_weight": 0.1, "unique_scientific_variable": "final-context pixel competition supervision",
            "gc_control": "gc001", "git_sha": code, "source_checkpoint": source,
            "source_manifest_path": str(SOURCE_MANIFEST),
            "source_manifest_sha256": receipt["source_manifest_sha256"],
            "source_training_plan_path": "/space/mawb/ssst/group_plus/object_locus_gc_sweep_v1/training_plan.json",
            "source_training_plan_sha256": receipt["plan_sha256"],
            "plan_sha256": plan_sha, "scenes": 128, "windows": 1008,
            "epochs": 8, "updates": 1008, "new_exposures": 8064,
            "model_exposure_endpoint": 58128,
            "checkpoint_epochs": [0, 2, 4, 8],
            "optimizer": "fresh AdamW; inherited exact GC parameter groups/schedule",
            "smoke": "two isolated updates at schedule indices 25 and 26; excluded from formal budget",
        })
        (arm_report / "training_rank0.jsonl").write_text("")
        write_json(arm_report / "progress.json", {
            "arm": ARM, "status": "INITIALIZING", "completed_updates": 0,
            "new_exposures": 0, "model_exposure": SOURCE_EXPOSURES,
            "smoke_completed": True, "job_id": os.environ.get("SLURM_JOB_ID"),
        })
    dist.barrier()
    save_checkpoint(model, optimizer, 0, path=arm_run / "checkpoint_epoch0.pt",
                    source=source, plan_sha=plan_sha, code_sha=code, counts=counts)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    while u < TOTAL_UPDATES:
        entry = plan["entries"][u]
        wi = int(entry["rank_windows"][rank])
        batch = build_batch(opt, manifest["expanded_train_windows"][wi], device)
        out, row = train_step(model, opt, optimizer, batch, u, ALPHA,
                              source_exposure=SOURCE_EXPOSURES,
                              diagnose=((u + 1) % 100 == 0))
        counts[wi] += 1
        u += 1
        checkpoint_node = u in (252, 504, 1008)
        if checkpoint_node:
            save_checkpoint(model, optimizer, u,
                path=arm_run / f"checkpoint_epoch{u // UPDATES_PER_EPOCH}.pt",
                source=source, plan_sha=plan_sha, code_sha=code, counts=counts)
        if u == 1 or u % 10 == 0 or checkpoint_node:
            vals = torch.tensor([row["loss_recon"], row["loss_understanding"],
                                 row["monitor_total_loss"], row["loss_under_old"],
                                 row["loss_competition"], row["weighted_loss_competition"]],
                                device=device, dtype=torch.float64)
            dist.all_reduce(vals, op=dist.ReduceOp.SUM); vals /= world
            if rank == 0:
                row.update(loss_recon=float(vals[0]), loss_understanding=float(vals[1]),
                           monitor_total_loss=float(vals[2]), loss_under_old=float(vals[3]),
                           loss_competition=float(vals[4]), weighted_loss_competition=float(vals[5]),
                           local_epoch=u / UPDATES_PER_EPOCH, local_update=u,
                           new_exposures=8*u, model_exposure=SOURCE_EXPOSURES+8*(u-1),
                           endpoint_model_exposure=SOURCE_EXPOSURES+8*u, alpha=ALPHA,
                           competition_lambda=COMPETITION_LAMBDA,
                           window_index=wi,
                           window_identity=manifest["expanded_train_windows"][wi],
                           recent_checkpoint=str(arm_run / f"checkpoint_epoch{max(x for x in (0,2,4,8) if x*126<=u)}.pt"))
                with (arm_report / "training_rank0.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                write_json(arm_report / "progress.json", {
                    "arm": ARM, "status": "TRAINING", "completed_updates": u,
                    "local_epoch": u/126, "new_exposures": 8*u,
                    "model_exposure": SOURCE_EXPOSURES+8*(u-1),
                    "endpoint_model_exposure": SOURCE_EXPOSURES+8*u,
                    "loss_recon": row["loss_recon"],
                    "loss_under_old": row["loss_under_old"],
                    "loss_competition": row["loss_competition"],
                    "loss_under_new": row["loss_under_new"],
                    "weighted_loss_competition": row["weighted_loss_competition"],
                    "understanding_weight": row["understanding_weight"],
                    "alpha": ALPHA, "competition_lambda": COMPETITION_LAMBDA,
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "recent_checkpoint": row["recent_checkpoint"],
                })
                if u == 1:
                    write_json(arm_report / "startup_confirmation.json", {
                        "status": "FIRST_FORMAL_UPDATE_FINITE", "arm": ARM,
                        "alpha": ALPHA, "competition_lambda": COMPETITION_LAMBDA,
                        "job_id": os.environ.get("SLURM_JOB_ID"),
                        "epoch0_checkpoint": str(arm_run / "checkpoint_epoch0.pt"),
                        "completed_updates": 1, "new_exposures": 8,
                        "model_exposure": row["model_exposure"],
                        "endpoint_model_exposure": SOURCE_EXPOSURES+8,
                        "finite": True, "code_sha": code, "plan_sha256": plan_sha,
                        "loss_recon": row["loss_recon"],
                        "loss_under_old": row["loss_under_old"],
                        "loss_competition": row["loss_competition"],
                        "loss_under_new": row["loss_under_new"],
                        "preclip_global_grad_norm": row["preclip_global_grad_norm"],
                    })
            dist.barrier()
        del out, batch
    totals = torch.tensor(counts, device=device, dtype=torch.int64)
    dist.all_reduce(totals)
    if totals.cpu().tolist() != plan["window_exposure_counts"]:
        raise RuntimeError("formal plan exposure counts mismatch")
    if rank == 0:
        write_json(arm_report / "progress.json", {
            "arm": ARM, "status": "COMPLETE", "completed_updates": u,
            "new_exposures": 8064, "model_exposure": 58128,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "recent_checkpoint": str(arm_run / "checkpoint_epoch8.pt"),
        })
        write_json(arm_report / "training_complete.json", {
            "arm": ARM, "completed_updates": u, "new_exposures": 8064,
            "source_exposure": SOURCE_EXPOSURES, "model_exposure": 58128,
            "window_exposure_counts": totals.cpu().tolist(),
            "training_sha": code, "plan_sha256": plan_sha,
            "evaluation": "scheduled dependent four-arm endpoint evaluation",
        })
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
