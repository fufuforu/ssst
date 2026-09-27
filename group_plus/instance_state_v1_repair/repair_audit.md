# repair_audit - instance_state_v1, revision 1

Object: `e0e6805f972114741e010f4e67e1f0712a100d08`.  Every item of the repair
prompt is listed with its fix and the evidence that produced it.  Status labels:
**fixed** / **fixed + verified** / **not executed**.

## 0. Workspace and locked samples

| item | outcome |
|---|---|
| HEAD at repair | `e0e6805` (my last commit) + concurrent-session commits; every dependency file of this task unchanged by them |
| report dirs | `group_plus/instance_state_v1_repair/`, `workspace_group_plus/instance_state_v1_repair/` (e0e6805 reports left untouched) |
| locked samples | `pilot_windows.json` sha256 `e5c5878b21d57218…`, `plan_paired_2000.json` `32c40a72e45843a9…`, `monitor_8pairs.json` `5dae7077779dd398…` - **byte-identical** to e0e6805 (copied, never re-sampled) |
| re-screening with the corrected GT rule (`2<=sem<=19`, `ins>0`, area>=100) | **all 4 windows still qualify** (`pilot_recheck.json`) -> no `DATA_PROTOCOL_BLOCKED` |

## 1. CUDA assert - reproduced and attributed

`cuda_repro.log` (+ `cuda_repro_summary.json`): a pristine e0e6805 copy was run with
`CUDA_LAUNCH_BLOCKING=1`; the 13 A/B/C checks passed and the first failing frame is

```
tokengs/models/instance_state_loss.py, line 120, in thing_loss
    y = tgt.reshape(tgt.shape[0], -1)[:, flat_idx]
RuntimeError: CUDA error: device-side assert triggered
```

That confirms the static finding `instance_state_loss.py:92-120`: `flat_idx` was
built from a 4-view valid mask while the GT tensor has 2 views, so the gather is
out of bounds.  Evidence boundary: the *synchronous* kernel is proven; the earlier
asynchronous run only gave the async text.

## 2. Loss / context contract (L1-L8) - fixed + verified on CPU

`scripts/check_instance_state_loss_contract.py` (CPU, no renderer/weights/dataset)
implements synthetic tensors whose pixel coordinates encode their own indices and
compares against closed-form NumPy algebra.  `loss_contract.json`: **14/14 PASS**.

| item | fix | evidence |
|---|---|---|
| L1 understanding uses only context (0,1) | `step_loss` builds a 2-camera `context_decoder`; `_region_prediction_from_state` renders region/semantic/identity/alpha on it; loss asserts V==2 | `d.novel_labels_ignored` identical with 2 and 4 GT views |
| L2 axis order | `_flat_regions` permutes `[B,V,C,H,W]->[B,C,VHW]`; `idx=((v*H)+y)*W+x` guard | `L2.permute_before_flatten`, `L2.flat_index_decoding` |
| L3 logit not log | `z = logit(clamp(p,1e-6,1-1e-6))`, shared by matcher and full-pixel BCE/Dice | `b.logit_roundtrip`: sigmoid(logit(0.9))=0.900000 vs sigmoid(log(0.9))=0.473684 |
| L4 CE from raw 19-d logits | `thing_class_logits[...,2:]`, target `cls-2`/18, per-class weight `[1]*18+[0.1]`, `cross_entropy(reduction='mean')` | `L4.ce_weighted_mean` 0.126928 = closed form |
| L5 stuff BCE mean of two classes | `torch.stack(per_class).mean()` | `stuff.finite` path + contract run |
| L6 identity normalised on dim=2 | `F.normalize(..., dim=2)` | `L6.identity_unit_norm` |
| L7 differentiable zero | `_nearby_zero(prediction["gaussians"])` | `c.empty_thing_finite` |
| L7/contract fail-fast | label range, single class per instance id, >100 things | `c.rejects_out_of_range_label`, `c.rejects_cross_class_instance` |
| L8 graph vs logs | `metrics["loss_understanding"]` keeps the graph, all metric copies are floats | used by the reworked smoke D |

## 3. State structure (S1-S7) - fixed

| item | fix |
|---|---|
| S1 single cosine temperature | `assign`: `cos/0.1`, then `-0.1*clamp(dist2,25)`; the final softmax no longer divides by 0.1 again (that double division was the defect) |
| S2 zero-init coupling projections | `proj_wh/proj_wmu/proj_wr/proj_wgs` + `token_void/proj_de/proj_gvoid` are zero-initialised; the other modules keep Xavier; seed 31415 island unchanged |
| S3 `ell` once | computed at `mu6`, detach, passed into `update_states` for all four stages |
| S4 low-mass keeps q | `q_new = where(mass<1e-4, q_old, q_new)` in addition to the c/s keep |
| S5 naming | `NUM_QUERIES=102` (states) vs `NUM_REGION_CHANNELS=103` (rendered channels); `VOID_INDEX=102` |
| S6 stateful reconstruction interfaces | the new model explicitly overrides `_decode`, `forward_reconstruction_only` and `_layer_objective`; `step_loss` reuses that `_layer_objective`; one `_gaussians_from_state` for both paths |
| S7 single grouping function | `run_instance_state_v1.build_optimizer` builds the four AdamW groups **and** the report; `query_init`, biases, norms and `_no_weight_decay` are excluded from decay; aliases deduplicated by `id()` |

## 4. Scripts and provenance

| item | outcome |
|---|---|
| `scripts/instance_state_paired.py` | added (`run_paired`, `run_full`, shared `lr_at`) |
| `scripts/submit_instance_state_v1.sh` | added (3090, cpu 8, mem 64G, `--exclude=3dimage-13`, 8h/12h) |
| digest bug | fixed: `pretrained_digest` / `pilot_digest` / `plan_digest` / `monitor_digest` are separate; see `corrected_provenance.json` |
| real checkpoint size | measured by an in-memory `torch.save` of `{model, optimizer, step, RNG}` -> **891,358,595 bytes**; caveat recorded in `actual_storage_budget.json`: this file has *empty* Adam state because no optimizer step ran, so the steady-state size is about 2x larger and the budget keeps that factor |
| `prepare` on CPU | re-run end-to-end in the repair dirs; `--help` paths of every referenced module import on CPU |

## 5. Not executed in this session

`eval_instance_state_v1.py` / `export_instance_state_official.py` are **not yet
written**, and the GPU smoke (A-J), the C/E paired 2000-step arms, the 8-pair
official subset, the 55/110 table, `gate.json` and the full phase were **not run**.
The repaired smoke has not been executed at all yet, so no claim of a passing
smoke is made.  `report.md` in this directory states the exact blocker.
