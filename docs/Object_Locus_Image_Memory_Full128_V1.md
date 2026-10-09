# Full-data image-memory paired continuation

Scientific starting commit: `9ce7b18b68bae9c4c7c519b670ad1fa79a66f9a0`, branch `object-locus-gc-sweep-v1`.
Task branch: `object-locus-image-memory-full128-v1`.

The only scientific variable is the image memory read at object layers L6/L8/L10/L12: C32 uses the existing bilinear 32×32 construction, U128 directly flattens the native 128×128 features. Both retain view-major/spatial order. Memory shapes are [B,2048,256] and [B,32768,256]. State keys, shapes, initialization and all other scientific computations remain unchanged. Legacy entry points default to 32. New run manifests and checkpoint config/top-level metadata explicitly store `object_image_memory_size`; evaluation must restore this field, never infer it from parameter shapes.

Both arms strictly load Full1201 epoch6 (SHA256 `68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a`, updates6258/exposures50064) and initialize fresh AdamW. No epoch0 model copy is saved. Seed42 and per-rank seed42+100003*rank match the GC runtime.

The unchanged source Full1201 manifest supplies 1191 scenes and 8337 two-context windows with GT poses. Original training plan and provenance are retained. Eight new epochs use default_rng(42+epoch), first7 permutation entries appended. Each epoch has1043 updates/8344 exposures. Each arm adds8344 updates/66752 exposures (56 padding exposures), ending at14602 cumulative updates/116816 cumulative exposures.

Node3dimage-11, four RTX3090, one window/rank/microstep, two microsteps. Rank r reads ids[r] then ids[4+r]. Local gradients are divided by2, cross-rank SUM divided by4 once, then one global clip1.0 and one optimizer step. Reconstruction receives g_rec+0.01*g_under; understanding/object receive g_rec+g_under. Under gradients include min(8*new_update/200,1). Both microsteps use model exposure50064+8*new_update and identical LR. Existing beta remains0.1. No parameters are frozen.

GC optimizer groups/WD are reused, including both level_embed WD0. FP32, TF32disabled, AdamW betas(.9,.95), eps1e-8, peak LRs1e-6/1e-5/1e-4. For t=new_update+1 and T=8344, warmup t/200 through200 then0.1+0.9*(1+cos(pi*(t-200)/(T-200)))/2.

Each epoch atomically writes and reads back a full latest recovery point, then removes the previous latest. New epoch4 additionally saves model-only; epoch8 endpoint hardlinks the final latest. Resolution, arm, plan/code SHA, optimizer, LR position, rank RNG and window counts accompany recovery. Cross-arm/recipe restore is rejected. No automatic restore or retry is registered.

Required gates: memory construction/order, complete strict initialization and identical groups, full plan/microstep mapping, independent small numeric eight-sample GC reference, U128 single-real-window forward/loss/backward, four-rank two-update smoke for each arm. Smoke states are discarded. No task-metric gates or evaluation jobs.

Launch each independent arm with `bash scripts/submit_object_locus_image_memory.sh u128 train` then c32. Jobs request partition3090/node11/four GPUs/16CPU/128G/72h, OMP4, without exclusive allocation or dependency. Fixed pushed clean checkout is bound to login-host provenance; no compute-host GitHub query or checkout hook. Formal initialization reloads the source. Every10 updates logs finite losses, LR, clip norm, warmup/beta, windows, peak memory and throughput. Once U128 reaches10 finite formal updates and startup_confirmation.json, the executor stops polling. Endpoint comparison and full official evaluation require a later user instruction.
