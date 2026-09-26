# Step 1 report: group-mask formation, read-only diagnosis (pre-registered stop)

**Read-only.** No training, no optimizer step, no checkpoint write, no threshold
change, no modification of the parallel SIU3R files.  Artifacts:
`group_plus/routing_v1/{step1_summary.json, fragmentation.csv, oracle.csv,
analysis.json}` (185 KB total) plus `step1_smoke.json`.  Script:
`scripts/audit_group_routing.py`.  GPU job `55389` on partition 3090 (-x
3dimage-13), ~24 min wall clock.

## 1. What the head actually does

`GroupQueryHead.forward_full` (verified against the source, `code_facts` in the
summary JSON):

* **one** query↔token interaction layer: a single 4-head
  `nn.MultiheadAttention` cross-attention from the 100 learnable queries to the
  token features;
* **no query↔query communication** — after the cross-attention only a per-query
  `query_norm(queries + attended)` and an MLP, no self-attention among queries;
* the anchor/radius encoding is added to the **token** features
  (`token_proj(token_norm(tokens)) + spatial_proj(build_anchor_encoding(...))`)
  before that cross-attention and before the similarity;
* the final logits are `(token_features · query_features) · logit_scale`
  concatenated with the shared scalar `background_bias` → `slot_logits[B,T,101]`;
  the objectness/class heads read the query features.

Checkpoints (SHA256 + mtime recorded before and after, both unchanged):
`workspace_group_plus/arm_g0plus/ckpt_step6000` (`a63dd485cc2d1b3f…`) and
`workspace_group_locusgs/arm_g0/ckpt_step6000` (`dbe5ea8311b466e6…`), same plan
`a2a65c13…`.

## 2. Smoke and cross-check (1 checkpoint × 1 window, then the 8 fixed windows)

Smoke (G0+ step6000, scene0059_00): T=1024 (smoke-window assertion only),
`slot_logits=[1,1024,101]`, fp32, finite, per-token 101-way probability sums to 1
(≤1e-5), `Σ group + background = alpha` ≤ 2e-6, repeat forward RGB/depth
bit-identical.

Cross-check against the recorded G0+ numbers:

| view scope | TP | FP | FN | records | mean AP50 |
|---|---|---|---|---|---|
| all 4 views | 6 | 140 | 104 | 110 | 0.0984375 (recorded 0.0984, Δ=4e-5) |
| novel views 2/3 | 3 | 71 | 52 | 55 | — |

Both match the recorded values exactly, so the harness is consistent; the branch
decision below therefore uses the **novel-only 55 records**.

## 3. Fragmentation of the 100 group masks (novel-only 55 records)

Per novel visible GT instance: rank the 100 single-group hard masks (mass > 0.5,
area ≥ 50 px) by IoU descending with group-ID tie-break, take the union of the
top 1/2/3 and record IoU1/IoU2/IoU3 and Δ = max(IoU1,IoU2,IoU3) − IoU1.

| variant | mean IoU1 | mean IoU3 | mean Δ | records with Δ ≥ 0.10 | union-3 area / GT area | union-3 area intruding other GT |
|---|---|---|---|---|---|---|
| G0+ step6000 | 0.181 | 0.159 | **0.026** | **4/55** | 27.1× | 39.2 % |
| G0 step6000 | 0.267 | 0.199 | 0.039 | 9/55 | 37.2× | 37.9 % |

Pre-registered "fragmentation significant" requires mean Δ ≥ 0.08 **and**
Δ ≥ 0.10 in ≥ 14/55: **not met** (both arms).  The top-2/3 union is not better
than the single best mask; instead the high-IoU masks are already much larger
than the instance (27–37× GT area, ~39 % of their area on other GT instances).

## 4. Token-granularity oracle (token assignment from context GT only)

For each window: render 1024 one-hot token channels with the same compositing
weights over the two context views, accumulate each token's contribution mass to
the context GT thing instances (alpha > 0.5, semantic 2–19, instance id > 0),
assign each token to its arg-max instance (else *rest*), then render the novel
per-instance masks by giving all 64 Gaussians of a token that token's one-hot.
Novel GT is used only for scoring.

| metric (novel-only 55) | value | pre-registered bar |
|---|---|---|
| oracle IoU ≥ 0.5 | **22/55** | ≥ 28/55 |
| context_visible subset | **55/55** (every novel instance has ≥1 valid GT pixel in a context frame) | — |
| context_visible pass rate | 22/55 = **40 %** | ≥ 50 % would mean "context-visibility limitation" |
| oracle IoU p25 / p50 / p75 / p90 / max | 0.202 / 0.453 / 0.558 / 0.683 / 0.844 | — |
| *rest* token fraction per scene | 0.020 – 0.627 (mean ≈ 0.21) | recorded |
| oracle pass, GT < 3000 px | **2/20** | — |
| oracle pass, GT ≥ 3000 px | **20/35** | — |

## 5. Pre-registered decision

* oracle feasible? **No** — 22/55 < 28/55.
* fragmentation significant? **No** — mean Δ 0.026 < 0.08 and 4/55 < 14/55.
* context_visible subset: not empty (55/55) and its pass rate is 40 % < 50 %.

**Branch: STOP_ORACLE_INFEASIBLE — stop category "oracle read-out insufficient
under this protocol"** (not the context-visibility limitation, and not
"cannot determine" since the subset is non-empty).

Consequences, exactly as pre-registered: Step 2 (module implementation) and
Step 3 (paired training) are **not executed**; no training job was created.  The
9 training instances are interpretation-only and were not used for the vote.

**This stop must not be read as "all query or spatial methods fail".** It rules
out, for this checkpoint and this read-out, that the existing 100 group masks
are *fragments* of one instance (merging the top-2/3 does not help: 0.026 mean Δ)
and that the token granularity, when given context GT, already suffices under the
protocol's 28/55 bar.  Both statements are specific to the current modules,
grid/thresholds and this read-out.

## 6. Interpretation (not branch voting)

* The oracle's failure is **strongly size-dependent**: 2/20 for GT < 3000 px
  versus 20/35 above — i.e. token-level grouping from context GT mostly works for
  the larger instances.
* Scenes with a very high *rest* fraction (scene0695_00 0.63, scene0472_01 0.30,
  scene0072_02 0.28) are also the ones with low oracle pass rates, suggesting
  that in those windows most token mass is not attributable to any single context
  thing instance.
* G0 and G0+ behave the same way (fragmentation 0.039 vs 0.026; oracle not
  re-measured separately for the branch), so the background-slot supervision did
  not change this mechanism.

## 7. Next-round hypotheses (at most 3; each with a mechanically checkable criterion)

1. **Footprint-scale limit.** Failing instances are smaller than the Gaussian
   footprint. Check next: for each failing novel instance compute the world-space
   equivalent radius from GT depth + intrinsics; if for ≥ 60 % of them it is
   below the frozen decode radius 0.15, the footprint/decoding scale is the
   binding factor (else reject).
2. **Mixed tokens, not input granularity.** A token that straddles several
   instances cannot vote for one. Check next: recompute the same oracle using
   only tokens whose contribution mass is ≥ 50 % inside a single instance; if
   the pass rate rises from 22/55 to ≥ 28/55, per-token purity is the binding
   factor (else reject).
3. **Read-out strictness for small masks.** Some failing oracle masks may be
   correct but below the frozen `mask > 0.5` / `area ≥ 50 px` bar. Check next
   (diagnostic only, thresholds stay frozen): the IoU of the same oracle masks
   without the area filter; if ≥ half of the failures cross IoU 0.5, the frozen
   read-out, not the representation, is implicated for them (else reject).

## 8. Limits

* One checkpoint per arm (step6000) for the diagnosis; the branch vote uses the
  novel-only 55 records of the 8 unseen scenes (the 110-record all-view numbers
  are reported only for the cross-check).
* The oracle uses the model's own Gaussians and compositing weights; it is an
  upper bound *of this read-out*, not of the architecture.
* `rest` tokens are assigned to no instance by construction; their fraction is
  reported per scene (0.02–0.63).
