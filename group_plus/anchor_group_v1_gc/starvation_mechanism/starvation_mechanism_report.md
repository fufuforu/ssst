# Anchor-Group V1-GC Slot-Starvation Mechanism Audit

Read-only endpoint diagnostics. No optimizer was constructed or stepped.

## Endpoint and historical query sets

- Endpoint: step 5000, `LOCUSGS_ANCHOR_GROUP_V1`, `ANCHOR_GROUP_V1_GC_ALPHA001`, shared gradient scale 0.01.
- Historical top5: [(27, 860), (75, 738), (52, 592), (71, 505), (38, 464)]
- Historical top10: [(27, 860), (75, 738), (52, 592), (71, 505), (38, 464), (66, 216), (36, 168), (64, 150), (0, 87), (68, 55)]
- Historical never: 42; rare (<1% positive-match rate): 42.

## Query-row gradient attribution

Values below are median per-query row gradient norms pooled over the fixed 16 windows.

| Component | Current matched q_init | Current unmatched q_init | Historical top10 q_init | Historical never q_init |
|---|---:|---:|---:|---:|
| U_matched_class | 0.1572 | 0.0023 | 0.0473 | 0.0015 |
| U_unmatched_noobject | 0.0164 | 0.0018 | 0.0241 | 0.0017 |
| U_pixel_bce | 0.1095 | 0.000578 | 0.0537 | 0.000531 |
| U_pixel_dice | 0.1340 | 0.000794 | 0.0558 | 0.000654 |
| U_anchor_ce | 0.1212 | 0.000567 | 0.0499 | 0.000469 |
| U_anchor_dice | 0.0259 | 0.000185 | 0.0146 | 0.000148 |
| U_stuff | 0.1257 | 0.0010 | 0.0598 | 0.000841 |
| U_semantic | 0.0772 | 0.000432 | 0.0326 | 0.000402 |
| U_identity | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

| Component | Current matched final_q | Current unmatched final_q | Historical top10 final_q | Historical never final_q |
|---|---:|---:|---:|---:|
| U_matched_class | 0.0180 | 0.0000 | 0.0000 | 0.0000 |
| U_unmatched_noobject | 0.0000 | 1.98e-05 | 0.000559 | 1.89e-05 |
| U_pixel_bce | 0.0054 | 7.6e-07 | 0.000466 | 6.68e-07 |
| U_pixel_dice | 0.0084 | 1.14e-06 | 0.000843 | 1.1e-06 |
| U_anchor_ce | 0.0053 | 8.26e-07 | 0.000948 | 7.65e-07 |
| U_anchor_dice | 0.0018 | 1.92e-07 | 0.000222 | 1.91e-07 |
| U_stuff | 0.0054 | 2.96e-06 | 0.000847 | 2.85e-06 |
| U_semantic | 0.0056 | 1.01e-06 | 0.000475 | 9.49e-07 |
| U_identity | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Directional measurements

| Query group | No-object increase drive | Anchor positive drive | Anchor negative suppression | Anchor net suppressive gradient |
|---|---:|---:|---:|---:|
| current_matched | 0.0000 | 0.0039 | 0.0030 | -3.31e-05 |
| current_unmatched | 0.0000 | 0.0000 | 1.49e-06 | 1.49e-06 |
| historical_top10 | 0.0000 | 0.0000 | 0.00059 | 6.47e-05 |
| historical_never | 0.0000 | 0.0000 | 1.43e-06 | 1.43e-06 |
| historical_rare | 0.0000 | 0.0000 | 1.44e-06 | 1.43e-06 |

### Direct answers from the measured rows

- Historical-never query_init largest component: `U_unmatched_noobject` (median row norm 0.0017); its norm-share proxy median is 0.4138.
- Current-unmatched query_init largest component: `U_matched_class` (median row norm 0.0023).
- Historical-never unmatched no-object CE q_init row-norm median 0.0017; its no-object-logit increase drive mean/median/p90 is 0.0000/0.0000/0.0000.
- Historical-never anchor positive drive mean/median/p90 is 0.0000/0.0000/0.0000; negative suppression is 3.68e-06/1.43e-06/1.03e-05. Suppression / anchor-CE q_init row norm = 0.0031.
- Historical-never q_init row medians: matched-class CE 0.0015, stuff 0.000841, semantic 0.000402, identity 0.0000.
- Anchor CE versus unmatched no-object CE q_init median row norm: 0.000469 vs 0.0017.

## Historical-never / rare counterfactual candidate quality

These are displaced-GT global Hungarian candidates; thresholds are the fixed audit thresholds.

| Scope | Removal | Candidate history | Count | Median Δcost | Pixel IoU median | IoU≥.25 | Anchor correct median | Anchor≥.25 | Strong alternate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| train1024 | top5 | historical_never | 1 | 2.0611 | 0.0000 | 0/1 | — | 0/0 | 0/1 |
| train1024 | top10 | historical_never | 92 | 2.2854 | 0.0000 | 0/92 | 0.0000 | 0/79 | 0/92 |
| val32 | top5 | historical_never | 0 | — | — | 0/0 | — | 0/0 | 0/0 |
| val32 | top10 | historical_never | 10 | 0.5986 | 0.0000 | 0/10 | 0.0000 | 0/8 | 0/10 |
| train1024 | top5 | historical_rare | 23 | 3.8231 | 0.0000 | 0/23 | 0.0000 | 0/22 | 0/23 |
| train1024 | top10 | historical_rare | 723 | 1.7751 | 0.0000 | 0/723 | 0.0000 | 0/604 | 0/723 |
| val32 | top5 | historical_rare | 1 | 15.8499 | 0.0000 | 0/1 | 0.0000 | 0/1 | 0/1 |
| val32 | top10 | historical_rare | 36 | 1.2623 | 0.0000 | 0/36 | 0.0000 | 0/33 | 0/36 |

## Winner-removal counterfactuals

Candidate quality is measured on fixed predictions; these are not model performance scores.

### train1024

| Removal | Displaced GT | Local-best Δcost median | Local-best IoU≥.5 | Local-best anchor≥.5 | Local strong | Local never alternate | Local never strong |
|---|---:|---:|---:|---:|---:|---:|---:|
| top5 | 3159 | 4.2067 (p90 23.0600) | 0.0000 | 0.0000 | 0.0000 | 0.000317 | 0/3159 |
| top10 | 3835 | 4.8226 (p90 28.7230) | 0.0000 | 0.0000 | 0.0000 | 0.0042 | 0/3835 |

| Removal | Displaced GT | CF Δcost median | CF IoU≥.5 | CF anchor≥.5 | CF strong | CF never alternate | CF never strong |
|---|---:|---:|---:|---:|---:|---:|---:|
| top5 | 3159 | 4.8353 (p90 23.2025) | 0.0000 | 0.0000 | 0.0000 | 0.000317 | 0/3159 |
| top10 | 3835 | 5.0872 (p90 28.7230) | 0.0000 | 0.0000 | 0.0000 | 0.0240 | 0/3835 |

### val32

| Removal | Displaced GT | Local-best Δcost median | Local-best IoU≥.5 | Local-best anchor≥.5 | Local strong | Local never alternate | Local never strong |
|---|---:|---:|---:|---:|---:|---:|---:|
| top5 | 109 | 3.9762 (p90 16.8638) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/109 |
| top10 | 148 | 3.5524 (p90 18.3567) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/148 |

| Removal | Displaced GT | CF Δcost median | CF IoU≥.5 | CF anchor≥.5 | CF strong | CF never alternate | CF never strong |
|---|---:|---:|---:|---:|---:|---:|---:|
| top5 | 109 | 5.2557 (p90 16.8638) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0/109 |
| top10 | 148 | 3.8667 (p90 18.3567) | 0.0000 | 0.0000 | 0.0000 | 0.0676 | 0/148 |

## Mechanism evidence

- **A_negative_supervision_starvation**: moderate evidence; measured evidence: `{"anchor_negative_suppression_median": 1.4317128602669982e-06, "anchor_net_logit_grad_median": 1.4317128602669982e-06, "anchor_positive_drive_median": 0.0, "anchor_suppression_over_U_anchor_ce_qinit_norm": 0.003052591187794259, "historical_never_U_unmatched_noobject_norm_share_proxy_median": 0.4138261209086911, "historical_never_U_unmatched_noobject_qinit_grad_median": 0.001654358464293182, "noobject_increase_drive_median": 0.0}`.
- **B_winner_monopolization_or_limited_slot_opportunity**: weak/no evidence; measured evidence: `{"candidate_rows": 7251, "historical_never_alternates": 103, "historical_never_strong": 0, "historical_rare_alternates": 783, "historical_rare_strong": 0, "strong_alternates": 0}`.
- **C_representation_level_dead_slots**: strong evidence; measured evidence: `{"aggregate_counterfactual_rows": 7251, "anchor_correct_ge_0_25_count": 0, "historical_never_candidate_rows": 103, "historical_rare_candidate_rows": 783, "pixel_iou_ge_0_25_count": 0, "strong_alternate_fraction": 0.0}`.
Val32 has the same counterfactual direction: neither Top5 nor Top10 displaced-GT alternate crosses pixel IoU≥.25 or anchor-correct fraction≥.25, and no strong alternate occurs. Gradient attribution was only defined on fixed training windows; no val32 gradient inference is made.

## Contracts

| Contract | Status |
|---|---|
| SM-C1 | PASS |
| SM-C2 | PASS |
| SM-C3 | PASS |
| SM-C4 | PASS |
| SM-C5 | PASS |
| SM-C6 | PASS |
| SM-C7 | PASS |
| SM-C8 | PASS |
| SM-C9 | PASS |
| SM-C10 | PASS |
| SM-C11 | PASS |
| SM-C12 | PASS |
| SM-C13 | PASS |
| SM-C14 | PASS |
| SM-C15 | PASS |

The audit identifies the starvation mechanism(s).
No corrective training strategy was implemented or selected.
