# SSST Superpoint–GT Audit: heldout128

Status: formal; 128 windows / 128 scenes / 492 GT / 7844 queries.

Checkpoint: `/space/mawb/ssst/group_plus/semantic_resolution_v1/attempt01_A_sem256_full_node17_8gpu/checkpoint_epoch4.pt`; SHA256 `18aee68f2bc8cdb1d154c742697ccc54657f38ad02c6151bf993405a455f4560`; Epoch4 / 4172 updates. One fixed prediction per input is shared by A/B/C/D; no backward or parameter update.

Code `9a949c52b86ac8796f1eb1f4d2965f34347752aa`; original execution base `b7a775f19e232ea2395c8611a47c81b1365fb86e` plus verbatim captured sem256 SemanticMemory (snapshot/identity.json). Train manifest `87e3cde298569c4e7735cf75ee596868eadf01e9a6a95438c6be51e42f55a523`; heldout manifest `9a310d764358ebd7cdd3d1844a48685264d25d48cced5d4263ef4da68c239a44`. Cohort records and implementation file hashes are in audit_config.json.

## Q1 / source coverage and object size

GT support recall (purity>=0.5, hits>=1): **427/492 = 86.789%**. Source denominator includes valid stuff and valid positive-ID thing observations; zero-valid denominator is clamped to one. Every allowed and score entry passed exact native parity.

Predeclared pooled-GT area quantiles: 1/3 and 2/3, linear interpolation; boundaries [2382.0, 11130.666666666664]. Small <=q1, Medium q1<area<=q2, Large >q2. Ties are retained, so group counts need not be equal. Area is the joint two-view native context GT mask, not original 3D object size.

| Size | GT | Covered | Recall | Uncovered | Eligible mean / median / P10 / P90 | 0 / 1 / 2–3 / >=4 queries |
|---|---:|---:|---:|---:|---|---|
| All | 492 | 427 | 86.789% | 65 | 7.453252032520325 / 4.0 / 0.0 / 17.0 | 65 / 57 / 105 / 265 |
| Small | 164 | 106 | 64.634% | 58 | 1.5304878048780488 / 1.0 / 0.0 / 4.0 | 58 / 38 / 50 / 18 |
| Medium | 164 | 157 | 95.732% | 7 | 5.036585365853658 / 4.0 / 1.0 / 10.0 | 7 / 17 / 46 / 94 |
| Large | 164 | 164 | 100.000% | 0 | 15.792682926829269 / 11.0 / 4.0 / 32.0 | 0 / 2 / 9 / 153 |

Zero Thing-source queries: 3150/7844 (40.158%); zero valid-source queries: 1507. Query maximum purity fractions (all queries, including stuff):
- >= 0.3: 3904/7844 (49.771%)
- >= 0.5: 3652/7844 (46.558%)
- >= 0.7: 3290/7844 (41.943%)
- >= 0.9: 2907/7844 (37.060%)

Uncovered GT causes: zero sampled GT observations 16; sampled but below purity 49. Overlapping sparse-observation flags among uncovered GT: 1–2 hits 9; 1–4 hits 16. These flags do not claim sparsity is a proven causal mechanism.

Purity 0.3/0.5/0.7/0.9 and min_hits 1/3/5 are offline sensitivity only, in audit_threshold_sweep.csv. No training configuration was changed.

## Q2 / candidate oracle and actual Hungarian

| Quantity | GT | Recall over all GT |
|---|---:|---:|
| Support upper bound | 427 | 86.789% |
| True maximum bipartite matching | 427 | 86.789% |
| Native constrained Hungarian | 427 | 86.789% |

Source-uncovered GT: 65. Candidate graph simultaneity deficit: 0. Hungarian deficit to oracle: 0. Eligible but unmatched GT: 0. Matched queries: 427.

Cost is captured from production unified_instance_loss: classification weight2, BCE5, Dice5, forbidden1e6, SciPy linear_sum_assignment then filter forbidden pairs. Its result and a second native call agree exactly. Oracle uses graph cardinality only. A finite penalty is not a general mathematical guarantee of maximum cardinality; any observed deficit is listed with its saved matrix and actual cost range.

## Q3 / same-cost per-GT Top-1 comparison

- Top-1 covered GT: 427/492 (86.789%).
- Scenes with exactly identical pair sets: 127/128 (99.219%). Empty pair sets count as equal.
- Same selected query among all valid GT: 426/492 (86.585%). Jointly unmatched GT do not enter the numerator.
- Different GT selections (including one-sided unmatched): 1.
- Queries with multiple allowed GT: 15; Top-1 conflicting queries: 1; excess selections: 1. Conflicts are preserved, never deduplicated.
- Hungarian total cost: 26.763007193803787; Top-1 total cost: 26.464195042848587; Hungarian minus Top-1 over common matched GT: 0.2988121509552002. Total costs are not comparable objectives if coverage or query reuse differs.
- Official OneFormer strict-second-min differs from task argmin in 0 windows; undefined for <2 queries in 0 windows. Task argmin breaks ties by smallest query index. Official SparseMatcher(topk=1) selects cost strictly below the second-smallest cost and may select none on a minimum tie. Reference excerpt is archived.

The observed algorithms differ. Inspect conflict/tie diagnostics before considering any matcher change; no training matcher was changed.

## Q4 / context raw-mask funnel

| Quantity | GT | Recall |
|---|---:|---:|
| Support | 427 | 86.789% |
| Matched | 427 | 86.789% |
| Raw mask IoU>=0.5 | 369 | 75.000% |
| Supported-query raw mask IoU>=0.5 | 353 | 71.748% |
| Hungarian-selected mask IoU>=0.5 | 348 | 70.732% |

Supported GT without any good raw mask: 67. Good raw mask but GT unmatched: 9; good raw mask exists but selected query lacks one: 21. GT whose good raw queries all predict No-object: 1; matched query predicts No-object: 1.

Raw membership is binarized >=0.5; IoU is class-agnostic, over the concatenated two context views. Native targets retain valid positive-ID things; evaluation pixels follow official export semantic validity 0..19 (including unknown thing instance=0). This adds 0 pixels versus loss-valid filtering. No NMS, area16 filter, confidence, packed-mask competition, AP or target-view evaluation is applied. These are diagnostic GT recalls, not official instance mAP or final recall.

Observed funnel deficits: source coverage 65 GT; global candidate conflict 0 GT; supported GT lacking a good supported raw query 74. These overlapping counts locate loss points but do not establish training causality. Inspect per-GT rows before prioritizing changes.

Recommendation: use measured source/size and raw-mask gaps to choose the next investigation. Do not infer final instance quality from source support alone. No new model, loss, matcher or training experiment is implemented.
