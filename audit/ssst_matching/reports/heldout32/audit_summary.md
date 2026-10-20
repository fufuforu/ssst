# SSST Superpoint–GT Audit: heldout32

Status: formal; 32 windows / 32 scenes / 122 GT / 2139 queries.

Checkpoint: `/space/mawb/ssst/group_plus/semantic_resolution_v1/attempt01_A_sem256_full_node17_8gpu/checkpoint_epoch4.pt`; SHA256 `18aee68f2bc8cdb1d154c742697ccc54657f38ad02c6151bf993405a455f4560`; Epoch4 / 4172 updates. One fixed prediction per input is shared by A/B/C/D; no backward or parameter update.

Code `9a949c52b86ac8796f1eb1f4d2965f34347752aa`; original execution base `b7a775f19e232ea2395c8611a47c81b1365fb86e` plus verbatim captured sem256 SemanticMemory (snapshot/identity.json). Train manifest `87e3cde298569c4e7735cf75ee596868eadf01e9a6a95438c6be51e42f55a523`; heldout manifest `9a310d764358ebd7cdd3d1844a48685264d25d48cced5d4263ef4da68c239a44`. Cohort records and implementation file hashes are in audit_config.json.

## Q1 / source coverage and object size

GT support recall (purity>=0.5, hits>=1): **111/122 = 90.984%**. Source denominator includes valid stuff and valid positive-ID thing observations; zero-valid denominator is clamped to one. Every allowed and score entry passed exact native parity.

Predeclared pooled-GT area quantiles: 1/3 and 2/3, linear interpolation; boundaries [2875.666666666665, 12459.33333333333]. Small <=q1, Medium q1<area<=q2, Large >q2. Ties are retained, so group counts need not be equal. Area is the joint two-view native context GT mask, not original 3D object size.

| Size | GT | Covered | Recall | Uncovered | Eligible mean / median / P10 / P90 | 0 / 1 / 2–3 / >=4 queries |
|---|---:|---:|---:|---:|---|---|
| All | 122 | 111 | 90.984% | 11 | 8.754098360655737 / 5.0 / 1.0 / 18.900000000000006 | 11 / 14 / 22 / 75 |
| Small | 41 | 30 | 73.171% | 11 | 1.975609756097561 / 1.0 / 0.0 / 4.0 | 11 / 10 / 10 / 10 |
| Medium | 40 | 40 | 100.000% | 0 | 6.25 / 4.0 / 1.9000000000000004 / 13.100000000000001 | 0 / 4 / 12 / 24 |
| Large | 41 | 41 | 100.000% | 0 | 17.975609756097562 / 13.0 / 5.0 / 33.0 | 0 / 0 / 0 / 41 |

Zero Thing-source queries: 822/2139 (38.429%); zero valid-source queries: 403. Query maximum purity fractions (all queries, including stuff):
- >= 0.3: 1126/2139 (52.641%)
- >= 0.5: 1061/2139 (49.603%)
- >= 0.7: 945/2139 (44.180%)
- >= 0.9: 832/2139 (38.897%)

Uncovered GT causes: zero sampled GT observations 2; sampled but below purity 9. Overlapping sparse-observation flags among uncovered GT: 1–2 hits 1; 1–4 hits 2. These flags do not claim sparsity is a proven causal mechanism.

Purity 0.3/0.5/0.7/0.9 and min_hits 1/3/5 are offline sensitivity only, in audit_threshold_sweep.csv. No training configuration was changed.

## Q2 / candidate oracle and actual Hungarian

| Quantity | GT | Recall over all GT |
|---|---:|---:|
| Support upper bound | 111 | 90.984% |
| True maximum bipartite matching | 111 | 90.984% |
| Native constrained Hungarian | 111 | 90.984% |

Source-uncovered GT: 11. Candidate graph simultaneity deficit: 0. Hungarian deficit to oracle: 0. Eligible but unmatched GT: 0. Matched queries: 111.

Cost is captured from production unified_instance_loss: classification weight2, BCE5, Dice5, forbidden1e6, SciPy linear_sum_assignment then filter forbidden pairs. Its result and a second native call agree exactly. Oracle uses graph cardinality only. A finite penalty is not a general mathematical guarantee of maximum cardinality; any observed deficit is listed with its saved matrix and actual cost range.

## Q3 / same-cost per-GT Top-1 comparison

- Top-1 covered GT: 111/122 (90.984%).
- Scenes with exactly identical pair sets: 32/32 (100.000%). Empty pair sets count as equal.
- Same selected query among all valid GT: 111/122 (90.984%). Jointly unmatched GT do not enter the numerator.
- Different GT selections (including one-sided unmatched): 0.
- Queries with multiple allowed GT: 7; Top-1 conflicting queries: 0; excess selections: 0. Conflicts are preserved, never deduplicated.
- Hungarian total cost: 12.244088113307953; Top-1 total cost: 12.244088113307953; Hungarian minus Top-1 over common matched GT: 0. Total costs are not comparable objectives if coverage or query reuse differs.
- Official OneFormer strict-second-min differs from task argmin in 0 windows; undefined for <2 queries in 0 windows. Task argmin breaks ties by smallest query index. Official SparseMatcher(topk=1) selects cost strictly below the second-smallest cost and may select none on a minimum tie. Reference excerpt is archived.

On these inputs no observed coverage benefit supports reimplementing the matcher. This is an empirical fixed-weight conclusion, not a claim that the algorithms are universally equivalent.

## Q4 / context raw-mask funnel

| Quantity | GT | Recall |
|---|---:|---:|
| Support | 111 | 90.984% |
| Matched | 111 | 90.984% |
| Raw mask IoU>=0.5 | 99 | 81.148% |
| Supported-query raw mask IoU>=0.5 | 93 | 76.230% |
| Hungarian-selected mask IoU>=0.5 | 91 | 74.590% |

Supported GT without any good raw mask: 15. Good raw mask but GT unmatched: 3; good raw mask exists but selected query lacks one: 8. GT whose good raw queries all predict No-object: 0; matched query predicts No-object: 0.

Raw membership is binarized >=0.5; IoU is class-agnostic, over the concatenated two context views. Native targets retain valid positive-ID things; evaluation pixels follow official export semantic validity 0..19 (including unknown thing instance=0). This adds 0 pixels versus loss-valid filtering. No NMS, area16 filter, confidence, packed-mask competition, AP or target-view evaluation is applied. These are diagnostic GT recalls, not official instance mAP or final recall.

Observed funnel deficits: source coverage 11 GT; global candidate conflict 0 GT; supported GT lacking a good supported raw query 18. These overlapping counts locate loss points but do not establish training causality. Inspect per-GT rows before prioritizing changes.

Recommendation: use measured source/size and raw-mask gaps to choose the next investigation. Do not infer final instance quality from source support alone. No new model, loss, matcher or training experiment is implemented.
