# SSST Superpoint–GT Audit: smoke8

Status: smoke; 8 windows / 8 unique scenes / 37 valid context GT / 575 queries. Smoke is engineering evidence only.

Checkpoint: `/space/mawb/ssst/group_plus/semantic_resolution_v1/attempt01_A_sem256_full_node17_8gpu/checkpoint_epoch4.pt`; SHA256 `18aee68f2bc8cdb1d154c742697ccc54657f38ad02c6151bf993405a455f4560`; Epoch4 / 4172 updates. One fixed prediction per input is shared by A/B/C/D; no backward or parameter update.

Code `b7a775f19e232ea2395c8611a47c81b1365fb86e`; original execution base `b7a775f19e232ea2395c8611a47c81b1365fb86e` plus verbatim captured sem256 SemanticMemory (snapshot/identity.json). Train manifest `87e3cde298569c4e7735cf75ee596868eadf01e9a6a95438c6be51e42f55a523`; heldout manifest `9a310d764358ebd7cdd3d1844a48685264d25d48cced5d4263ef4da68c239a44`. Cohort records and implementation file hashes are in audit_config.json.

## Q1 / source coverage and object size

GT support recall (purity>=0.5, hits>=1): **37/37 = 100.000%**. Source denominator includes valid stuff and valid positive-ID thing observations; zero-valid denominator is clamped to one. Every allowed and score entry passed exact native parity.

Predeclared pooled-GT area quantiles: 1/3 and 2/3, linear interpolation; boundaries [2570.0, 13130.0]. Small <=q1, Medium q1<area<=q2, Large >q2. Ties are retained, so group counts need not be equal. Area is the joint two-view native context GT mask, not original 3D object size.

| Size | GT | Covered | Recall | Uncovered | Eligible mean / median / P10 / P90 | 0 / 1 / 2–3 / >=4 queries |
|---|---:|---:|---:|---:|---|---|
| All | 37 | 37 | 100.000% | 0 | 7.891891891891892 / 4.0 / 1.0 / 16.19999999999999 | 0 / 7 / 8 / 22 |
| Small | 13 | 13 | 100.000% | 0 | 2.076923076923077 / 2.0 / 1.0 / 3.8000000000000007 | 0 / 5 / 6 / 2 |
| Medium | 12 | 12 | 100.000% | 0 | 5.25 / 4.5 / 1.1 / 12.500000000000002 | 0 / 2 / 2 / 8 |
| Large | 12 | 12 | 100.000% | 0 | 16.833333333333332 / 11.0 / 7.0 / 33.2 | 0 / 0 / 0 / 12 |

Zero Thing-source queries: 235/575 (40.870%); zero valid-source queries: 141. Query maximum purity fractions (all queries, including stuff):
- >= 0.3: 303/575 (52.696%)
- >= 0.5: 290/575 (50.435%)
- >= 0.7: 264/575 (45.913%)
- >= 0.9: 223/575 (38.783%)

Uncovered GT causes: zero sampled GT observations 0; sampled but below purity 0. Overlapping sparse-observation flags among uncovered GT: 1–2 hits 0; 1–4 hits 0. These flags do not claim sparsity is a proven causal mechanism.

Purity 0.3/0.5/0.7/0.9 and min_hits 1/3/5 are offline sensitivity only, in audit_threshold_sweep.csv. No training configuration was changed.

## Q2 / candidate oracle and actual Hungarian

| Quantity | GT | Recall over all GT |
|---|---:|---:|
| Support upper bound | 37 | 100.000% |
| True maximum bipartite matching | 37 | 100.000% |
| Native constrained Hungarian | 37 | 100.000% |

Source-uncovered GT: 0. Candidate graph simultaneity deficit: 0. Hungarian deficit to oracle: 0. Eligible but unmatched GT: 0. Matched queries: 37.

Cost is captured from production unified_instance_loss: classification weight2, BCE5, Dice5, forbidden1e6, SciPy linear_sum_assignment then filter forbidden pairs. Its result and a second native call agree exactly. Oracle uses graph cardinality only. A finite penalty is not a general mathematical guarantee of maximum cardinality; any observed deficit is listed with its saved matrix and actual cost range.

## Q3 / same-cost per-GT Top-1 comparison

- Top-1 covered GT: 37/37 (100.000%).
- Scenes with exactly identical pair sets: 8/8 (100.000%). Empty pair sets count as equal.
- Same selected query among all valid GT: 37/37 (100.000%). Jointly unmatched GT do not enter the numerator.
- Different GT selections (including one-sided unmatched): 0.
- Queries with multiple allowed GT: 2; Top-1 conflicting queries: 0; excess selections: 0. Conflicts are preserved, never deduplicated.
- Hungarian total cost: 8.250899195671082; Top-1 total cost: 8.250899195671082; Hungarian minus Top-1 over common matched GT: 0. Total costs are not comparable objectives if coverage or query reuse differs.
- Official OneFormer strict-second-min differs from task argmin in 0 windows; undefined for <2 queries in 0 windows. Task argmin breaks ties by smallest query index. Official SparseMatcher(topk=1) selects cost strictly below the second-smallest cost and may select none on a minimum tie. Reference excerpt is archived.

On these inputs no observed coverage benefit supports reimplementing the matcher. This is an empirical fixed-weight conclusion, not a claim that the algorithms are universally equivalent.

## Q4 / context raw-mask funnel

| Quantity | GT | Recall |
|---|---:|---:|
| Support | 37 | 100.000% |
| Matched | 37 | 100.000% |
| Raw mask IoU>=0.5 | 31 | 83.784% |
| Supported-query raw mask IoU>=0.5 | 29 | 78.378% |
| Hungarian-selected mask IoU>=0.5 | 28 | 75.676% |

Supported GT without any good raw mask: 6. Good raw mask but GT unmatched: 0; good raw mask exists but selected query lacks one: 3. GT whose good raw queries all predict No-object: 0; matched query predicts No-object: 0.

Raw membership is binarized >=0.5; IoU is class-agnostic, over the concatenated two context views. Native targets retain valid positive-ID things; evaluation pixels follow official export semantic validity 0..19 (including unknown thing instance=0). This adds 0 pixels versus loss-valid filtering. No NMS, area16 filter, confidence, packed-mask competition, AP or target-view evaluation is applied. These are diagnostic GT recalls, not official instance mAP or final recall.

Observed funnel deficits: source coverage 0 GT; global candidate conflict 0 GT; supported GT lacking a good supported raw query 8. These overlapping counts locate loss points but do not establish training causality. Inspect per-GT rows before prioritizing changes.

Recommendation: use measured source/size and raw-mask gaps to choose the next investigation. Do not infer final instance quality from source support alone. No new model, loss, matcher or training experiment is implemented.
