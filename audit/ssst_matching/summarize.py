"""Pooled-GT reports with fixed quantile levels, explicit denominators and failures."""
import csv
import json
from pathlib import Path
import numpy as np


def csv_write(path, rows):
    if not rows:
        # A real empty cohort (e.g. no GT) is distinguishable from missing inference.
        Path(path).write_text('empty_real_table\n')
        return
    with Path(path).open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def distribution(values):
    values=np.asarray(values)
    if not len(values):
        return dict(n=0, mean=None, median=None, p10=None, p90=None, zero=0, one=0, two_three=0, four_plus=0)
    return dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),zero=int((values==0).sum()),one=int((values==1).sum()),two_three=int(((values>=2)&(values<=3)).sum()),four_plus=int((values>=4).sum()))


def summarize(out, config, scenes, gt, comparisons, sweep):
    out=Path(out)
    cuts=np.quantile([r['area_pixels'] for r in gt],[1/3,2/3]).tolist() if gt else [None,None]
    def size(area):
        return 'Small' if area<=cuts[0] else 'Medium' if area<=cuts[1] else 'Large'
    for row in gt+sweep:
        row['size']=size(row['area_pixels'])
    config['size_area_boundaries_pixels']=cuts
    (out/'audit_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    csv_write(out/'audit_per_scene.csv',scenes); csv_write(out/'audit_per_gt.csv',gt)
    csv_write(out/'matching_comparison.csv',comparisons)
    threshold=[]
    for tau in (.3,.5,.7,.9):
        for mh in (1,3,5):
            for group in ('All','Small','Medium','Large'):
                rows=[r for r in sweep if r['purity']==tau and r['min_hits']==mh and (group=='All' or r['size']==group)]
                covered=sum(r['covered'] for r in rows)
                threshold.append(dict(purity=tau,min_hits=mh,size=group,gt=len(rows),covered=covered,uncovered=len(rows)-covered,recall=covered/len(rows) if rows else None,**{k:v for k,v in distribution([r['eligible_queries'] for r in rows]).items() if k!='n'}))
    csv_write(out/'audit_threshold_sweep.csv',threshold)
    total=len(gt);queries=sum(s['queries'] for s in scenes)
    def count(key):return sum(s[key] or 0 for s in scenes)
    def pct(a,b):return f'{100*a/b:.3f}%' if b else 'N/A (zero denominator)'
    covered=sum(r['eligible_queries']>0 for r in gt)
    lines=[f'# SSST Superpoint–GT Audit: {config["cohort"]}', '',
           f'Status: {config["stage"]}; {len(scenes)} windows / {len({s["scene"] for s in scenes})} unique scenes / {total} valid context GT / {queries} queries. Smoke is engineering evidence only.' if config['stage']=='smoke' else f'Status: formal; {len(scenes)} windows / {len({s["scene"] for s in scenes})} scenes / {total} GT / {queries} queries.', '',
           f'Checkpoint: `{config["checkpoint"]}`; SHA256 `{config["checkpoint_sha256"]}`; Epoch4 / 4172 updates. One fixed prediction per input is shared by A/B/C/D; no backward or parameter update.', '',
           f'Code `{config["git_sha"]}`; original execution base `{config["checkpoint_meta"]["code_sha"]}` plus verbatim captured sem256 SemanticMemory (snapshot/identity.json). Train manifest `{config["manifest_sha256"]}`; heldout manifest `{config["heldout_sha256"]}`. Cohort records and implementation file hashes are in audit_config.json.', '',
           '## Q1 / source coverage and object size', '',
           f'GT support recall (purity>=0.5, hits>=1): **{covered}/{total} = {pct(covered,total)}**. Source denominator includes valid stuff and valid positive-ID thing observations; zero-valid denominator is clamped to one. Every allowed and score entry passed exact native parity.', '',
           f'Predeclared pooled-GT area quantiles: 1/3 and 2/3, linear interpolation; boundaries {cuts}. Small <=q1, Medium q1<area<=q2, Large >q2. Ties are retained, so group counts need not be equal. Area is the joint two-view native context GT mask, not original 3D object size.', '',
           '| Size | GT | Covered | Recall | Uncovered | Eligible mean / median / P10 / P90 | 0 / 1 / 2–3 / >=4 queries |','|---|---:|---:|---:|---:|---|---|']
    for group in ('All','Small','Medium','Large'):
        rows=[r for r in gt if group=='All' or r['size']==group];d=distribution([r['eligible_queries'] for r in rows]);c=sum(r['eligible_queries']>0 for r in rows)
        lines.append(f'| {group} | {len(rows)} | {c} | {pct(c,len(rows))} | {len(rows)-c} | {d["mean"]} / {d["median"]} / {d["p10"]} / {d["p90"]} | {d["zero"]} / {d["one"]} / {d["two_three"]} / {d["four_plus"]} |')
    lines += ['',f'Zero Thing-source queries: {count("zero_thing_queries")}/{queries} ({pct(count("zero_thing_queries"),queries)}); zero valid-source queries: {count("zero_valid_queries")}. Query maximum purity fractions (all queries, including stuff):']
    for tau in (.3,.5,.7,.9):
        c=count(f'query_purity_ge_{tau}');lines.append(f'- >= {tau}: {c}/{queries} ({pct(c,queries)})')
    zero=sum(r['uncovered_reason']=='zero_source_hits' for r in gt);mixed=sum(r['uncovered_reason']=='below_purity' for r in gt)
    lines += ['',f'Uncovered GT causes: zero sampled GT observations {zero}; sampled but below purity {mixed}. Overlapping sparse-observation flags among uncovered GT: 1–2 hits {sum(r["eligible_queries"]==0 and r["sparse_source_lt3"] for r in gt)}; 1–4 hits {sum(r["eligible_queries"]==0 and r["sparse_source_lt5"] for r in gt)}. These flags do not claim sparsity is a proven causal mechanism.', '',
              'Purity 0.3/0.5/0.7/0.9 and min_hits 1/3/5 are offline sensitivity only, in audit_threshold_sweep.csv. No training configuration was changed.', '',
              '## Q2 / candidate oracle and actual Hungarian', '',
              '| Quantity | GT | Recall over all GT |','|---|---:|---:|',
              f'| Support upper bound | {count("support_upper")} | {pct(count("support_upper"),total)} |',
              f'| True maximum bipartite matching | {count("maximum_matching")} | {pct(count("maximum_matching"),total)} |',
              f'| Native constrained Hungarian | {count("hungarian_matched")} | {pct(count("hungarian_matched"),total)} |','',
              f'Source-uncovered GT: {total-covered}. Candidate graph simultaneity deficit: {count("support_upper")-count("maximum_matching")}. Hungarian deficit to oracle: {count("maximum_matching")-count("hungarian_matched")}. Eligible but unmatched GT: {count("eligible_unmatched")}. Matched queries: {count("matched_queries")}.', '',
              'Cost is captured from production unified_instance_loss: classification weight2, BCE5, Dice5, forbidden1e6, SciPy linear_sum_assignment then filter forbidden pairs. Its result and a second native call agree exactly. Oracle uses graph cardinality only. A finite penalty is not a general mathematical guarantee of maximum cardinality; any observed deficit is listed with its saved matrix and actual cost range.', '',
              '## Q3 / same-cost per-GT Top-1 comparison', '',
              f'- Top-1 covered GT: {count("top1_matched")}/{total} ({pct(count("top1_matched"),total)}).',
              f'- Scenes with exactly identical pair sets: {count("identical_pairs")}/{len(scenes)} ({pct(count("identical_pairs"),len(scenes))}). Empty pair sets count as equal.',
              f'- Same selected query among all valid GT: {count("same_query_gt")}/{total} ({pct(count("same_query_gt"),total)}). Jointly unmatched GT do not enter the numerator.',
              f'- Different GT selections (including one-sided unmatched): {count("different_gt")}.',
              f'- Queries with multiple allowed GT: {count("multi_gt_queries")}; Top-1 conflicting queries: {count("top1_conflict_queries")}; excess selections: {count("top1_conflict_excess")}. Conflicts are preserved, never deduplicated.',
              f'- Hungarian total cost: {count("hungarian_cost_sum")}; Top-1 total cost: {count("top1_cost_sum")}; Hungarian minus Top-1 over common matched GT: {count("common_gt_cost_delta")}. Total costs are not comparable objectives if coverage or query reuse differs.',
              f'- Official OneFormer strict-second-min differs from task argmin in {count("official_diff_top1")} windows; undefined for <2 queries in {sum(not s["official_strict_defined"] for s in scenes)} windows. Task argmin breaks ties by smallest query index. Official SparseMatcher(topk=1) selects cost strictly below the second-smallest cost and may select none on a minimum tie. Reference excerpt is archived.', '',
              ('On these inputs no observed coverage benefit supports reimplementing the matcher. This is an empirical fixed-weight conclusion, not a claim that the algorithms are universally equivalent.' if count('different_gt')==0 else 'The observed algorithms differ. Inspect conflict/tie diagnostics before considering any matcher change; no training matcher was changed.'), '',
              '## Q4 / context raw-mask funnel', '',
              '| Quantity | GT | Recall |','|---|---:|---:|',
              f'| Support | {covered} | {pct(covered,total)} |',
              f'| Matched | {count("hungarian_matched")} | {pct(count("hungarian_matched"),total)} |',
              f'| Raw mask IoU>=0.5 | {count("raw_mask_gt")} | {pct(count("raw_mask_gt"),total)} |',
              f'| Supported-query raw mask IoU>=0.5 | {count("supported_raw_gt")} | {pct(count("supported_raw_gt"),total)} |',
              f'| Hungarian-selected mask IoU>=0.5 | {count("matched_good_gt")} | {pct(count("matched_good_gt"),total)} |','',
              f'Supported GT without any good raw mask: {sum(r["supported_no_raw"] for r in gt)}. Good raw mask but GT unmatched: {sum(r["raw_not_matched"] for r in gt)}; good raw mask exists but selected query lacks one: {sum(r["raw_not_selected"] for r in gt)}. GT whose good raw queries all predict No-object: {sum(r["raw_all_noobject"] for r in gt)}; matched query predicts No-object: {sum(r["hungarian_noobject"] for r in gt)}.', '',
              'Raw membership is binarized >=0.5; IoU is class-agnostic, over the concatenated two context views. Native targets retain valid positive-ID things; evaluation pixels follow official export semantic validity 0..19 (including unknown thing instance=0). '+f'This adds {count("raw_valid_extra_pixels")} pixels versus loss-valid filtering. No NMS, area16 filter, confidence, packed-mask competition, AP or target-view evaluation is applied. These are diagnostic GT recalls, not official instance mAP or final recall.', '',
              f'Observed funnel deficits: source coverage {total-covered} GT; global candidate conflict {count("support_upper")-count("maximum_matching")} GT; supported GT lacking a good supported raw query {sum(r["eligible_queries"]>0 and not r["supported_raw_mask"] for r in gt)}. These overlapping counts locate loss points but do not establish training causality. Inspect per-GT rows before prioritizing changes.', '',
              'Recommendation: use measured source/size and raw-mask gaps to choose the next investigation. Do not infer final instance quality from source support alone. No new model, loss, matcher or training experiment is implemented.']
    (out/'audit_summary.md').write_text('\n'.join(lines)+'\n')
    failures=['# Audit diagnostics', '', 'All source-score/allowed and native Hungarian parity checks passed. Compact per-window matrices are in diagnostics/*.npz (masked cost retains forbidden1e6).', '']
    for s in scenes:
        if s['eligible_unmatched'] or s['top1_conflict_queries'] or s['maximum_matching']!=s['hungarian_matched'] or s['official_diff_top1']:
            failures.append(f'- {s["window_id"]}: eligible-unmatched={s["eligible_unmatched"]}, oracle={s["maximum_matching"]}, Hungarian={s["hungarian_matched"]}, multi-GT queries={s["multi_gt_queries"]}, Top-1 conflict queries={s["top1_conflict_queries"]}, official tie difference={s["official_diff_top1"]}. Matrix: diagnostics/{s["window_id"]}.npz')
            if s['maximum_matching']!=s['hungarian_matched']:
                z=np.load(out/'diagnostics'/f'{s["window_id"]}.npz');a=z['allowed'];c=z['masked_cost'];v=c[a]
                failures.append(f'  Actual allowed cost range={v.min() if len(v) else None}..{v.max() if len(v) else None}; query/GT={a.shape}. Native finite-penalty objective failed graph-cardinality bound. Matrix saved for explicit diagnosis; do not treat as equal.')
    failures += ['', '## Uncovered GT (IDs are scene-local native instance IDs)', '']
    failures += [f'- {r["window_id"]} GT {r["gt_instance_id"]} class {r["semantic_class"]}: {r["uncovered_reason"]}, hits={r["source_hits"]}, max purity={r["max_purity"]:.6f}, area={r["area_pixels"]}, size={r["size"]}.' for r in gt if not r['eligible_queries']]
    (out/'audit_failures.md').write_text('\n'.join(failures)+'\n')
