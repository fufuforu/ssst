"""Summarize the authorized two-recipe comparison; never infer single-variable causality."""
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]);labels=['old_A4_plus4_A8','single16_epoch8_L6L12']
reports={};exports={};configs={}
for label in labels:
    d=root/label
    reports[label]=json.loads((d/'evaluation/metrics.json').read_text())
    exports[label]=json.loads((d/'evaluation/export_complete.json').read_text())
    configs[label]=json.loads((d/'configuration.json').read_text())
    assert reports[label]['status']=='OFFICIAL_VAL_SCORED' and reports[label]['windows']==1860
    assert json.loads((d/'COMPLETE.json').read_text())['windows']==1860
assert reports[labels[0]]['official_commit']==reports[labels[1]]['official_commit']
def signatures(export):
    return [(x['name'],x['source_index'],x['frame_ids'],x['input_rgb_sha256']) for x in export['records']]
assert signatures(exports[labels[0]])==signatures(exports[labels[1]]),'Inputs differ'
def gt(d):
    return {x['name']:x for x in map(json.loads,(d/'evaluation/gt_reference_manifest.jsonl').read_text().splitlines())}
assert gt(root/labels[0])==gt(root/labels[1]),'GT differs'
fields=['AbsRel','RMSE','PSNR','SSIM','LPIPS','ctx_mIoU','ctx_mAP','ctx_PQ','tgt_mIoU','tgt_mAP','tgt_PQ']
def values(report):
    raw=report['official_native_result']
    return [raw[k] for k in ('absrel','rmse','psnr','ssim','lpips')]+[raw[v+'_miou'] if m=='mIoU' else raw[v+'_map']['map'] if m=='mAP' else raw[v+'_pq'] for v in ('context','target') for m in ('mIoU','mAP','PQ')]
a,b=[values(reports[k]) for k in labels];delta=[y-x for x,y in zip(a,b)]
summary=root/'comparison';summary.mkdir(exist_ok=True)
with (summary/'comparison_native_units.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['recipe']+fields)
    for label,val in [(labels[0],a),(labels[1],b),('single_minus_old',delta)]:w.writerow([label]+val)
lines=['# A8 official 1860 comparison','', 'Both checkpoints: epoch8, 8344 updates. Different LR schedules, supervision layers and native mask resolution; recipe comparison, not a single-variable test.','', '| Metric | Old A4+4 A8 | Single16 epoch8 | Single − old |','|---|---:|---:|---:|']
for i,(field,x,y,d) in enumerate(zip(fields,a,b,delta)):
    scale=100 if i>=5 else 1
    suffix=' (%) / delta pp' if i>=5 else ''
    lines.append(f'| {field}{suffix} | {x*scale:.5f} | {y*scale:.5f} | {d*scale:+.5f} |')
lines+=['','All 1860 input identities and GT hashes match. Target follows the official six-frame scope. Small PQ differences require per-class and TP/FP/FN analysis; no capacity conclusion is automatic.']
(summary/'comparison.md').write_text('\n'.join(lines)+'\n')
perclass={k:{key:value for key,value in reports[k]['official_native_result'].items() if 'per_class' in key or key.endswith('_map')} for k in labels}
(summary/'per_class_native.json').write_text(json.dumps(perclass,indent=2,allow_nan=False)+'\n')
(summary/'COMPLETE.json').write_text(json.dumps(dict(status='A8_OFFICIAL1860_COMPARISON_COMPLETE',windows=1860,matched_inputs=True,matched_gt=True,checkpoints=configs,delta_native_units=dict(zip(fields,delta))),indent=2)+'\n')
print((summary/'comparison.md').read_text())
