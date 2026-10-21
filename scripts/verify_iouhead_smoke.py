"""Fail closed before authorizing the dependent full run."""
import json,math,sys
from pathlib import Path
import torch
r=Path(sys.argv[1]);checks={}
for rank in range(8):
    rows=[json.loads(line) for line in (r/f'train_rank{rank:02d}.jsonl').read_text().splitlines()]
    assert [row['update'] for row in rows]==[1,2]
    assert all(math.isfinite(row['loss_iou']) and row['loss_iou']>=0 and 0<=row['pred_iou_min']<=row['pred_iou_max']<=1 for row in rows)
    assert rows[0]['pred_iou_min']==rows[0]['pred_iou_max']==.5
    assert json.loads((r/f'FIRST_GRADIENTS_rank{rank:02d}.json').read_text())['iou_head']>0
assert json.loads((r/'FIRST_UPDATE_VERIFIED.json').read_text())['status']=='PASSED'
assert json.loads((r/'LIMIT_REACHED.json').read_text())['progress']['full_updates']==2
saved=torch.load(r/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
assert saved['progress']['full_updates']==2
assert torch.count_nonzero(saved['model']['core.iou_head.3.weight'])>0
export=json.loads((r/'export_smoke/export_complete.json').read_text())
assert export['status']=='EXPORTED' and export['windows']==1 and len(export['records'])==1
for rec in export['records']:
    m=rec['official_pair']['iou_score_multiplier']
    assert m['enabled'] and 0<=m['minimum']<=m['maximum']<=1
checks=dict(status='PASSED',training_updates=2,ranks=8,first_update_synchronized=True,iou_gradient_nonzero=True,iou_output_weight_updated=True,pred_iou_finite_in_unit_interval=True,export_windows=1,score_multiplier_non_none=True)
(r/'SMOKE_PASSED.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks),flush=True)
