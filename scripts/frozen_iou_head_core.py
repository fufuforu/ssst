"""A separate quality regressor; no backbone parameters are added or changed."""
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
from scripts.frozen_iou_data_plan import SOURCE_SHA,TARGET,scene_split


class FrozenTokenIoUHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(256), nn.Linear(256,256), nn.GELU(), nn.Linear(256,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, updated_tokens):
        if updated_tokens.ndim != 2 or updated_tokens.shape[1] != 256:
            raise ValueError('Expected final updated thing-query tokens [N,256]')
        return self.net(updated_tokens.detach().float()).squeeze(-1).sigmoid()


def context_iou_targets(prediction, semantic, instance):
    from scripts.fixed_query_oracles import context_ranking_scores
    p = prediction['p_class'][0]
    keep = prediction['instance_eligible'] & (p.argmax(-1) != 18) & (p[:, :18].max(-1).values >= .05)
    targets = torch.tensor(context_ranking_scores(prediction, keep, semantic, instance), device=p.device, dtype=torch.float32)
    return targets, keep


def rerank_rows(rows, predicted_iou):
    """Direct predicted IoU replaces score; IDs/classes and masks stay fixed."""
    values = predicted_iou.detach().cpu().tolist() if torch.is_tensor(predicted_iou) else list(predicted_iou)
    if any(not (0 <= x <= 1) for x in values):raise ValueError('Nonfinite/out-of-range predicted IoU')
    result = []
    for row in rows:
        q = int(row['id'])-1
        if not 0 <= q < len(values):raise ValueError('Prediction query ID differs from updated token bank')
        result.append(dict(row, score=float(values[q])))
    return result


def make_reranked_pair(source, destination, predicted_iou):
    source,destination=Path(source),Path(destination)
    for sub in ('rgb','rgb_gt','depth','depth_gt','context_seg_gt','target_seg_gt'):
        (destination/sub).parent.mkdir(parents=True,exist_ok=True)
        (destination/sub).symlink_to((source/sub).resolve(),target_is_directory=True)
    for scope in ('context','target'):
        origin=source/(scope+'_seg_pred');target=destination/(scope+'_seg_pred');target.mkdir(parents=True,exist_ok=True)
        for p in origin.glob('*.png'):os.link(p,target/p.name)
        rows=json.loads((origin/'pred.json').read_text())
        write_json(target/'pred.json',rerank_rows(rows,predicted_iou))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def atomic_save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary);temporary.replace(path)
