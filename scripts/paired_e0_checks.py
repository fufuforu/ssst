"""Strict same-forward controls and observational historic prediction audits."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def capture_native(out):
    from scripts.export_object_locus_v3_set_official import assemble_panoptic
    sem, ins, _ = assemble_panoptic(out)
    official = torch.where((sem >= 0) & (sem <= 19), sem + 1, 0)
    p = out['p_class'][0]
    keep = (p.argmax(-1) != 18) & (p[:, :18].max(-1).values >= .05)
    if 'instance_eligible' in out:
        keep = keep & out['instance_eligible']
    return dict(packed=(official * 1000 + ins).cpu().numpy(), keep=keep.clone(),
                versions={k: (out[k].data_ptr(), out[k]._version) for k in
                          ('region_mass', 'p_class', 'alpha', 'semantic_scores', 'instance_eligible') if k in out})


def verify_unmodified(out, control, e0, keep):
    if e0 is not out or not torch.equal(keep, control['keep']):
        raise ValueError('E0 changed native prediction or candidate eligibility')
    for key, identity in control['versions'].items():
        if (out[key].data_ptr(), out[key]._version) != identity:
            raise ValueError('Oracle changed native control tensor: ' + key)


def decode(path):
    pixels = np.asarray(Image.open(path), dtype=np.int64)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError('Packed segmentation must be RGB: ' + str(path))
    return pixels[..., 0] + 256*pixels[..., 1] + 65536*pixels[..., 2]


def verify_gt_files(pair, historical):
    for split in ('context', 'target'):
        current = sorted((Path(pair)/(split+'_seg_gt')).glob('*.png'))
        reference = sorted((Path(historical)/(split+'_seg_gt')).glob('*.png'))
        if not current or [p.name for p in current] != [p.name for p in reference]:
            raise ValueError('GT file set differs from historical input contract')
        if any(p.read_bytes() != q.read_bytes() for p,q in zip(current, reference)):
            raise ValueError('GT differs from historical input contract')


def audit_pair(pair, historical, expected, frames, scene):
    """Any native mismatch fails; historic differences are measured, never hidden."""
    pair, historical = Path(pair), Path(historical)
    audit = dict(native_control_verified=True, checked_pngs=0, historical_byte_differing_pngs=0,
                 historical_differing_pngs=0, historical_differing_pixels=0, compared_pixels=0,
                 historical_semantic_differing_pixels=0, historical_instance_differing_pixels=0,
                 historical_max_score_difference=0., historical_id_changes=[], historical_label_changes=[], details=[])
    for split, indices in [('context', range(2)), ('target', range(len(frames)))]:
        sub = split + '_seg_pred'
        for i in indices:
            name = f'{scene}_pred{frames[i]}.png'
            path, reference = pair/sub/name, historical/sub/name
            current = decode(path)
            if not np.array_equal(current, expected[i]):
                raise ValueError('E0 differs from same-forward native assembly: ' + str(path))
            old = decode(reference)
            if old.shape != current.shape:
                raise ValueError('Historical prediction shape differs: ' + str(reference))
            different = current != old
            count = int(different.sum())
            audit['checked_pngs'] += 1
            audit['compared_pixels'] += current.size
            audit['historical_byte_differing_pngs'] += int(path.read_bytes() != reference.read_bytes())
            audit['historical_differing_pngs'] += int(count > 0)
            audit['historical_differing_pixels'] += count
            audit['historical_semantic_differing_pixels'] += int(((current//1000) != (old//1000)).sum())
            audit['historical_instance_differing_pixels'] += int(((current%1000) != (old%1000)).sum())
            if count:
                audit['details'].append(dict(file=sub+'/'+name, differing_pixels=count, total_pixels=current.size))
        current_rows = {r['id']: r for r in json.loads((pair/sub/'pred.json').read_text())}
        old_rows = {r['id']: r for r in json.loads((historical/sub/'pred.json').read_text())}
        if current_rows.keys() != old_rows.keys():
            audit['historical_id_changes'].append(dict(scope=split, added=sorted(current_rows.keys()-old_rows.keys()),
                                                        removed=sorted(old_rows.keys()-current_rows.keys())))
        for q in current_rows.keys() & old_rows.keys():
            if current_rows[q]['label_id'] != old_rows[q]['label_id']:
                audit['historical_label_changes'].append(dict(scope=split, id=q, new=current_rows[q]['label_id'], old=old_rows[q]['label_id']))
            audit['historical_max_score_difference'] = max(audit['historical_max_score_difference'], abs(current_rows[q]['score']-old_rows[q]['score']))
    return audit


def verify_export_control(exported, receipt):
    rows = exported['records']
    if (exported['windows'] != 1860 or exported['unique_scenes'] != 312 or exported['parameter_updates'] != 0
            or len(rows) != 1860 or len({r['name'] for r in rows}) != 1860
            or not all(r.get('native_control_verified') is True for r in rows)
            or receipt.get('status') != 'NATIVE_SAME_FORWARD_E0_VERIFIED'
            or receipt.get('windows') != 1860):
        raise ValueError('Incomplete or invalid same-forward E0 control')
