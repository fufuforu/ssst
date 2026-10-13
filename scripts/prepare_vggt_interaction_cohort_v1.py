"""Freeze score-independent scene/window subsets from the existing manifests."""
import hashlib
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json')
MANIFEST_SHA = '87e3cde298569c4e7735cf75ee596868eadf01e9a6a95438c6be51e42f55a523'
VALID_SHA = '59cf5594ec2f3223a41d27d7af4da49d348ce9b00c1d151020378c9a8f4b4b3b'
COHORT = REPO/'docs/vggt_interaction_cohort_v1.json'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def make():
    if sha(MANIFEST) != MANIFEST_SHA: raise ValueError('train manifest changed')
    m = json.loads(MANIFEST.read_text())
    train = m.get('expanded_train_windows') or m.get('train_windows') or m['windows']
    valid_path = Path(m['full_validation_source'])
    if sha(valid_path) != VALID_SHA: raise ValueError('official validation changed')
    valid = []
    for row in json.loads(valid_path.read_text()):
        ctx = list(map(int, row['context_ids']))
        valid.append(dict(scene=row['scan'], context=ctx,
                          novel=[int(x) for x in row['target_ids'] if int(x) not in ctx], pair_iou=row.get('iou')))
    if len(train) != 8337 or len(valid) != 1860: raise ValueError('source counts differ')
    output = dict(selection='seed42, without replacement; no loss/score filtering',
                  train_manifest_sha256=MANIFEST_SHA, validation_manifest_sha256=VALID_SHA)
    for split, windows, count, per_scene in [('train', train, 128, 4), ('heldout', valid, 64, 1)]:
        rng = np.random.default_rng(42)
        selected = []
        for scene in rng.choice(sorted({w['scene'] for w in windows}), count, replace=False):
            candidates = [i for i, w in enumerate(windows) if w['scene'] == scene]
            for i in rng.choice(candidates, per_scene, replace=False):
                selected.append(dict(windows[int(i)], split=split, source_index=int(i)))
        output[split] = selected
    if {w['scene'] for w in output['train']} & {w['scene'] for w in output['heldout']}:
        raise ValueError('train/validation overlap')
    return output

if __name__ == '__main__':
    content = json.dumps(make(), indent=2, allow_nan=False)+'\n'
    if COHORT.exists() and COHORT.read_text() != content: raise ValueError('preserve frozen cohort')
    COHORT.write_text(content)
    print(str(COHORT), sha(COHORT))
