"""Prepare isolated runs without submitting jobs or copying large weights."""
import hashlib,json,os,subprocess
from pathlib import Path
import torch
root=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/iouhead_20261020_v1')
zero=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_untrained_initial.pt')
sha='3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744'
cohort=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json')
h=hashlib.sha256()
with zero.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==sha
saved=torch.load(zero,map_location='cpu',weights_only=False,mmap=True)
assert saved['progress']==dict(epoch=0,update_in_epoch=0,full_updates=0) and not saved['optimizer']['state']
assert saved['meta']['world_size']==len(saved['rank_rng'])==8
assert not any(k.startswith('core.iou_head.') for k in saved['model'])
records=json.loads(cohort.read_text())['records'];assert len(records)==1860
assert len({(w['scene'],tuple(w['context'])) for w in records})==1860
assert len({w['scene'] for w in records})==312
snapshot=Path(__file__).resolve().parents[1]
code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=snapshot,text=True).strip()
assert not root.exists(), 'Refusing to reuse an existing run bundle'
for name in ('smoke','train16'):
    run=root/name;run.mkdir(parents=True)
    os.link(zero,run/'checkpoint_untrained_initial.pt')
    (run/'untrained_initial.sha256').write_text(sha+'\n')
    config=dict(recipe='iouhead_'+name+'_20261020_v1',architecture=saved['meta']['architecture'],supervised_layers=[6,12],layer_weights=[1/3,2/3],epochs=16,global_batch=8,world_size=8,learning_rate=1e-4,warmup_updates=100,understanding_weight=.05,understanding_warmup=1043,iou_head=True,iou_weight=.5,initial_checkpoint=str(zero),initial_sha256=sha,code_snapshot=str(snapshot),code_sha=code_sha,checkpoint_retention='initial + latest; epoch probes retained',max_updates=2 if name=='smoke' else None)
    (run/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
evaldir=root/'train16/eval1860';evaldir.mkdir()
(evaldir/'configuration.json').write_text(json.dumps(dict(recipe='iouhead_train16_official1860_20261020_v1',architecture=saved['meta']['architecture'],windows_expected=1860,cohort=str(cohort),cohort_sha256=hashlib.sha256(cohort.read_bytes()).hexdigest(),code_snapshot=str(snapshot),code_sha=code_sha),indent=2)+'\n')
(root/'PREFLIGHT.json').write_text(json.dumps(dict(status='PASSED',zero_sha256=sha,zero_world_size=8,zero_updates=0,empty_adam=True,legacy_parameter_count=len(saved['model']),official_windows=1860,official_scenes=312,code_snapshot=str(snapshot),code_sha=code_sha),indent=2)+'\n')
print((root/'PREFLIGHT.json').read_text())
