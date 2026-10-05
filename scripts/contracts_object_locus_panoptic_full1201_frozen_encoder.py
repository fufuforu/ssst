"""CPU contracts for the frozen-encoder full1201 comparison."""
import json,hashlib
from pathlib import Path
import torch
from scripts.object_locus_panoptic_full1201_frozen_encoder_runtime import *

def state_digest(model):
 h=hashlib.sha256()
 for key,value in sorted(model.state_dict().items()):
  h.update(key.encode());h.update(str(tuple(value.shape)).encode());h.update(str(value.dtype).encode())
  h.update(memoryview(value.detach().cpu().contiguous().numpy()))
 return h.hexdigest()

def optimizer_map(optimizer):
 result={}
 for group in optimizer.param_groups:
  for name in group['param_names']:
   if name in result: raise AssertionError('duplicate optimizer parameter '+name)
   result[name]=(group['name'],group['peak_lr'],group['weight_decay'])
 return result

def main():
 configure()
 source=Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu')
 for name in ('manifest.json','training_plan.json','weights_provenance.json','asset_hashes.json'):
  if sha(REPORT/name)!=sha(source/name): raise AssertionError('control asset SHA mismatch: '+name)
 manifest,plan,hashes,provenance=assets()
 assert len(manifest['candidate_train_scenes'])==1201 and len(manifest['zero_candidate_scenes'])==10
 assert manifest['S']==1191 and len(manifest['windows'])==8337
 assert all(len(window['context'])==2 for window in manifest['windows'])
 assert (plan['N'],plan['U'],plan['P'])==(8337,1043,7)
 assert len(plan['padding_window_ids'])==8 and all(len(x)==7 for x in plan['padding_window_ids'])
 assert len(plan['orders'])==8 and all(len(x)==8344 for x in plan['orders'])
 assert all(len(x)==7 for x in plan['padding_window_ids'])
 assert sum(len(x) for x in plan['padding_window_ids'])==56
 assert sum(plan['actual_expected_counts'])==66752
 assert (8*1043,8*1043*8)==(8344,66752)
 assert provenance['weights']['mast3r']['encoder_count']==292
 assert (provenance['weights']['panoptic']['adapter_count'],provenance['weights']['panoptic']['mask_decoder_count'])==(187,326)
 ref,ref_opt=base.build_model('cpu',report=False)
 ref_model_digest=state_digest(ref)
 ref_optimizer=optimizer_map(base.build_optimizer(ref))
 del ref_opt,ref
 frozen,opt=build_model('cpu',report=True)
 frozen_digest=state_digest(frozen)
 if frozen_digest!=ref_model_digest: raise AssertionError('initial state_dict mismatch')
 frozen.train()
 if frozen.understanding.encoder.training: raise AssertionError('encoder train() mode was not overridden')
 expected={f'understanding.encoder.{name}' for name,_ in frozen.understanding.encoder.named_parameters()}
 got={name for name,p in frozen.named_parameters() if not p.requires_grad}
 if not expected or got!=expected: raise AssertionError('frozen parameter set differs')
 optimizer=build_optimizer(frozen);actual_optimizer=optimizer_map(optimizer)
 if set(actual_optimizer)!={n for n,p in frozen.named_parameters() if p.requires_grad}: raise AssertionError('optimizer coverage mismatch')
 if set(ref_optimizer)-set(actual_optimizer)!=expected: raise AssertionError('reference optimizer differs only by frozen encoder')
 for name,value in actual_optimizer.items():
  if ref_optimizer[name]!=value: raise AssertionError('LR/WD changed: '+name)
 mapping=json.loads((REPORT/'weights_mapping.json').read_text())['mapping']
 encoder_map=[row for row in mapping if isinstance(row.get('target'),str) and row['target'].startswith('understanding.encoder.')]
 assert len(encoder_map)==292 and all(row['status']=='LOADED' for row in encoder_map)
 import math
 assert sum(math.prod(row['target_shape']) for row in encoder_map)==sum(p.numel() for p in frozen.understanding.encoder.parameters())
 payload=dict(status='PASS',device='cpu',control_asset_sha_equal=True,control_state_dict_sha256=ref_model_digest,
   frozen_state_dict_sha256=frozen_digest,initial_state_exact=True,module_path='understanding.encoder',
   frozen_tensor_count=len(expected),frozen_numel=sum(p.numel() for p in frozen.understanding.encoder.parameters()),
   trainable_tensor_count=sum(1 for p in frozen.parameters() if p.requires_grad),
   trainable_numel=sum(p.numel() for p in frozen.parameters() if p.requires_grad),
   frozen_names=sorted(expected),pretrained_loading_keys=[row['source'] for row in encoder_map],
   pretrained_loading_map=[dict(source=row['source'],target=row['target']) for row in encoder_map],
   pretrained_mapping_count=len(encoder_map),optimizer_parameter_count=len(actual_optimizer),
   optimizer_coverage_exact=True,lr_wd_matches_control=True,train_mode_eval=True,
   training=dict(scenes=1191,windows=8337,updates=8344,exposures=66752,padding_per_epoch=7,padding_total=56,
    checkpoints=[0,1,2,4,6,8],checkpoint_updates=[0,1043,2086,4172,6258,8344]))
 base.write_json(REPORT/'cpu_contracts.json',payload)
 base.write_json(REPORT/'frozen_encoder_manifest.json',dict(module_path=payload['module_path'],
   parameter_names=sorted(expected),parameter_count=payload['frozen_numel'],parameter_tensor_count=len(expected),
   pretrained_loading_keys=payload['pretrained_loading_keys'],pretrained_loading_map=payload['pretrained_loading_map'],pretrained_mapping_count=292,
   state_sha256_at_initialization=frozen_state_digest(frozen)))
 print(json.dumps({k:v for k,v in payload.items() if k not in ('frozen_names','pretrained_loading_keys')}))
if __name__=='__main__': main()
