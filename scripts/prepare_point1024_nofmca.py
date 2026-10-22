"""Real-zero migration, paired weight equality and immutable recipe preparation. No jobs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import hashlib,json,os,copy,unittest,gc
from types import SimpleNamespace
import torch
from torch import nn
from tokengs.options import config_defaults
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tokengs.models.point1024_nofmca import Point1024AnchorOnlyCore,Point1024NeitherCACore,C_ARCH,D_ARCH
from scripts.point1024_isoadapter_initial_state import initialize_point1024_isoadapter_from_legacy_zero
from scripts.train_dual_stream_v1 import cohorts,MANIFEST,HELDOUT
from scripts.training_windows_2ctx4novel import load_training_windows,DEFAULT_MANIFEST
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/point1024_nofmca_ablation_20261022_v1')
SOURCE=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_untrained_initial.pt')
SOURCE_SHA='3c7f1860f3897e1fba305564b30f8e4d67954e6280891bf8a33296bf42b25744'

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main():
 assert not RUN.exists(),'Refuse duplicate preparation'
 # First verify meaningful routing regressions, then audit full-sized trainable cores.
 suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_point1024_nofmca.NoFMReadTests','tests.test_point1024_nofmca.CBranchRoutingTests','tests.test_point1024_nofmca.DBranchRoutingTests','tests.test_point1024_noca.NoCALayerTests','tests.test_point1024_noca.NoCARoutingTests','tests.test_point1024_isoadapter.MigrationTests'])
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 assert result.wasSuccessful()
 assert digest(SOURCE)==SOURCE_SHA
 blob=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True);config=UnifiedQueryConfig(**blob['meta']['config'])
 training,_,_=cohorts();windows,data=load_training_windows(DEFAULT_MANIFEST,training,digest(MANIFEST))
 assert len(windows)==8337 and len({w['scene'] for w in windows})==1191
 RUN.mkdir(parents=True);audits={};adapter_hashes=[]
 for label,core,arch,key,kept,removed in [('C',Point1024AnchorOnlyCore,C_ARCH,'dual_point1024_anchor_only',625,68),('D',Point1024NeitherCACore,D_ARCH,'dual_point1024_neither_ca',593,100)]:
  torch.manual_seed(42);model=nn.Module();model.architecture_name=arch;model.config=config
  model.frozen_omega=SimpleNamespace(source_identity=blob['meta']['omega_identity']);model.sam_proposals=SimpleNamespace(source_identity=blob['meta']['sam_identity'])
  model.gfm_projection=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256));model.core=core(config_defaults['train_siu3r_unified_query_v1'],config)
  evidence=initialize_point1024_isoadapter_from_legacy_zero(blob,model,manifest_sha256=digest(MANIFEST),heldout_sha256=digest(HELDOUT),world=8)
  assert (len(evidence['copied_keys']),len(evidence['removed_keys']),len(evidence['new_keys']))==(kept,removed,16)
  state=model.state_dict()
  assert all(torch.equal(state[k],blob['model'][k]) for k in evidence['copied_keys'])
  assert not any('.image_attention.' in k for k in state)
  assert any('.anchor_attention.' in k for k in state)==(label=='C')
  h=hashlib.sha256()
  for k in sorted(k for k in state if k.startswith('core.understanding_adapters.')):h.update(state[k].contiguous().numpy().tobytes())
  adapter_hashes.append(h.hexdigest())
  audits[label]=dict(status='PASSED',architecture=arch,retained_tensors_bitwise_equal=True,retained_tensor_count=kept,removed_tensor_count=removed,new_adapter_tensor_count=16,adapter_initial_sha256=h.hexdigest(),migration=evidence,trainable_parameters=sum(p.numel() for p in model.parameters()))
  branch=RUN/label;branch.mkdir()
  for stage in ('smoke','train','eval'):
   r=branch/stage;r.mkdir()
   cfg=dict(recipe=f'point1024_{label}_no_fm_ca_full4_official1860',branch=label,architecture=arch,model_key=key,
    training_root=str(branch/'train'),smoke_root=str(branch/'smoke'),code_snapshot=str(ROOT),
    checkpoint_epoch=4,full_updates=4172,epochs=4,training_windows=8337,training_scenes=1191,global_batch=8,world_size=8,
    learning_rate=1e-4,lr_warmup=100,lr_floor=1e-5,understanding_weight=.05,understanding_warmup=1043,
    mask_resolution=256,supervised_layers=[6,12],layer_weights=[1/3,2/3],gradient_isolation=True,
    object_fm_attention='none',object_anchor_attention='learned_cross_attention' if label=='C' else 'none',geometry_routing='learned_attention_weights' if label=='C' else 'normalized_geometric_distance_softmax_v1',
    cohort='/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json',official_commit='8ea80166be76854f938e90521f1a5b688b755c87',
    initial_checkpoint=str(r/'checkpoint_untrained_initial.pt'),initial_checkpoint_sha256=SOURCE_SHA,checkpoint_every=50,epoch_probes=True,**data)
   cfg['cohort_sha256']=digest(cfg['cohort']);write(r/'configuration.json',cfg);write(r/'model_config.json',config.to_dict())
   if stage!='eval':os.link(SOURCE,r/'checkpoint_untrained_initial.pt');(r/'untrained_initial.sha256').write_text(SOURCE_SHA+'\n')
   write(r/'PREFLIGHT.json',dict(status='PASSED_CPU_PREFLIGHT',source_sha256=SOURCE_SHA,tests=result.testsRun,zero_migration=audits[label],gpu_verified=False))
  del model,state;gc.collect()
 assert len(set(adapter_hashes))==1
 # Recipe and model bodies unrelated to attention identity remain byte-identical.
 parent=Path('/space/mawb/ssst_point1024_isoadapter_noca_code_20261021_v1')
 unchanged=['tokengs/models/point1024_isoadapter_core.py','tokengs/models/point1024_isoadapter_model.py','tokengs/models/unified_query_loss.py','tokengs/models/dual_stream_initialization.py','scripts/training_windows_2ctx4novel.py','scripts/dual_stream_training_policy.py','scripts/score_dual_stream_v1.py']
 assert all((ROOT/p).read_bytes()==(parent/p).read_bytes() for p in unchanged)
 audit=dict(status='PASSED',tests=result.testsRun,source_sha256=SOURCE_SHA,branches=audits,adapters_identical_between_branches=True,unchanged_files=unchanged,training_protocol=data,world_size=8,global_batch=8,full_updates=4172,gpu_verified=False)
 write(RUN/'CPU_VALIDATION.json',audit)
 print(json.dumps(dict(status='PASSED',tests=result.testsRun,run=str(RUN),branches={k:{n:v for n,v in x.items() if n!='migration'} for k,x in audits.items()})),flush=True)

if __name__=='__main__':main()
