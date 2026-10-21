"""Read-only checkpoint audit plus small independent run-directory preparation."""
import ast,hashlib,json,os,subprocess
from pathlib import Path
import torch
root=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/A8_compare_official1860_20261021_v1')
snapshot=Path(__file__).resolve().parents[1]
assert not root.exists(),'Refusing duplicate experiment directory'
sources=dict(old_A4_plus4_A8=Path('/space/mawb/ssst/group_plus/unified_query_training_v1/attempt08_legacy_dual_superpoint_extend8_node13_8gpu/checkpoint_latest.pt'),single16_epoch8_L6L12=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/single16_L6L12/checkpoint_epoch8.pt'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
blobs={label:torch.load(p,map_location='cpu',weights_only=False,mmap=True) for label,p in sources.items()}
a,b=blobs.values()
for blob in blobs.values():
    assert blob['progress']==dict(epoch=8,update_in_epoch=0,full_updates=8344)
    assert len(blob['rank_rng'])==8 and blob['meta']['world_size']==8
    assert not any('iou_head.' in key for key in blob['model'])
for key in ('architecture','config','manifest_sha256','heldout_sha256','omega_identity','sam_identity','global_batch'):
    assert a['meta'][key]==b['meta'][key],key
assert a['model'].keys()==b['model'].keys()
assert all(a['model'][k].shape==b['model'][k].shape for k in a['model'])
assert a['meta']['epochs']==8 and b['meta']['epochs']==16
assert a['meta']['manifest_sha256']==digest(Path('/space/mawb/ssst/group_plus/object_locus_panoptic_full1201_8gpu/manifest.json'))
assert a['meta']['heldout_sha256']==digest(Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/representative128_cohort_20261014.json'))
# The resolution switch must reproduce each original SemanticMemory.forward exactly.
def forward(path):
    module=ast.parse(path.read_text());cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='SemanticMemory')
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
new=forward(snapshot/'tokengs/models/object_locus_dual_stream_v1.py')
for resolution,native in [(128,Path('/space/mawb/ssst_superpoint_dual_extend_code_20261017')),(256,Path('/space/mawb/ssst_sem_res256_code_20261019'))]:
    import copy
    check=copy.deepcopy(new)
    check.body=[n for n in check.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='resolution' for t in n.targets)) and not isinstance(n,ast.If)]
    class Resolve(ast.NodeTransformer):
        def visit_Name(self,node):return ast.copy_location(ast.Constant(resolution),node) if node.id=='resolution' else node
    check=Resolve().visit(check)
    assert ast.dump(check,include_attributes=False)==ast.dump(forward(native/'tokengs/models/object_locus_dual_stream_v1.py'),include_attributes=False),resolution
cohort=Path('/space/mawb/ssst/group_plus/vggt_understanding_bottleneck_v1/official1860_cohort_20261020.json')
records=json.loads(cohort.read_text())['records'];official=json.loads(Path('/space/mawb/SIU3R/data/scannet/val_pair.json').read_text())
assert {(x['scene'],tuple(x['context']),tuple(x['frame_ids'])) for x in records}=={(x['scan'],tuple(x['context_ids']),tuple(x['target_ids'])) for x in official}
assert len(records)==1860 and len({x['scene'] for x in records})==312
official_sha='8ea80166be76854f938e90521f1a5b688b755c87'
assert subprocess.check_output(['git','-C','/space/mawb/SIU3R','rev-parse','HEAD'],text=True).strip()==official_sha
assert not subprocess.check_output(['git','-C','/space/mawb/SIU3R','status','--porcelain','--','src'],text=True).strip()
checks={}
for label,p in sources.items():
    sha=digest(p)
    if label=='old_A4_plus4_A8':assert sha=='0ed6cd8f4fb740698c01ca345a0bbad9a2b1b96302acdbf106864874e5e282c9'
    d=root/label;d.mkdir(parents=True)
    os.link(p,d/'source_checkpoint.pt')
    config=dict(recipe=label,architecture=a['meta']['architecture'],checkpoint_source=str(p),checkpoint_sha256=sha,checkpoint_epoch=8,full_updates=8344,mask_resolution=128 if label.startswith('old_') else 256,supervised_layers=[6,8,10,12] if label.startswith('old_') else [6,12],cohort=str(cohort),cohort_sha256=digest(cohort),code_snapshot=str(snapshot),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=snapshot,text=True).strip(),iou_head=False,official_commit=official_sha)
    (d/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    checks[label]=dict(checkpoint_sha256=sha,meta=blobs[label]['meta'],progress=blobs[label]['progress'])
(root/'PREFLIGHT.json').write_text(json.dumps(dict(status='PASSED',checkpoints=checks,same_model_keys_and_shapes=True,same_data_and_frozen_assets=True,native_resolution_forward_ast_exact=True,windows=1860,scenes=312,official_src_unmodified=True,comparison_caveat='LR schedule, supervision layers and mask resolution all differ'),indent=2)+'\n')
print(json.dumps(dict(status='PASSED',checkpoints={k:v['checkpoint_sha256'] for k,v in checks.items()},root=str(root))))
