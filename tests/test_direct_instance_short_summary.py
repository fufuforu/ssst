"""CPU-only end-to-end audit: fresh Adam versus accidentally inherited updates."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from tests.test_direct_instance_short_policy import receipt
from scripts.summarize_direct_instance_short import main,parameter_sha


def write(path,data):path.write_text(json.dumps(data)+'\n')


def fixture(root):
    baseline=root/'baseline';reference=root/'reference';baseline.mkdir();reference.mkdir()
    prefixes=['gfm_projection.','core.initializer.','core.blocks.','core.geometry.','core.activation_head.',
        'core.classifier.','core.instance_attribute_head.','core.instance_prototype_head.','core.stuff_attribute_head.']
    names=[prefix+'weight' for prefix in prefixes]
    params=[torch.nn.Parameter(torch.tensor([.2+i*.01])) for i in range(len(names))]
    initial={name:p.detach().clone() for name,p in zip(names,params)}
    optimizer=torch.optim.AdamW(params,lr=1e-4)
    for _ in range(256):
        optimizer.zero_grad();sum(p.square().sum() for p in params).backward();optimizer.step()
    final={name:p.detach().clone() for name,p in zip(names,params)}
    windows=[dict(scene=f'scene{i}',context=[i,i+1]) for i in range(16)];held=[dict(scene=f'held{i}',context=[i,i+1]) for i in range(4)]
    metadata=dict(architecture='test',code_sha='code',initial_checkpoint=None,training_windows=windows,world_size=4,global_batch=8,recipe={'test':True})
    a=dict(model=initial,optimizer=dict(state={}),meta=metadata,progress=dict(epoch=0,update_in_epoch=0,full_updates=0))
    z=dict(model=final,optimizer=optimizer.state_dict(),meta=metadata,progress=dict(epoch=128,update_in_epoch=0,full_updates=256))
    torch.save(a,root/'checkpoint_initial.pt');torch.save(z,root/'checkpoint_latest.pt');torch.save(dict(model=initial),baseline/'checkpoint_initial.pt')
    write(reference/'SUMMARY.json',dict(mask_summary=[]))
    pre=dict(code_sha='code',architecture='test',extra_checkpoint_metadata=dict(recipe={'test':True}),training_windows=windows,
        heldout_probes=held,baseline_run=str(baseline),baseline_reference=str(reference),baseline_reference_summary_sha256=hashlib.sha256((reference/'SUMMARY.json').read_bytes()).hexdigest(),
        common_parameter_names=names[:6],common_initial_tensors_exact=6,trainable_parameter_names=names)
    write(root/'preflight.json',pre);s=receipt();s['preflight_sha256']=hashlib.sha256((root/'preflight.json').read_bytes()).hexdigest()
    sha0=parameter_sha(initial,names);sha1=parameter_sha(final,names);s['initial_parameter_sha256']=sha0
    for r in s['ranks']:r['initial_parameter_sha256']=sha0
    write(root/'SMOKE_PASSED.json',s)
    write(root/'FIRST_UPDATE_VERIFIED.json',dict(status='PASSED',ranks=[dict(sha256='same',changed=True) for _ in range(4)]))
    for rank in range(4):
        rows=[dict(update=i,seconds=1.,peak_cuda_memory_bytes=0,understanding_weight=.05,instance_feature_effective_weight=.0005,
            loss_instance_pull=.1,loss_instance_push=.1,loss_instance_cross=.1,instance_attribute_norm_mean=1.,instance_prototype_norm_mean=1.,
            instance_membership_logit_scale=10.,instance_membership_logit_bias=0.) for i in range(1,257)]
        (root/f'train_rank{rank:02d}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for stage,sha in [('initial',sha0),('final',sha1)]:
        folder=root/f'readouts_{stage}';folder.mkdir();write(folder/'COMPLETE.json',dict(windows=8))
        rows=[dict(index=i,scene=w['scene'],context=w['context']) for i,w in enumerate(windows[:4]+held)]
        summary=[dict(scope=scope,method=method,one_to_one_gt_iou=.2) for scope in ('train','heldout') for method in ('prototype','prototype_raw','prototype_nms','cluster')]
        checks=[dict(before=sha,after=sha,no_parameter_gradients=True,all_parameter_versions_unchanged=True) for _ in range(4)]
        write(folder/'SUMMARY.json',dict(rows=rows,summary=summary,fixed_weight_checks=checks,parameter_updates=0,optimizer_created=False))
    native=[dict(scope='train' if i<4 else 'heldout',scene=w['scene'],supported_gt_count=1,best_supported_gt_mask_iou=.2,metrics=dict(psnr=10.)) for i,w in enumerate(windows[:4]+held)]
    for folder in (root,baseline):
        for stage in ('INITIAL','FINAL'):write(folder/f'PROBES_{stage}.json',dict(rows=native))


class SummaryTests(unittest.TestCase):
    def test_complete_audit_and_no_loss_decrease_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            with patch('sys.argv',['summary','--run-root',str(root)]),patch('builtins.print'):main()
            d=json.loads((root/'SHORT_VALIDATION_SUMMARY.json').read_text())
            self.assertEqual(d['adam_steps'],[256]);self.assertEqual(len(d['readout_comparison']),16)
            self.assertTrue((root/'COMPLETE.json').exists());self.assertTrue(d['smoke_updates_discarded'])

    def test_reject_initial_checkpoint_that_inherited_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            path=root/'checkpoint_initial.pt';saved=torch.load(path,weights_only=False);saved['progress']['full_updates']=2;torch.save(saved,path)
            with patch('sys.argv',['summary','--run-root',str(root)]),self.assertRaises(AssertionError):main()
            self.assertFalse((root/'COMPLETE.json').exists())


if __name__=='__main__':unittest.main()
