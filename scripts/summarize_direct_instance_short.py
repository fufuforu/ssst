"""Audit a fresh direct-attribute short run and paired frozen readouts, no retraining."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,math,statistics
import torch
from scripts.direct_instance_short_policy import HEADS,validate_attribute_smoke
from scripts.train_dual_stream_v1 import digest


def parameter_sha(state,names):
    h=hashlib.sha256()
    for name in names:h.update(name.encode());h.update(state[name].contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);a=parser.parse_args();root=a.run_root
    pre=json.loads((root/'preflight.json').read_text());smoke=json.loads((root/'SMOKE_PASSED.json').read_text())
    validate_attribute_smoke(smoke,preflight_sha256=digest(root/'preflight.json'),code_sha=pre['code_sha'])
    initial=torch.load(root/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    final=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    assert initial['progress']==dict(epoch=0,update_in_epoch=0,full_updates=0) and not initial['optimizer']['state']
    assert initial['meta']['initial_checkpoint'] is None and final['progress']['full_updates']==256 and final['progress']['epoch']==128
    assert final['meta']['architecture']==pre['architecture'] and final['meta']['code_sha']==pre['code_sha']
    for key,value in pre['extra_checkpoint_metadata'].items():assert final['meta'][key]==value
    assert final['meta']['training_windows']==pre['training_windows'] and final['meta']['world_size']==4 and final['meta']['global_batch']==8
    assert all(torch.isfinite(v).all() for v in final['model'].values())
    assert sorted({float(v['step']) for v in final['optimizer']['state'].values()})==[256.]
    before_sha=parameter_sha(initial['model'],pre['trainable_parameter_names']);after_sha=parameter_sha(final['model'],pre['trainable_parameter_names'])
    assert before_sha==smoke['initial_parameter_sha256'] and after_sha!=before_sha
    baseline=torch.load(Path(pre['baseline_run'])/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    assert all(torch.equal(initial['model'][k],baseline['model'][k]) for k in pre['common_parameter_names'])
    changes={k:float((v-initial['model'][k]).float().norm()) for k,v in final['model'].items()}
    prefixes=dict(projection='gfm_projection.',initializer='core.initializer.',decoder='core.blocks.',geometry='core.geometry.',
        reconstruction='core.activation_head.',classifier='core.classifier.',attributes='core.instance_attribute_head.',
        prototypes='core.instance_prototype_head.',stuff='core.stuff_attribute_head.')
    families={name:sum(v for k,v in changes.items() if k.startswith(prefix)) for name,prefix in prefixes.items()}
    assert all(math.isfinite(v) and v>0 for v in families.values())
    sync=json.loads((root/'FIRST_UPDATE_VERIFIED.json').read_text());assert sync['status']=='PASSED' and len(sync['ranks'])==4
    assert len({r['sha256'] for r in sync['ranks']})==1 and all(r['changed'] for r in sync['ranks'])
    ranks=[]
    for rank in range(4):
        rows=[json.loads(x) for x in (root/f'train_rank{rank:02d}.jsonl').read_text().splitlines()]
        assert len(rows)==256 and [r['update'] for r in rows]==list(range(1,257))
        assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
        assert all(abs(r['instance_feature_effective_weight']-.01*r['understanding_weight'])<1e-12 for r in rows)
        ranks.append(dict(rank=rank,updates=256,mean_step_seconds=statistics.mean(r['seconds'] for r in rows),
            peak_cuda_memory_bytes=max(r['peak_cuda_memory_bytes'] for r in rows),
            last20={k:statistics.mean(r[k] for r in rows[-20:]) for k in ('loss_instance_pull','loss_instance_push','loss_instance_cross','instance_attribute_norm_mean','instance_prototype_norm_mean','instance_membership_logit_scale','instance_membership_logit_bias')}))
    reference_path=Path(pre['baseline_reference'])/'SUMMARY.json';assert digest(reference_path)==pre['baseline_reference_summary_sha256']
    old_reference=json.loads(reference_path.read_text());comparison=[];evaluations={}
    mapping=dict(membership_raw='prototype_raw',membership_nms='prototype_nms',membership_assembled='prototype',hdbscan8d='cluster')
    for r in old_reference['mask_summary']:
        if r['method'] in mapping:
            comparison.append(dict(model=r['model'],stage='FINAL',scope=r['scope'],method=mapping[r['method']],
                gt_count=r['gt_count'],pred_count=r['pred_count'],tp50=r['tp50'],one_to_one_gt_iou=r['one_to_one_gt_iou'],precision50=r['precision50'],recall50=r['recall50']))
    for stage,saved,expected_sha in [('initial',initial,before_sha),('final',final,after_sha)]:
        folder=root/f'readouts_{stage}';ev=json.loads((folder/'SUMMARY.json').read_text());done=json.loads((folder/'COMPLETE.json').read_text())
        assert done['windows']==8 and ev['parameter_updates']==0 and not ev['optimizer_created']
        assert len(ev['rows'])==8 and len(ev['fixed_weight_checks'])==4
        assert all(r['before']==r['after']==expected_sha and r['no_parameter_gradients'] and r['all_parameter_versions_unchanged'] for r in ev['fixed_weight_checks'])
        ordered=sorted(ev['rows'],key=lambda r:r['index']);assert [r['index'] for r in ordered]==list(range(8))
        assert [(r['scene'],r['context']) for r in ordered]==[(w['scene'],w['context']) for w in pre['training_windows'][:4]+pre['heldout_probes']]
        assert {r['method'] for r in ev['summary']}=={'prototype','prototype_raw','prototype_nms','cluster'}
        evaluations[stage]=dict(checkpoint_sha256=digest(root/f'checkpoint_{"initial" if stage=="initial" else "latest"}.pt'),fixed_weight_checks=ev['fixed_weight_checks'],summary=ev['summary'])
        for r in ev['summary']:comparison.append(dict(model='direct_attributes',stage=stage.upper(),**r))
    native=[]
    for label,run in [('baseline67050',Path(pre['baseline_run'])),('direct_attributes',root)]:
        for stage in ('INITIAL','FINAL'):
            rows=json.loads((run/f'PROBES_{stage}.json').read_text())['rows']
            assert len(rows)==8
            for scope in ('train','heldout'):
                selected=[r for r in rows if r['scope']==scope];count=sum(r['supported_gt_count'] for r in selected)
                native.append(dict(model=label,stage=stage,scope=scope,psnr=statistics.mean(r['metrics']['psnr'] for r in selected),
                    supported_gt_count=count,source_matched_probe_iou=sum(r['best_supported_gt_mask_iou']*r['supported_gt_count'] for r in selected)/max(1,count)))
    report=dict(status='DIRECT_ATTRIBUTE_SHORT_AND_READOUTS_COMPLETE_REVIEW_QUALITY',updates=256,world_size=4,training_scenes=16,global_batch=8,
        common_initial_tensors_exact=pre['common_initial_tensors_exact'],smoke_updates_discarded=True,fresh_initial_parameter_sha256=before_sha,
        final_parameter_sha256=after_sha,adam_steps=[256],parameter_change_norms=families,first_update_sync=sync,ranks=ranks,
        native_probe_comparison=native,readout_comparison=comparison,evaluations=evaluations,
        baseline_retrained=False,official128=False,auto_full_training=False,
        limits=['Four train/four heldout probes, one seed; no official mAP/PQ.',
            'Shared zero-step weights match; replaced attribute/prototype/stuff readouts are new and have different initial outputs.',
            'Source-matched two-view probe IoU and all-GT per-image one-to-one IoU are different metrics.',
            'Prototype raw, NMS, assembled and class-agnostic HDBSCAN are kept separate.'])
    (root/'SHORT_VALIDATION_SUMMARY.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (root/'COMPLETE.json').write_text(json.dumps(dict(status=report['status'],updates=256,readout_stages=['initial','final'],official128=False,auto_full_training=False))+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
