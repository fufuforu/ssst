"""Compare new residual native masks with immutable 66637 serial records."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,math,shutil
import torch
from scripts.summarize_serial_query_masks import aggregate_stage,mean
from scripts.train_dual_stream_v1 import digest
from scripts.serial_residual_initial import NEW_KEYS,RESIDUAL_ARCH,rng_equal


def summarize_rows(rows):
    result={}
    for scope in ('train','heldout'):
        data=[r for r in rows if r['scope']==scope]
        result[scope]=dict(windows=len(data),gt=sum(r['gt_count'] for r in data),supported_gt=sum(r['supported_gt'] for r in data),
            queries=sum(r['query_count'] for r in data),class_eligible=sum(r['class_eligible'] for r in data),nms_kept=sum(r['nms_kept'] for r in data),
            final_instances=sum(r['final_instances'] for r in data),queries_no_rendered_half=sum(r['queries_no_rendered_half'] for r in data),
            queries_gaussian_below_half=sum(r['queries_gaussian_below_half'] for r in data),
            stage_metrics={stage:{kind:aggregate_stage(data,['stage_metrics',stage,kind]) for kind in ('class_agnostic','correct_class','source_supported_only')} for stage in data[0]['stage_metrics']},
            representation_separation={name:mean([r['representation'][name]['cosine_separation'] for r in data]) for name in data[0]['representation']})
        count=result[scope]['queries'];result[scope]['empty_rendered_query_fraction']=result[scope]['queries_no_rendered_half']/count
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);args=p.parse_args();root=args.run_root;out=root/'diagnostics'
    if json.loads((out/'INFERENCE_COMPLETE.json').read_text())!=dict(windows=32,parameter_updates=0,baseline_reused=True):raise ValueError('Residual diagnostic incomplete')
    final_sha=digest(root/'checkpoint_latest.pt')
    for rank in range(4):
        receipt=json.loads((out/f'UNCHANGED_rank{rank:02}.json').read_text());assert receipt['unchanged'] and not receipt['optimizer_created'] and receipt['parameter_updates']==0
        assert receipt['checkpoint_sha256']==final_sha
        assert json.loads((root/f'SOURCE_RNG_VERIFIED_rank{rank:02}.json').read_text())['exact_python_numpy_torch_cuda_rng']
    preflight=json.loads((root/'preflight.json').read_text());baseline=Path(preflight['baseline_diagnostics'])/'serial'
    data={'baseline_serial':[],'identity_residual':[]};warnings=[]
    for index in range(32):
        path=baseline/f'{index:02d}.json';assert digest(path)==preflight['baseline_window_sha256'][str(index)]
        a=json.loads(path.read_text());b=json.loads((out/'residual'/f'{index:02d}.json').read_text())
        for key in ('index','scope','scene','context','input_rgb_sha256','gt_ids','gt_classes'):assert a[key]==b[key]
        for label,row in [('baseline_serial',a),('identity_residual',b)]:
            assert row['same_forward_replay_verified'] and row['final_head_replay_verified']
            data[label].append(row)
            if row.get('native_repeat_warning'):warnings.append(dict(model=label,index=index,witness=row['witness']))
    initial=torch.load(root/'checkpoint_initial.pt',map_location='cpu',weights_only=False,mmap=True)
    final=torch.load(root/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    source=torch.load(preflight['source_initial'],map_location='cpu',weights_only=False,mmap=True)
    assert final['meta']['architecture']==RESIDUAL_ARCH and final['progress']['full_updates']==256 and len(final['model'])==386
    assert initial['progress']==dict(epoch=0,update_in_epoch=0,full_updates=0) and not initial['optimizer']['state']
    assert all(torch.equal(initial['model'][k],v) for k,v in source['model'].items())
    assert digest(Path(preflight['source_initial']))==preflight['source_initial_sha256']
    assert rng_equal(initial['rank_rng'],source['rank_rng'])
    assert len(final['optimizer']['state'])==386 and all(float(s['step'])==256 for s in final['optimizer']['state'].values())
    changes={k:float((v-initial['model'][k]).float().norm()) for k,v in final['model'].items()}
    assert all(math.isfinite(v) for v in changes.values()) and all(torch.isfinite(v).all() for v in final['model'].values())
    models={label:summarize_rows(rows) for label,rows in data.items()}
    report=dict(status='COMPLETE_WITH_REPEAT_WARNINGS' if warnings else 'COMPLETE',parameter_updates_during_diagnosis=0,training_updates=256,
        baseline_reused=True,baseline_diagnostics=str(baseline.parent),same_32_rgb_and_gt=True,official_metrics=False,
        common_initial_tensors_exact=383,changed_tensors=sum(v>0 for v in changes.values()),total_tensors=386,
        new_parameter_change_norms={k:changes[k] for k in sorted(NEW_KEYS)},
        initial_gate=float(initial['model']['core.identity_gate_logit'].sigmoid()),final_gate=float(final['model']['core.identity_gate_logit'].sigmoid()),models=models,native_repeat_warnings=warnings,
        counterfactual_warning='Final-trained intermediate heads are off-distribution; use native stage metrics for the primary comparison.')
    (out/'SUMMARY.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 串行超点身份残差短测与原串行对照','', '同16场景/256更新/全局batch8；原串行直接复用66603训练及66637诊断，无基线重训或重推理。每组32窗口（16训练、16验证），权重固定；下表为GT辅助诊断，不是官方mAP/PQ。', '',
        '| 模型/范围 | 原始全部GT最佳IoU | 原始类别正确最佳IoU | 最终全部GT最佳IoU | 最终Recall@0.5 | 空渲染Mask Query比例 | 最终实例/GT数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label,scopes in models.items():
        for scope,stats in scopes.items():
            raw=stats['stage_metrics']['raw_alpha'];last=stats['stage_metrics']['final']['class_agnostic']
            lines.append(f"| {label}/{scope} | {raw['class_agnostic']['mean_iou']:.4f} | {raw['correct_class']['mean_iou']:.4f} | {last['mean_iou']:.4f} | {last['recall50']:.4f} | {stats['empty_rendered_query_fraction']:.2%} | {stats['final_instances']}/{stats['gt']} |")
    lines+=['',f"门控 {report['initial_gate']:.6f} → {report['final_gate']:.6f}；新增参数变化范数见SUMMARY。保留{len(warnings)}个完整前向重复告警（含原66637已知告警）；同一次seed/memory重放、最终头重放和参数不变严格通过。",'',
        '最终实例数不是正确匹配数；Recall为GT辅助最佳Query召回，不是官方检测召回。中间层最终头读出为分布外辅助结果，不根据其高低宣判层有害。没有改变模型阈值或官方协议，也没有自动进入全量训练。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    copies=out/'baseline';copies.mkdir(exist_ok=True);html=['<html><meta charset="utf-8"><body><h1>原串行 / 身份残差：相同GT可视化</h1>']
    for i in (0,1,2,3,16,17,18,19):
        if (baseline/f'{i:02d}.png').exists():shutil.copyfile(baseline/f'{i:02d}.png',copies/f'{i:02d}.png')
        html+=[f'<h2>{i}: {data["identity_residual"][i]["scene"]}</h2>',f'<p>原串行（复用66637）</p><img style="max-width:100%" src="baseline/{i:02d}.png">',f'<p>身份残差</p><img style="max-width:100%" src="residual/{i:02d}.png">']
    (out/'gallery.html').write_text('\n'.join(html+['</body></html>']))
    (root/'COMPLETE.json').write_text(json.dumps(dict(status='SERIAL_RESIDUAL_SHORT_AND_NATIVE_DIAG_COMPLETE',diagnostic_status=report['status'],baseline_reused=True,training_updates=256,report=str(out/'REPORT.md'),gallery=str(out/'gallery.html')),indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['models','native_repeat_warnings']},indent=2))


if __name__=='__main__':main()
