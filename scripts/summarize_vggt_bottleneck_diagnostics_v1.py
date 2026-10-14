"""CPU summaries of raw-mask losses or actual spikes; no new GPU calls."""
import argparse
import json
import math
from pathlib import Path


def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def raw(root,out,arms='ABC'):
    records=[];expected=None;selection=None
    for rank in range(4):
        report=read(root/f'rank{rank:02d}/COMPLETE.json')
        assert report['status']=='RAW_READOUT_PROBED' and report['new_optimizer_updates']==0
        assert report['model_parameters_unchanged'] and report['gt_only_used_after_context_generation']
        if expected is None:expected=report.get('cohort_windows',20);selection=report['cohort_selection']
        assert report.get('cohort_windows',20)==expected
        records+=report['records']
    assert len(records)==len(arms)*expected
    reference=None;results={}
    for arm in arms:
        selected=[r for r in records if r['arm']==arm];names=sorted(r['name'] for r in selected)
        assert len(names)==len(set(names))==expected
        if reference is None:reference=names
        assert names==reference
        assert all(r['endpoint_pngs_reproduced'] and r['provider_rgb_reproduced'] for r in selected)
        results[arm]={}
        for view in ('context','target'):
            stages={};gt=sum(r['views'][view]['final']['gt'] for r in selected)
            for stage in ('raw_all','eligible_raw','winner_before_area_filter','final'):
                values=[r['views'][view][stage] for r in selected]
                stages[stage]={k:sum(z[k] for z in values) for k in ('best50','same50','best75','same75','nonempty_queries')}
                stages[stage].update(gt=gt,best_mask_recall50=stages[stage]['best50']/gt,
                    same_class_best_mask_recall50=stages[stage]['same50']/gt)
            transitions={}
            for before,after in [('raw_all','eligible_raw'),('eligible_raw','winner_before_area_filter'),('winner_before_area_filter','final')]:
                lost=recovered=0
                for r in selected:
                    b=r['views'][view][before]['gt_best_iou'];a=r['views'][view][after]['gt_best_iou']
                    assert len(a)==len(b)
                    lost+=sum(x>=.5 and y<.5 for x,y in zip(b,a));recovered+=sum(x<.5 and y>=.5 for x,y in zip(b,a))
                transitions[before+' to '+after]=dict(lost_geometric_matches50=lost,recovered_geometric_matches50=recovered)
            null=low=both=other=0
            for r in selected:
                before=r['views'][view]['raw_all'];after=r['views'][view]['eligible_raw']
                for old,new,q in zip(before['gt_best_iou'],after['gt_best_iou'],before['gt_best_query_index']):
                    if old>=.5 and new<.5:
                        p=r['query_class_probabilities'][q];isnull=p.index(max(p))==18;islow=max(p[:18])<.05
                        if isnull and islow:both+=1
                        elif isnull:null+=1
                        elif islow:low+=1
                        else:other+=1
            result=dict(stages=stages,transitions=transitions,eligibility_loss_best_raw_query_reason=dict(no_object_only=null,thing_probability_below005_only=low,both=both,other=other))
            if view=='context':
                proxy=[r['views'][view]['visual_qpre_proxy'] for r in selected]
                result['visual_qpre_mask_proxy']=dict(gt=gt,best50=sum(x['best50'] for x in proxy),
                    best_mask_recall50=sum(x['best50'] for x in proxy)/gt,
                    caveat='class-agnostic 2D dot-product proxy, not native pretrained segmentation or a model upper bound')
                assigned=[z for r in selected for z in zip(r['native_context_hungarian']['matched_hard_mask_iou'],r['native_context_hungarian']['matched_query_argmax'])] if all('native_context_hungarian' in r for r in selected) else []
                if assigned:
                    result['native_matched_queries']=dict(gt=len(assigned),unassigned_gt=sum(i is None for i,c in assigned),
                        matched_raw_mask50=sum(i is not None and i>=.5 for i,c in assigned),
                        matched_query_no_object=sum(c==18 for i,c in assigned),good_matched_mask_but_no_object=sum(i is not None and i>=.5 and c==18 for i,c in assigned))
            results[arm][view]=result
    write(out/'results.json',dict(status='RAW_READOUT_BOTTLENECKS_ANALYZED',records=len(records),selected_scenes=expected,cohort_selection=selection,arms=results,
        native_endpoint_reproduced=True,new_optimizer_updates=0,
        caveat='Stage diagnostics on the stated cohort; not full official validation performance or architecture causality.'))
    lines=['# 原始query与实例组装诊断','', f'{expected}个场景，{list(arms)}每组{expected}窗口；零训练，原生PNG重现。选择协议：{selection}。诊断指标不替换官方13列。','',
        '| 对照 | GT条目 | 原始mask类别无关recall@0.5 | 资格筛选后 | 像素竞争后 | 面积筛选后 |','|---|---:|---:|---:|---:|---:|']
    for arm in arms:
        s=results[arm]['target']['stages'];vals=[s[k]['best_mask_recall50']*100 for k in ('raw_all','eligible_raw','winner_before_area_filter','final')]
        lines.append('|'+arm+'|'+str(s['final']['gt'])+'|'+'|'.join(f'{v:.2f}%' for v in vals)+'|')
    lines+=['','逐GT损失/恢复、原始最佳query的no-object/概率原因，以及输入2D代理见results.json。',
        '竞争可能改善mask IoU，阶段召回不保证单调。原始100个query的类别无关覆盖是离线可用mask诊断，不是实际预测性能。',
        '不直接取消筛选：大量无对象query可能增加误检。应结合独立248的FP、GT排序诊断和实际分类/mask绑定进一步判断。',
        '历史U3三尺度query前融合已无收益，本诊断没有重复融合训练。主训练与clip保持不变。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def spike(root,out):
    reports={}
    for arm in 'ABC':
        a=root/arm;c=read(a/'COMPLETE.json');assert c['status'] in ('ACTUAL_SPIKE_CAPTURED','NO_SPIKE_WITHIN_BOUNDED_REPLAY')
        ranks=[[json.loads(x) for x in (a/f'replay_rank{i}.jsonl').read_text().splitlines()] for i in range(4)]
        assert all(len(r)==c['replayed_calls'] and [x['completed_updates'] for x in r]==list(range(301,301+c['replayed_calls'])) for r in ranks)
        assert c['frozen_parameters_unchanged'] and c['clip_unchanged']
        observations=[]
        for step in [301]+c['actual_spike_steps']:
            saved=read(a/f'global_gradients_update{step:04d}.json');assert len(saved['rank_rows'])==4
            d=saved['rank_rows'][0];norm=d['native_preclip_norm'];groups=d['groups'];total=sum(x['combined_squared'] for x in groups.values())
            assert d['global_recombination_error_norm']<1e-4*max(norm,1.)
            assert abs(math.sqrt(sum(x['postclip_squared'] for x in groups.values()))-min(norm,1.))<1e-5
            def scale(group):return .01 if group in ('memory_adapter','shared_gs_decoder','gs_anchor_head','gs_activation_head','reconstruction_other') else 1.
            rr=math.sqrt(sum(x['rec_squared'] for x in groups.values()));uu=math.sqrt(sum(scale(g)**2*x['weighted_U_squared'] for g,x in groups.items()))
            observations.append(dict(step=step,actual_spike=saved['actual_spike'],native_norm=norm,
                original_logged_norm=saved['original_norm'],global_R_norm=rr,global_GC_scaled_U_norm=uu,
                clip_multiplier=d['global_clip_multiplier'],groups={g:dict(combined_norm=x['combined_preclip_norm'],
                    combined_squared_share=x['combined_squared']/total,R_norm=x['reconstruction_norm'],
                    GC_scaled_U_norm=scale(g)*x['weighted_understanding_norm'],R_U_cosine=x['global_R_U_cosine'],
                    actual_Adam_parameter_update_norm=x['actual_parameter_update_norm']) for g,x in groups.items()}))
        drift=max(abs(row[k]-row['original_logged_values'][k]) for r in ranks for row in r for k in ('loss_recon','loss_understanding'))
        reports[arm]=dict(status=c['status'],replayed_calls=c['replayed_calls'],observations=observations,max_R_U_loss_replay_drift=drift)
    write(out/'results.json',dict(status='ACTUAL_SPIKE_DIAGNOSIS_ANALYZED',arms=reports,
        original_799_attribution_not_assumed=True,main_training_untouched=True,clip_unchanged=1.,
        caveat='Observed replay events only. One spike is not a frequency/stability ranking; clip multiplier is not Adam update multiplier. Batch RNG files are not full resumable event checkpoints.'))
    lines=['# 实际尖峰捕获结果','', '从各自300步恢复，首次真实norm>=20触发后停止，最多141调用。原loss/GC/AdamW/clip1不变。','',
        '| 对照 | 重放调用 | 实际捕获步 | 范数 | 全局R范数 | GC缩放后U范数 |','|---|---:|---:|---:|---:|---:|']
    for arm,r in reports.items():
        events=[x for x in r['observations'] if x['actual_spike']]
        if events:
            e=events[0];lines.append(f"|{arm}|{r['replayed_calls']}|{e['step']}|{e['native_norm']:.3f}|{e['global_R_norm']:.3f}|{e['global_GC_scaled_U_norm']:.3f}|")
        else:lines.append(f"|{arm}|{r['replayed_calls']}|未捕获||||")
    lines+=['','分模块R/U、裁剪前范数份额和真实Adam更新见results.json。',
        '这些是实际重放事件，不能自动代表历史第441步799尖峰，不能凭一个事件给三组稳定性排序。',
        '没有修改clip阈值或启动新的模型训练。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('raw','spike'),required=True)
    p.add_argument('--arms',nargs='+',default=list('ABC'),help='Raw readout model labels; default A B C.')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.mode=='raw':raw(a.root,a.output,a.arms)
    else:spike(a.root,a.output)
    write(a.output/'COMPLETE.json',dict(status='DIAGNOSTIC_SUMMARY_COMPLETE',mode=a.mode,new_gpu_calls=0))


if __name__=='__main__':main()
