"""Pair adapter interventions with the original identical-source fitting curves."""
import argparse
import csv
import json
from pathlib import Path

VARIANTS=('adapter_u001','adapter_u0')
STEPS=(0,100,200,300)
PARENT=Path('/space/mawb/ssst/group_plus/vggt_overfit_diagnostic_v1/evaluation/attempt01')


def reduce(root):
    exports=[json.loads((root/f'rank{i:02d}/export_complete.json').read_text()) for i in range(4)]
    scores=[json.loads((root/f'score_rank{i:02d}.json').read_text()) for i in range(4)]
    if any(x['status']!='EXPORT_COMPLETED' or x['optimizer_updates']!=0 for x in exports):raise ValueError('export incomplete')
    if any(x['status']!='SCORING_COMPLETED' or x['optimizer_updates']!=0 for x in scores):raise ValueError('scores incomplete')
    rows=[r for e in exports for r in e['rows']]
    images=[r for s in scores for r in s['rows']]
    if len(rows)!=32 or len(images)!=128 or len({(r['arm'],r['step'],r['source_index']) for r in rows})!=32:
        raise ValueError('wrong evaluation count or duplicate exports')
    if len({(r['arm'],r['step'],r['window'],r['item']) for r in images})!=128:raise ValueError('duplicate score records')
    previous=json.loads((PARENT/'analysis/results.json').read_text())
    if any(s['official_siu3r_commit']!=previous['official_siu3r_commit'] or s['metric_protocol']!=previous['metric_protocol'] for s in scores):
        raise ValueError('metric protocol differs from baseline')
    conditions={k:v for k,v in previous['conditions'].items() if k.startswith(('pred_joint/','pred_recon/'))}
    details=[]
    for variant in VARIANTS:
        for step in STEPS:
            rs=[r for r in rows if (r['arm'],r['step'])==(variant,step)]
            scored=[r for r in images if (r['arm'],r['step'])==(variant,step)]
            if len(rs)!=4 or len(scored)!=16:raise ValueError('incomplete condition')
            condition={k:sum(r[k] for r in scored)/16 for k in ('psnr','ssim','lpips')}
            condition.update(reconstruction_loss=sum(r['reconstruction_loss'] for r in rs)/4,
                understanding_loss_probe=sum(r['understanding_loss_probe'] for r in rs)/4,
                raw_float_psnr=sum(sum(r['raw_float_psnr']) for r in rs)/16,per_scene={})
            for r in rs:
                name=r['scene']+'_context'+'_'.join(map(str,r['context']))
                directory=root/'png'/variant/f'update{step:04d}'/name
                baseline=PARENT/'png/pred_joint/update0000'/name
                for frame in r['frame_ids']:
                    filename=f'{frame:06d}.png'
                    if (directory/'rgb_gt'/filename).read_bytes()!=(baseline/'rgb_gt'/filename).read_bytes():
                        raise ValueError('GT pixels differ from original baseline')
                    if step==0 and (directory/'rgb'/filename).read_bytes()!=(baseline/'rgb'/filename).read_bytes():
                        raise ValueError('initial rendered pixels differ from original baseline')
                ss=[s for s in scored if s['window']==name]
                if len(ss)!=4:raise ValueError('wrong per-scene score count')
                detail=dict(arm=variant,step=step,scene=r['scene'],**{k:sum(s[k] for s in ss)/4 for k in ('psnr','ssim','lpips')})
                condition['per_scene'][r['scene']]=detail;details.append(detail)
            conditions[f'{variant}/{step}']=condition
    out=root/'analysis'
    if out.exists():raise ValueError('preserve previous analysis')
    out.mkdir()
    result=dict(diagnostic_only=True,unique_training_windows=4,images_per_condition=16,
        baseline_source=str(PARENT),initial_png_and_gt_identical_to_baseline=True,conditions=conditions,
        official_siu3r_commit=previous['official_siu3r_commit'],metric_protocol=previous['metric_protocol'],
        optimizer_updates_during_evaluation=0)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=['arm','step','psnr','ssim','lpips','raw_float_psnr','reconstruction_loss','understanding_loss_probe']
    with (out/'learning_curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for arm in ('pred_joint','pred_recon')+VARIANTS:
            for step in STEPS:
                c=conditions[f'{arm}/{step}'];w.writerow(dict(arm=arm,step=step,**{k:c[k] for k in fields[2:]}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'pred_joint':'Joint: adapter U=1','pred_recon':'Reconstruction only',
        'adapter_u001':'Joint: adapter U=0.01','adapter_u0':'Joint: adapter U=0'}
    fig,axes=plt.subplots(1,4,figsize=(17,4.4),layout='constrained')
    for ax,key in zip(axes,('psnr','ssim','lpips','understanding_loss_probe')):
        for arm,label in labels.items():ax.plot(STEPS,[conditions[f'{arm}/{s}'][key] for s in STEPS],marker='o',label=label)
        ax.set(title=key,xlabel='Optimizer updates',xticks=STEPS);ax.grid(alpha=.25)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=4)
    fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.svg');plt.close(fig)
    lines=['# Adapter 理解梯度路由短对照','',
        '同四个训练窗口、每条件16张图像；SIU3R原生PNG指标。这是拟合诊断，不能作为完整验证集成绩。理解模块冻结、反馈开启。新对照保留理解损失，只改变adapter的理解梯度系数；GS保持0.01。','',
        '| 对照 | 初始 PSNR | 100 | 200 | 300 | 最终 SSIM | 最终 LPIPS | 理解损失探针 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,label in labels.items():
        c=conditions[f'{arm}/300']
        lines.append('| '+label+' | '+' | '.join(f"{conditions[f'{arm}/{s}']['psnr']:.3f}" for s in STEPS)+
            f" | {c['ssim']:.4f} | {c['lpips']:.4f} | {c['understanding_loss_probe']:.5f} |")
    lines.extend(['','理解探针不能替代mAP/PQ；完整理解任务的保留情况需在理解分支可训练、具有留出窗口的下一阶段验证。',''])
    (out/'report.md').write_text('\n'.join(lines))
    (root/'COMPLETE.json').write_text(json.dumps(dict(status='ADAPTER_GRADIENT_DIAGNOSTIC_EVALUATION_COMPLETED',
        diagnostic_only=True,optimizer_updates=0,checkpoint_conditions=8,window_evaluations=32,image_evaluations=128,
        initial_png_and_gt_identical_to_baseline=True,evaluation_git_sha=exports[0]['evaluation_git_sha']),indent=2)+'\n')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    reduce(p.parse_args().root)


if __name__=='__main__':main()
