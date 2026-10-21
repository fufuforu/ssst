"""Reduce paired fitting diagnostics and produce scientific curves/image comparisons."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS = ('pred_joint', 'pred_recon', 'gt_recon')
STEPS = (0, 100, 200, 300)
LABELS = dict(pred_joint='Predicted geometry + joint', pred_recon='Predicted geometry + reconstruction', gt_recon='GT geometry + reconstruction')


def reduce(root):
    exports = [json.loads((root / f'rank{i:02d}' / 'export_complete.json').read_text()) for i in range(4)]
    scores = [json.loads((root / f'score_rank{i:02d}.json').read_text()) for i in range(4)]
    if any(x['status'] != 'EXPORT_COMPLETED' for x in exports) or any(x['status'] != 'SCORING_COMPLETED' for x in scores):
        raise ValueError('evaluation incomplete')
    rendered = [r for e in exports for r in e['rows']]
    images = [r for s in scores for r in s['rows']]
    if len(rendered) != 48 or len(images) != 192:
        raise ValueError('incorrect evaluation size')
    if len({(r['arm'],r['step'],r['source_index']) for r in rendered}) != 48 or len({(r['arm'],r['step'],r['window'],r['item']) for r in images}) != 192:
        raise ValueError('duplicate evaluation records')
    result = dict(diagnostic_only=True, unique_train_windows=4, images_per_condition=16,
        metric_protocol=scores[0]['metric_protocol'], official_siu3r_commit=scores[0]['official_siu3r_commit'],
        checkpoint_steps=list(STEPS), conditions={}, paired_comparisons={})
    details = []
    for arm in ARMS:
        for step in STEPS:
            rs = [r for r in rendered if (r['arm'],r['step']) == (arm,step)]
            scored = [r for r in images if (r['arm'],r['step']) == (arm,step)]
            condition = {k:float(np.mean([r[k] for r in scored])) for k in ('psnr','ssim','lpips')}
            condition.update(raw_float_psnr=float(np.mean([v for r in rs for v in r['raw_float_psnr']])),
                reconstruction_loss=float(np.mean([r['reconstruction_loss'] for r in rs])),
                understanding_loss_probe=float(np.mean([r['understanding_loss_probe'] for r in rs])),
                rgb_outside_0_1=float(np.mean([r['rgb_outside_0_1'] for r in rs])),
                alpha_mean=float(np.mean([r['alpha_mean'] for r in rs])), per_scene={})
            for r in rs:
                name = r['scene'] + '_context' + '_'.join(map(str,r['context']))
                scene_scores = [s for s in scored if s['window'] == name]
                if len(scene_scores) != 4:
                    raise ValueError('incorrect scene image count')
                detail = dict(arm=arm, step=step, scene=r['scene'],
                    **{k:float(np.mean([s[k] for s in scene_scores])) for k in ('psnr','ssim','lpips')},
                    raw_float_psnr=float(np.mean(r['raw_float_psnr'])), reconstruction_loss=r['reconstruction_loss'],
                    understanding_loss_probe=r['understanding_loss_probe'], alpha_mean=r['alpha_mean'])
                condition['per_scene'][r['scene']] = detail
                details.append(detail)
            result['conditions'][f'{arm}/{step}'] = condition
    # Both predicted-geometry initial conditions have identical source, rays,
    # feedback and readout. Confirm exact exported images, not just close means.
    truth_hashes = {}
    for r in rendered:
        name = r['scene'] + '_context' + '_'.join(map(str,r['context']))
        directory = root / 'png' / r['arm'] / f"update{r['step']:04d}" / name
        for frame in r['frame_ids']:
            path = directory / 'rgb_gt' / f'{frame:06d}.png'
            key = (name,frame)
            truth_hashes.setdefault(key,set()).add(hashlib.sha256(path.read_bytes()).hexdigest())
            if r['step']==0 and r['arm']=='pred_joint':
                other = root / 'png/pred_recon/update0000' / name / 'rgb' / f'{frame:06d}.png'
                if (directory / 'rgb' / f'{frame:06d}.png').read_bytes()!=other.read_bytes():
                    raise ValueError('predicted initial conditions are not identical')
    if len(truth_hashes)!=16 or any(len(v)!=1 for v in truth_hashes.values()):
        raise ValueError('ground truth pixels differ across arms or stages')
    result['initial_predicted_png_identical']=True
    result['ground_truth_png_identical_across_conditions']=True
    for arm in ARMS:
        first,last = result['conditions'][f'{arm}/0'],result['conditions'][f'{arm}/300']
        result['paired_comparisons'][arm+'_gain']={k:last[k]-first[k] for k in ('psnr','ssim','lpips','raw_float_psnr')}
    for name,left,right in [('remove_understanding_loss','pred_recon','pred_joint'),('replace_geometry','gt_recon','pred_recon')]:
        a,b = result['conditions'][f'{left}/300'],result['conditions'][f'{right}/300']
        result['paired_comparisons'][name]=dict(left=left,right=right,
            **{k:a[k]-b[k] for k in ('psnr','ssim','lpips','raw_float_psnr')},
            per_scene_psnr={s:a['per_scene'][s]['psnr']-b['per_scene'][s]['psnr'] for s in a['per_scene']})
    output=root/'analysis'
    if output.exists():
        raise ValueError('analysis output exists; preserve evidence')
    output.mkdir()
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'per_scene.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]))
        writer.writeheader();writer.writerows(details)
    with (output/'learning_curves.csv').open('w',newline='') as f:
        fields=['arm','step','psnr','ssim','lpips','raw_float_psnr','reconstruction_loss','understanding_loss_probe','alpha_mean']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for arm in ARMS:
            for step in STEPS:
                c=result['conditions'][f'{arm}/{step}']
                writer.writerow(dict(arm=arm,step=step,**{k:c[k] for k in fields[2:]}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig,axes=plt.subplots(1,4,figsize=(17,4.4),layout='constrained')
    for ax,key,title in zip(axes,('psnr','ssim','lpips','understanding_loss_probe'),('PSNR ↑ (dB)','SSIM ↑','LPIPS ↓','Understanding loss probe ↓')):
        for arm in ARMS:
            ax.plot(STEPS,[result['conditions'][f'{arm}/{s}'][key] for s in STEPS],marker='o',label=LABELS[arm])
        ax.set(title=title,xlabel='Optimizer updates',xticks=STEPS)
        ax.grid(alpha=.25)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.savefig(output/'learning_curves.png',dpi=180)
    fig.savefig(output/'learning_curves.svg')
    plt.close(fig)
    # Fixed first novel frame per committed scene, selected before scores.
    fig,axes=plt.subplots(4,6,figsize=(18,12),layout='constrained')
    columns=[('GT',None,None),('Predicted init','pred_recon',0),('GT geometry init','gt_recon',0),
             ('Joint 300','pred_joint',300),('Reconstruction 300','pred_recon',300),('GT geometry 300','gt_recon',300)]
    for i,export in enumerate(exports):
        r=next(x for x in export['rows'] if x['arm']=='pred_recon' and x['step']==0)
        name=r['scene']+'_context'+'_'.join(map(str,r['context']))
        frame=r['frame_ids'][2]
        for j,(title,arm,step) in enumerate(columns):
            if arm is None:path=root/'png/pred_recon/update0000'/name/'rgb_gt'/f'{frame:06d}.png'
            else:path=root/'png'/arm/f'update{step:04d}'/name/'rgb'/f'{frame:06d}.png'
            axes[i,j].imshow(Image.open(path));axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(title)
            if j==0:axes[i,j].set_ylabel(r['scene']+f'\nframe {frame}')
    fig.savefig(output/'fixed_frame_comparison.png',dpi=150)
    plt.close(fig)
    lines=['# 四窗口重建拟合诊断结果','',
           '三组各300更新/4800重复曝光；冻结VGGT、理解及反馈模块参数，保留反馈，训练adapter和GS decoder。下面是同四个训练窗口、16张图像的平均值，使用固定SIU3R原生PNG评测；不拆分视图范围，不代表完整验证集指标。','',
           '| 对照 | 初始 PSNR | 100次 | 200次 | 300次 | 最终 SSIM | 最终 LPIPS |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm,label in [('pred_joint','预测几何＋联合'),('pred_recon','预测几何＋仅重建'),('gt_recon','GT几何＋仅重建')]:
        c=result['conditions'][f'{arm}/300']
        values=[result['conditions'][f'{arm}/{s}']['psnr'] for s in STEPS]
        lines.append('| '+label+' | '+' | '.join(f'{v:.3f}' for v in values)+f" | {c['ssim']:.4f} | {c['lpips']:.4f} |")
    lines+=['','## 300次更新时的配对差异','',
            '| 场景 | 去掉理解损失：PSNR变化 | 换GT几何：PSNR变化 |',
            '|---|---:|---:|']
    scenes=list(result['conditions']['pred_joint/300']['per_scene'])
    for scene in scenes:
        x=result['paired_comparisons']['remove_understanding_loss']['per_scene_psnr'][scene]
        y=result['paired_comparisons']['replace_geometry']['per_scene_psnr'][scene]
        lines.append(f'| {scene} | {x:+.3f} | {y:+.3f} |')
    lines+=['','## 数值与解释范围','',
            '初始预测几何两组PNG逐字节相同；各组各阶段GT图像逐字节相同。每个checkpoint的冻结张量均与共同源权重逐张量相等。缓存重放与原生GS/损失已验证；评测无优化器、更新为0。',
            'raw_float_psnr、重建目标、理解损失探针及覆盖率保留在results.json和CSV，PNG裁剪/量化不会被误写为训练收益。理解损失探针不能替代官方mAP/PQ。',
            'GT条件只用于诊断：初始GT分数允许因源GS补偿预测相机而下降；比较最终绝对分数及学习曲线，而不是只比较相对初始的涨幅。四窗口短预算不能直接确定全数据最佳解冻轮数、泛化指标或架构收敛上限。',
            'SIU3R commit: '+result['official_siu3r_commit'],
            '', '学习曲线：learning_curves.png / learning_curves.svg；固定首个novel帧对照：fixed_frame_comparison.png。','']
    (output/'report.md').write_text('\n'.join(lines))
    validation=json.loads((root/'checkpoint_validation.json').read_text())
    (root/'COMPLETE.json').write_text(json.dumps(dict(status='DIAGNOSTIC_EVALUATION_COMPLETED',diagnostic_only=True,
        optimizer_updates=0,checkpoints=12,conditions=12,window_evaluations=48,image_evaluations=192,
        official_siu3r_commit=result['official_siu3r_commit'],training_git_sha=validation['training_git_sha'],
        evaluation_git_sha=exports[0]['evaluation_git_sha']),indent=2)+'\n')
    print('\n'.join(lines[:10]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    reduce(p.parse_args().root)


if __name__=='__main__':main()
