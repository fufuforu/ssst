"""Summarize measured native gradients without assigning unmeasured causal effects."""
import argparse
import json
from pathlib import Path


def format_number(x):return '—' if x is None else f'{x:.4g}'


def summarize(root):
    global_data=json.loads((root/'global_gradient_statistics.json').read_text())
    shards=[json.loads((root/f'rank{i:02d}/COMPLETE.json').read_text()) for i in range(4)]
    rows=global_data['rows']
    if len(rows)!=8 or any(s['optimizer_updates']!=0 or len(s['rows'])!=8 for s in shards):
        raise ValueError('zero-update gradient attribution incomplete')
    local=[r for s in shards for r in s['rows']]
    if len({(r['arm'],r['step'],r['source_index']) for r in local})!=32:raise ValueError('duplicate probe records')
    lines=['# 理解梯度零更新诊断','',
        '八个已有checkpoint条件、四个固定训练窗口，共32次窗口分析；没有优化器、没有参数更新。参数梯度先在四窗口间准确平均，再统计范数与夹角。输出Gaussian/像素梯度按窗口保留，不能当成全模型梯度。','',
        '| 来源 | 更新 | 家族 | 重建梯度范数 | 理解梯度范数 | U/R 范数比 | cos(R,U) |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        for group,stats in row['component_gradients']['understanding'].items():
            lines.append(f"| {row['arm']} | {row['step']} | {group} | "+' | '.join(format_number(stats[k])
                for k in ('reconstruction_norm','understanding_norm','norm_ratio','cosine'))+' |')
    lines+=['','## 各理解项的梯度大小','',
        '下面列的是加上原生损失权重后的组件梯度范数，不是可相加的百分比。组件之间也可能抵消。','',
        '| 来源 | 更新 | 家族 | classification | thing BCE | thing Dice | stuff BCE | stuff Dice |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    components=('classification','thing_bce','thing_dice','stuff_bce','stuff_dice')
    for row in rows:
        for group in ('adapter','decoder','gaussian_head','anchor_geometry'):
            values=[row['component_gradients'][key].get(group,{}).get('understanding_norm') for key in components]
            lines.append(f"| {row['arm']} | {row['step']} | {group} | "+' | '.join(map(format_number,values))+' |')
    lines+=['','## 概率边界与像素梯度','',
        '取四窗口中最大的 region_mass 像素梯度，列出其对应概率。近边界比例按整个region张量计算，不能等同于错误概率的比例。','',
        '| 来源 | 更新 | 损失项 | 最大像素梯度 | 此处预测概率 | 场景 |',
        '|---|---:|---|---:|---:|---|']
    for row in rows:
        selected=[r for r in local if (r['arm'],r['step'])==(row['arm'],row['step'])]
        for key in ('understanding','thing_bce','stuff_bce'):
            best=max(selected,key=lambda r:r['component_gradients'][key]['region_probability'].get('max_abs_gradient',0))
            p=best['component_gradients'][key]['region_probability']
            lines.append(f"| {row['arm']} | {row['step']} | {key} | {format_number(p.get('max_abs_gradient'))} | {format_number(p.get('probability_at_max_abs_gradient'))} | {best['scene']} |")
    lines+=['','## 静态路由及解释边界','',
        'global_gradient_statistics.json保存同一梯度下adapter系数1/0.01/0的组合范数、与重建梯度的夹角和clip系数；这不是带Adam历史的实际更新预测。',
        '五个原生加权理解项之和与原生总损失相等；梯度之和与总理解梯度已检查。没有重写损失、冻结策略、相机或反馈。',
        '当前联合checkpoint不能精确重现第218次更新的瞬时尖峰；组件归因揭示当前权重上的路径，尚需短对照验证其训练影响。',
        '理解模块在本诊断中冻结；完整训练允许理解模块适配，不能直接把测量外推为全数据的最终任务权衡。','']
    (root/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for col,group in enumerate(('adapter','decoder','gaussian_head')):
        for arm,label in [('pred_joint','Joint checkpoints'),('pred_recon','Reconstruction checkpoints')]:
            rs=[r for r in rows if r['arm']==arm]
            for rowkey,ylabel,ax in [('norm_ratio','U/R gradient norm',axes[0,col]),('cosine','cos(R,U)',axes[1,col])]:
                vals=[r['component_gradients']['understanding'].get(group,{}).get(rowkey) for r in rs]
                ax.plot([r['step'] for r in rs],[float('nan') if v is None else v for v in vals],marker='o',label=label)
                ax.set(title=group,xlabel='Checkpoint updates',ylabel=ylabel);ax.grid(alpha=.25)
        axes[0,col].set_yscale('log');axes[1,col].axhline(0,color='gray',linewidth=.8)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    fig.savefig(root/'gradient_families.png',dpi=180);fig.savefig(root/'gradient_families.svg');plt.close(fig)
    (root/'COMPLETE.json').write_text(json.dumps(dict(status='ZERO_UPDATE_GRADIENT_PROBE_COMPLETED',
        optimizer_updates=0,checkpoint_conditions=8,window_conditions=32,execution_git_sha=global_data['execution_git_sha'],
        max_component_sum_relative_gradient_error=max(r['component_sum_relative_gradient_error'] for r in rows)),indent=2)+'\n')
    print('\n'.join(lines[:12]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    summarize(p.parse_args().root)


if __name__=='__main__':main()
