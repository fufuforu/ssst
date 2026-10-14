"""Show existing progress and estimated training remainder without Slurm polling."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/space/mawb/ssst/group_plus/vggt_classification_readback_v1/attempts/attempt01'))
    a=p.parse_args()
    for label,arm in zip('BCD',('head_only','object_support','scene_support')):
        path=a.root/arm/'training/progress.json';end=a.root/arm/'evaluation/COMPLETE.json'
        if end.exists():print(label,arm,'训练和128场景官方评测已完成');continue
        if not path.exists():print(label,arm,'启动中；尚无正式更新记录');continue
        progress=json.loads(path.read_text());step=progress['completed_updates']
        if step==300:
            exported=sum((a.root/arm/'evaluation'/f'rank{i:02d}/export_complete.json').exists() for i in range(4))
            scored=sum((a.root/arm/'evaluation'/f'rank{i:02d}/score_complete.json').exists() for i in range(4))
            print(label,arm,f'训练完成；导出{exported}/4，评分{scored}/4')
        else:
            print(label,arm,f'{step}/300步，{progress["completed_exposures"]}/2400窗口曝光；'
                f'平均{progress["seconds_per_update"]:.1f}秒/步，训练剩余约{progress["remaining_seconds"]/60:.1f}分钟（不含评测）')
    print('估计来自已完成训练步；不包含资源等待、模型加载和评测时间。')


if __name__=='__main__':main()
