"""Read-only progress/ETA for the four interaction diagnostics."""
import json
from pathlib import Path
import subprocess
from datetime import timedelta

REPORT=Path('/space/mawb/ssst/group_plus/vggt_interaction_controls_v1')
attempt=REPORT/'attempts/attempt02'
receipt=attempt/'formal_submission.json'
names={'bidirectional':'当前双向','forward_only':'仅 GS→query',
       'gated_feedback':'门控反馈','independent_feedback':'独立反馈注意力'}
if not receipt.exists():
    print('正式训练尚未提交，四卡预检进度：')
    for arm,name in names.items():
        root=attempt/'four_smoke'/arm
        training=(root/'smoke_report.json').exists()
        scored=(root/'evaluation_preflight/score_complete.json').exists()
        print(name, '训练及评分通过' if training and scored else '训练通过，评分中' if training else '等待或初始化')
else:
    r=json.loads(receipt.read_text());run=Path(r['run_root'])
    jobs=r['runs']+r['evaluations'];ids=','.join(j['job_id'] for j in jobs)
    rows=subprocess.check_output(['squeue','-h','-j',ids,'-o','%i|%T|%M|%N|%R'],text=True).splitlines()
    states={v[0]:v[1:] for v in (line.split('|') for line in rows)}
    print('四组均为4张4090、global batch8、500更新/4000曝光；ETA不含排队。')
    for j in r['runs']:
        arm=j['arm'];path=run/arm/'progress.json'
        p=json.loads(path.read_text()) if path.exists() else {}
        state=states.get(j['job_id'],['已退出队列；请查sacct','','',''])
        step=p.get('completed_updates',0)
        eta=str(timedelta(seconds=round(p['estimated_remaining_seconds']))) if p else '初始化后估计'
        print('{}  job={}  {}  节点={}  更新={}/500  曝光={}/4000  预计剩余={}'.format(
            names[arm],j['job_id'],state[0],state[2] or '待分配',step,step*8,eta))
    print('自动评测：'+', '.join(j['arm']+'='+j['job_id'] for j in r['evaluations']))
    print('结果目录：'+str(attempt/'evaluation'))
