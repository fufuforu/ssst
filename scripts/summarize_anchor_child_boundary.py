"""Validate coverage, write paired scene bootstrap tables and compact publication report."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,csv,hashlib,json,shutil,subprocess
import numpy as np


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['smoke','full'],required=True)
    args=p.parse_args();root=args.output;config=json.loads(args.config.read_text());n=2 if args.mode=='smoke' else 128
    labels=list(config['sources']);data={};receipts={}
    for label in labels:
        done=sorted(root.glob(f'{label}_rank*_complete.json'));assert len(done)==(1 if args.mode=='smoke' else 8)
        receipts[label]=[json.loads(p.read_text()) for p in done]
        assert all(r['backward_calls']==r['parameter_updates']==0 and r['weight_sha256_before']==r['weight_sha256_after'] for r in receipts[label])
        rows=[json.loads(Path(record['path']).read_text()) for r in receipts[label] for record in r['records']]
        rows=sorted(rows,key=lambda r:r['index']);assert [r['index'] for r in rows]==list(range(n))
        assert len({r['scene'] for r in rows})==n
        for row in rows:
            folder=root/label/f'window_{row["index"]:03d}_{row["scene"]}'
            for vi in range(2):
                assert (folder/f'context{vi}_sixpanel.png').is_file()
                assert (folder/f'context{vi}_boundary.npz').is_file()
        data[label]=rows
    for a,b in zip(data[labels[0]],data[labels[1]]):
        for k in ('scene','context','input_rgb_sha256','gt_sha256','source_superpoint_sha256'):
            assert a[k]==b[k],f'Paired protocol mismatch: {k}'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    receipt=dict(status='SMOKE_PASSED' if args.mode=='smoke' else 'DIAGNOSTICS_COMPLETE',windows_per_checkpoint=n,
                 context_views_per_checkpoint=2*n,backward_calls=0,parameter_updates=0,config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                 code_commit=commit,checkpoints=config['sources'],rank_receipts=receipts,
                 equivalence_checks={k:[r['equivalence'] for r in rows] for k,rows in data.items()})
    if args.mode=='smoke':
        dump(root/'SMOKE_PASSED.json',receipt)
        return
    # Flatten per-window means, keeping independent scenes as bootstrap units.
    def flattened(row):
        values={}
        for k,v in row['feature'].items():
            if isinstance(v,dict):values['feature.'+k]=v['mean']
            elif isinstance(v,(float,int)):values['feature.'+k]=v
        for k in row['boundary'][0]:
            x=[v[k] for v in row['boundary'] if v[k] is not None]
            values['boundary.'+k]=float(np.mean(x)) if x else None
        for scope in ('all','cross'):
            for stage in ('child','final','residual'):
                base=f'feature.{scope}.purity0.7.{stage}.cosine_'
                a,b=values[base+'same'],values[base+'different']
                values[f'feature.{scope}.{stage}.cosine_gap_same_minus_different']=a-b if a is not None and b is not None else None
            base=f'feature.{scope}.purity0.7.membership_js_';a,b=values[base+'different'],values[base+'same']
            values[f'feature.{scope}.membership_js_gap_different_minus_same']=a-b if a is not None and b is not None else None
        return values
    flat={k:[flattened(r) for r in rows] for k,rows in data.items()};comparison=[]
    rng=np.random.default_rng(20261021)
    for key in sorted(flat[labels[0]][0]):
        pairs=[(a[key],b[key]) for a,b in zip(flat[labels[0]],flat[labels[1]]) if a[key] is not None and b[key] is not None]
        if not pairs:continue
        x=np.asarray(pairs);delta=x[:,1]-x[:,0]
        boot=delta[rng.integers(len(delta),size=(4000,len(delta)))].mean(1)
        lo,hi=np.percentile(boot,[2.5,97.5])
        comparison.append(dict(metric=key,paired_scenes=len(x),A4=float(x[:,0].mean()),A8=float(x[:,1].mean()),A8_minus_A4=float(delta.mean()),ci95_low=float(lo),ci95_high=float(hi)))
    with (root/'paired_scene_statistics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    with (root/'anchor_statistics.csv').open('w') as f:
        fields=['checkpoint','window','scene']+list(data[labels[0]][0]['anchors'][0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for label,rows in data.items():
            for row in rows:
                for anchor in row['anchors']:w.writerow(dict(checkpoint=label,window=row['index'],scene=row['scene'],**anchor))
    by={r['metric']:r for r in comparison}
    cosine=by.get('feature.cross.final.cosine_gap_same_minus_different');membership=by.get('feature.cross.membership_js_gap_different_minus_same')
    supported=cosine and membership and cosine['ci95_low']>0 and membership['ci95_low']>0
    worsened=cosine and membership and cosine['ci95_high']<0 and membership['ci95_high']<0
    discrimination='跨实例区分度有一致改善证据' if supported else '跨实例区分度有一致退化证据' if worsened else '证据不足以认定跨实例区分度一致改善；请结合各阶段与覆盖率读表'
    # Conservative preregistered pilot gate, not a learned threshold or loss-benefit guarantee.
    ready={}
    for label in labels:
        rows=flat[label];f1=np.array([r['boundary.f1_2px'] for r in rows]);ap=np.array([r['boundary.continuous_ap_2px'] for r in rows if r['boundary.continuous_ap_2px'] is not None]);interior=np.array([r['boundary.interior_raw_response_fraction'] for r in rows]);activation=np.array([r['boundary.interior_threshold_fraction'] for r in rows if r['boundary.interior_threshold_fraction'] is not None])
        low=float(np.percentile(f1[rng.integers(len(f1),size=(4000,len(f1)))].mean(1),2.5))
        ready[label]=dict(mean_f1_2px=float(f1.mean()),f1_lower95=low,mean_ap_2px=float(ap.mean()) if len(ap) else None,
                          mean_interior_response_fraction=float(interior.mean()),mean_interior_threshold_fraction=float(activation.mean()) if len(activation) else None)
        z=ready[label];z['ready_for_small_boundary_loss_pilot']=bool(z['mean_f1_2px']>=.70 and low>=.65 and len(ap) and ap.mean()>=.5 and interior.mean()<=.2 and len(activation) and activation.mean()<=.1)
    dump(root/'conclusions.json',dict(discrimination=discrimination,boundary=ready,causal_limit='Fixed checkpoint comparison is observational; shared-anchor suppression is not established by cosine alone.'))
    publication=root/'publication';publication.mkdir(exist_ok=True)
    for name in ('paired_scene_statistics.csv','anchor_statistics.csv','conclusions.json'):shutil.copy2(root/name,publication/name)
    # First four fixed cohort windows, chosen before observing output; 16 paired context montages.
    for label,rows in data.items():
        for row in rows[:4]:
            folder=root/label/f'window_{row["index"]:03d}_{row["scene"]}'
            for image in folder.glob('*sixpanel.png'):shutil.copy2(image,publication/f'{label}_{row["index"]:03d}_{image.name}')
    lines=['# A4 → A4+4 Anchor/Child 与实例边界诊断','',f'固定 128 个不同验证场景，每个 checkpoint 256 个 Context 视角；执行代码 `{commit}`。',
           '','## 实际结论','',discrimination+'。','共享 Anchor 是否构成容量瓶颈仍需因果消融；本轮不训练，不以原始 child cosine 单独下结论。','',
           '|checkpoint|BF1@2px|BF1 lower95|连续 AP@2px|内部响应占比|内部激活率|可进入小规模 Boundary Loss 试验|','|---|---:|---:|---:|---:|---:|---|']
    for label,z in ready.items():lines.append(f'|{label}|{z["mean_f1_2px"]:.4f}|{z["f1_lower95"]:.4f}|{z["mean_ap_2px"]}|{z["mean_interior_response_fraction"]:.4f}|{z["mean_interior_threshold_fraction"]}|{z["ready_for_small_boundary_loss_pilot"]}|')
    lines+=['','所有配对统计及场景 bootstrap CI 见 paired_scene_statistics.csv；逐 Anchor 统计见 anchor_statistics.csv。','GT 归属仅使用完整场景 alpha-compositing 贡献，模糊/不可见 GS 不强行标注；同/异实例配对使用纯度 0.5/0.7/0.9。','原始连续响应与 soft map 为每视角 float32 NPZ，全部六图本地保留；这里选前四个固定窗展示。','反向调用 0，参数更新 0，各 rank 权重哈希见 COMPLETE.json。',
             '','本门槛只用于建议小规模 Boundary-Aware RGB Loss 试验，不能证明该损失有效。报告及示例需在用户通知完成后推送 GitHub。']
    (publication/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(root/'COMPLETE.json',receipt)
    shutil.copy2(root/'COMPLETE.json',publication/'COMPLETE.json')

if __name__=='__main__':main()
