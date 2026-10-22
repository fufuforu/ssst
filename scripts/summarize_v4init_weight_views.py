"""CPU-only comparison after baseline and BOTH new official evaluations succeed."""
from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.olda4_ablation_pipeline import read,write,digest,signatures,gt,REFERENCE
from scripts.v4init_isolated_recipe import check_meta
RUN=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_weight_views_20261022_v1')
BASELINE=RUN.parent/'v4init_isolated_noca_v4recipe_20261022_v1'

def main():
    target=RUN/'comparison';target.mkdir(exist_ok=True)
    rows=[];checks={};data={}
    for label,base in [('baseline_U1_2ctx2',BASELINE),('U005_2ctx2',RUN/'U005'),('U1_2ctx4',RUN/'Views2ctx4')]:
        root=base/'eval';cfg=read(root/'configuration.json');identity=read(root/'evaluation/checkpoint_identity.json')
        check_meta(identity['meta'],cfg)
        assert read(base/'train/TRAINING_COMPLETE.json')['progress']==dict(epoch=4,update_in_epoch=0,full_updates=4172)
        assert read(root/'EVALUATION_VERIFIED.json')['status']=='PASSED'
        assert cfg['initial_checkpoint_sha256']=='bf5b5325adcc8346eb89ebdceac36149c510402d18e8c7dd0c8fe8046f245397'
        assert signatures(root)==signatures(REFERENCE) and gt(root)==gt(REFERENCE)
        assert digest(root/'source_checkpoint.pt')==read(root/'FINAL_CHECKPOINT_FROZEN.json')['sha256']
        metrics=read(root/'evaluation/metrics.json');assert metrics['status']=='OFFICIAL_VAL_SCORED' and metrics['windows']==1860 and not metrics['nonfinite_native_paths']
        assert metrics['parameter_updates']==0 and identity['parameter_updates']==0
        assert metrics['official_commit']=='8ea80166be76854f938e90521f1a5b688b755c87'
        d=metrics['official_segmentation']['all'];data[label]=d
        row=dict(model=label,psnr=d['psnr'],absrel=d['absrel'],rmse=d['rmse'])
        for scope in ('context','target'):
            for k in ('miou','pq'):row[scope+'_'+k]=100*d[scope+'_'+k]
            row[scope+'_map']=100*d[scope+'_map']['map']
        rows.append(row);checks[label]=dict(status='PASSED',initial_sha256=cfg['initial_checkpoint_sha256'],progress=dict(epoch=4,full_updates=4172),matched_inputs_and_gt=True,parameter_updates=0)
    fields=list(rows[0]);
    with (target/'vs_isolated_baseline.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
        for r in rows[1:]:w.writerow(dict(model=r['model']+'_minus_baseline',**{k:r[k]-rows[0][k] for k in fields[1:]}))
    pq_names=['cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','desk','curtain','refrigerator','shower curtain','toilet','sink','bathtub','otherfurniture','wall','floor']
    with (target/'per_class_pq.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['scope','class','baseline_U1_2ctx2','U005_2ctx2','U1_2ctx4','U005_delta','views_delta'])
        for scope in ('context','target'):
            for i,name in enumerate(pq_names):
                values=[100*data[r['model']][scope+'_pqs_per_class'][0][i] for r in rows]
                w.writerow([scope,name]+values+[values[1]-values[0],values[2]-values[0]])
    write(target/'SUMMARY_VERIFIED.json',dict(status='PASSED',windows=1860,scenes=312,target_protocol='target-all6',same_zero_and_budget=True,checks=checks,rows=rows,single_seed=True,axis_comparisons=['U005 vs baseline: understanding_weight only','2ctx4 vs baseline: reconstruction training views only']))
    lines=['# V4init隔离：理解权重与训练视角消融','','三组同一零步、4epoch/4172updates；固定1860窗/312场景；Target含context。','','|'+'|'.join(fields)+'|','|'+'|'.join(['---']+['---:']*(len(fields)-1))+'|']
    lines += ['|'+r['model']+'|'+'|'.join(f'{r[k]:.5f}' for k in fields[1:])+'|' for r in rows]
    lines += ['','理解指标以百分数展示；差值见CSV。每组只改变一个实验轴，不能据此断言两项组合的交互；单seed未验证统计稳定性。']
    (target/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
