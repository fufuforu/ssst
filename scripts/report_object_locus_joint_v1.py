"""Fixed epoch64 paired inference and <=28MiB verified result bundle."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
import csv
import json
import subprocess
import tempfile
import zipfile
import numpy as np
from scripts.object_locus_joint_v1_runtime import REPORTS_DEFAULT,SPLITS,write_json,sha256_file
from scripts.train_object_locus_v3_set import _scope_official_metrics


def numeric(value):
    return isinstance(value,(int,float)) and np.isfinite(value) and value!=-1


def bootstrap(deltas,seed):
    d=np.asarray(deltas,dtype=np.float64)
    rng=np.random.default_rng(seed)
    samples=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(1)
    return {'macro_mean_difference':float(d.mean()),
            'CI95':np.quantile(samples,[.025,.975],method='linear').tolist(),
            'seed':seed,'draws':10000,'scene_count':len(d)}


def main():
    root=REPORTS_DEFAULT;pair=root/'pair';pair.mkdir(parents=True,exist_ok=True)
    validation=root/'validation';engineering=all(
        (validation/f).exists() and json.loads((validation/f).read_text()).get('passed')
        for f in ('smoke.json','m2.json'))
    endpoints={};starts={};summaries={};missing=[];rows=[]
    for arm in ('control','joint'):
        for name,destination in [('curves_step_3584.json',endpoints),('curves_step_0000.json',starts),
                                 ('training_summary.json',summaries)]:
            path=root/arm/name
            if path.exists():destination[arm]=json.loads(path.read_text())
            else:missing.append(str(path))
        for path in sorted((root/arm).glob('curves_step_*.json')):
            node=json.loads(path.read_text())
            for split,x in node['splits'].items():
                for scope,m in x['local'].items():
                    row={'arm':arm,'step':node['step'],'split':split,'scope':scope,
                         **_scope_official_metrics(x.get('official',{}),scope)}
                    for key,value in m.items():
                        if isinstance(value,(str,int,float)):row[key]=value
                        elif key in ('candidate_ap','candidate_ca','candidate_cw','panoptic_ca','panoptic_cw'):
                            row.update({f'{key}_{k}':v for k,v in value.items()})
                    rows.append(row)
    write_json(pair/'task_metrics.json',rows)
    keys=sorted({k for row in rows for k in row})
    with (pair/'task_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    complete=all(a in summaries and summaries[a]['completed_updates']==3584 and summaries[a]['finite']
                 for a in ('control','joint'))
    statistics={};checks={};sources=[]
    if all(a in endpoints and a in starts for a in ('control','joint')):
        for index,split in enumerate(SPLITS):
            c=endpoints['control'];j=endpoints['joint']
            scenes=sorted(c.get('per_scene',{}).get(split,{}))
            if set(scenes)!=set(j.get('per_scene',{}).get(split,{})) or not scenes:
                missing.append(f'{split} per-scene identities');continue
            if split in SPLITS[:2] and len(scenes)!=8:
                missing.append(f'{split} requires 8 scenes');continue
            deltas=[]
            for scene in scenes:
                values=[]
                for arm,node in [('control',c),('joint',j)]:
                    official=node['per_scene'][split][scene].get('official',{})
                    metric=_scope_official_metrics(official,'context')
                    value=metric['scope_official_ap50'];values.append(value)
                    sources.append({'arm':arm,'split':split,'scene':scene,**metric})
                if not all(numeric(v) for v in values):missing.append(f'{split}/{scene} official AP50 UNDEFINED/MISSING')
                else:deltas.append(values[1]-values[0])
            if len(deltas)==len(scenes):
                statistics[split]={'scene_order':scenes,'paired_differences':deltas,
                                   **bootstrap(deltas,20261002+index)}
            else:statistics[split]={'status':'UNDEFINED/MISSING; fixed bootstrap unavailable'}
        for split in SPLITS:
            for scope in ('context','novel'):
                try:
                    c=endpoints['control']['splits'][split]['local'][scope]['psnr']
                    j=endpoints['joint']['splits'][split]['local'][scope]['psnr']
                    z=starts['joint']['splits'][split]['local'][scope]['psnr']
                    checks[f'{split}_{scope}_PSNR_vs_control']=j-c>=-.5
                    checks[f'{split}_{scope}_PSNR_vs_step0']=j-z>=-.5
                except KeyError:missing.append(f'{split}/{scope} PSNR')
        for split,margin in [('train_all56',0),('same_scene_holdout8',-.01)]:
            values=[_scope_official_metrics(endpoints[a]['splits'][split].get('official',{}),'context')['scope_official_ap50'] for a in ('control','joint')]
            stats=statistics.get(split,{})
            if all(numeric(v) for v in values) and 'CI95' in stats:
                delta=values[1]-values[0]
                checks[f'{split}_pooled_AP50']=delta>0 if margin==0 else delta>=margin
                checks[f'{split}_CI_lower']=stats['CI95'][0]>0 if margin==0 else stats['CI95'][0]>=margin
                stats['pooled_AP50_difference']=delta
            else:missing.append(f'{split} primary official endpoint')
    grade='B' if engineering and complete and not missing and checks and all(checks.values()) else (
        'A' if engineering and complete and not missing else 'C')
    m1_path=validation/'m1.json'
    m1=json.loads(m1_path.read_text()) if m1_path.exists() else []
    failed_m1=next((r for r in m1 if not r['passed']),None)
    result={'M1_failure':failed_m1['first_divergence'] if failed_m1 else None,
            'engineering' :'A' if engineering else 'C','task':grade,'complete':complete,
            'checks':checks,'statistics':statistics,'missing':missing,
            'caveat':'One training run per arm; per-scene CI does not estimate training variance.',
            'strict_instance_locality_claim':False,'unregistered_scientific_changes':False}
    write_json(pair/'paired_report.json',result);write_json(pair/'official_per_scene_sources.json',sources)
    lines=['# Object-Locus Joint V1 配对报告','',f'工程：{result["engineering"]}；任务：{grade}。',
        '','每臂仅一次训练，未估计训练 run-to-run 方差。共同 conditioning 不等于严格实例局部。',
        '', 'B 仅支持固定小规模训练条件下的任务价值；dev8/val32 不提升时没有未见场景泛化证据。',
        '',json.dumps(result,ensure_ascii=False,indent=2)]
    (pair/'analysis_report.md').write_text('\n'.join(lines)+'\n')
    (pair/'source.patch').write_text(subprocess.check_output(
        ['git','-C',str(REPO),'diff','b00b94fdc45af6d2f78c55f05671aaa75906204f','--',
         'tokengs/models/object_locus_joint_v1.py','tokengs/models/__init__.py','tokengs/options.py',
         'tokengs/rendering/gs.py','tokengs/models/canonical_recon.py',
         'scripts/*object_locus_joint_v1*','tests/test_object_locus_joint_v1_contracts.py',
         'docs/object_locus_joint_v1_codex_spec.md'],text=True))
    from scripts.object_locus_joint_v1_runtime import IMPLEMENTATION_FILES
    patch=pair/'source.patch'
    with patch.open('a') as stream:
        for name in IMPLEMENTATION_FILES:
            tracked=subprocess.run(['git','-C',str(REPO),'ls-files','--error-unmatch',name],capture_output=True)
            if tracked.returncode:
                diff_process=subprocess.run(['git','diff','--no-index','/dev/null',str(REPO/name)],capture_output=True,text=True)
                stream.write(diff_process.stdout)
    target=pair/'result_bundle.zip' 
    files=[p for p in root.rglob('*') if p.is_file() and p!=target and p.name!='bundle_verification.json' and
        'official' not in p.relative_to(root).parts and p.suffix not in ('.pt','.pth','.ckpt','.zip')]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(REPO/'docs/object_locus_joint_v1_codex_spec.md','spec.md')
        for name in IMPLEMENTATION_FILES:archive.write(REPO/name,'source/'+name)
        for path in files:archive.write(path,str(path.relative_to(root)))
    if target.stat().st_size>28*1024**2:raise RuntimeError('bundle exceeds 28MiB; packaging repair required, no retraining')
    with tempfile.TemporaryDirectory(prefix='joint_v1_unpack_') as directory:
        with zipfile.ZipFile(target) as archive:
            if archive.testzip():raise RuntimeError('ZIP CRC failed')
            archive.extractall(directory)
            for name in archive.namelist():
                if not (Path(directory)/name).is_file():raise RuntimeError('ZIP extraction failed')
    write_json(pair/'bundle_verification.json',{'bytes':target.stat().st_size,'sha256':sha256_file(target),'unpacked':True})
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
