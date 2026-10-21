"""Bounded read of existing JSONL prefixes; never waits for a training job."""
import argparse,csv,json,math,statistics
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path

def percentile(values,p):
    x=sorted(values);at=(len(x)-1)*p;lo=math.floor(at);hi=math.ceil(at)
    return x[lo]+(x[hi]-x[lo])*(at-lo)

def summarize(run,output):
    meta=json.loads((run/'STARTED.json').read_text());world=meta['world_size'];batch=meta['global_batch']
    if world<1 or batch%world:raise ValueError('Rank count must divide global batch')
    microbatches=batch//world
    paths=[run/f'train_rank{rank:02d}.jsonl' for rank in range(world)]
    sizes=[p.stat().st_size for p in paths];ranks=[]
    for p,size in zip(paths,sizes):
        data={}
        with p.open('rb') as f:lines=f.read(size).splitlines()
        for line in lines:
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if 'num_thing_queries' not in row:continue
            if 'microbatch_query_counts' in row:
                counts=[x['num_superpoints'] for x in row['microbatch_query_counts']]
                if len(counts)!=microbatches:raise ValueError('Missing per-scene microbatch counts')
            elif microbatches==1:counts=[row['num_thing_queries']]
            else:raise ValueError('Historical scalar microbatch mean cannot recover per-scene distribution')
            if any(not math.isfinite(n) or n<1 or n!=int(n) for n in counts):raise ValueError('Not individual integer query counts')
            counts=tuple(int(n) for n in counts);step=row['update']
            if step in data and data[step]!=counts:raise ValueError('Conflicting duplicate step')
            data[step]=counts
        ranks.append(data)
    common=sorted(set.intersection(*(set(x) for x in ranks)))
    if not common:raise ValueError('No complete global iterations')
    values=[n for step in common for rank in ranks for n in rank[step]]
    totals=[sum(n for rank in ranks for n in rank[step]) for step in common]
    histogram=Counter(values)
    result=dict(run=str(run),world_size=world,global_batch=batch,log_prefix_bytes=sizes,first_update=common[0],last_complete_update=common[-1],complete_iterations=len(common),scene_samples=len(values),
        per_scene_superpoint_count=dict(mean=sum(values)/len(values),median=statistics.median(values),p05=percentile(values,.05),p95=percentile(values,.95),minimum=min(values),maximum=max(values)),
        per_global_iteration_sum=dict(mean=sum(totals)/len(totals),minimum=min(totals),maximum=max(totals)),
        histogram={str(k):histogram[k] for k in sorted(histogram)},count_definition='Initial superpoints / thing queries, excluding two stuff queries per scene. Reconstruction Q+P expands anchors separately; counts here retain each individual scene.')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'per_iteration_counts.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['update']+[f'rank{r:02d}_micro{m:02d}_scene_superpoints' for r in range(world) for m in range(microbatches)]+['global_sum','scene_mean'])
        for step in common:
            counts=[n for rank in ranks for n in rank[step]];w.writerow([step]+counts+[sum(counts),sum(counts)/len(counts)])
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('runs',type=Path,nargs='+');a=p.parse_args()
    results=[]
    for i,run in enumerate(a.runs):
        result=summarize(run,a.output/f'run{i:02d}_{run.name}');results.append(result)
        print(json.dumps({k:result[k] for k in ('run','complete_iterations','scene_samples','per_scene_superpoint_count','per_global_iteration_sum')}),flush=True)
    (a.output/'all_runs.json').write_text(json.dumps(dict(read_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),bounded_prefix_read=True,runs=results),indent=2)+'\n')
if __name__=='__main__':main()
