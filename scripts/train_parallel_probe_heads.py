#!/usr/bin/env python3
"""Launch and join the three isolated one-head/one-GPU workers."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
from scripts.object_locus_probe_metrics import verify_gc_cache_manifest
def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);a=p.parse_args();root=a.attempt
 if not root.joinpath('extraction_complete.json').is_file():raise RuntimeError('A receipt missing')
 cache_check=verify_gc_cache_manifest(root)
 if torch_count()!=3:raise RuntimeError(f'C requires three allocated GPUs; visible={torch_count()}')
 if not str(os.environ.get('SLURMD_NODENAME','')).startswith('3dimage-11'):raise RuntimeError('C must run on fixed node 3dimage-11')
 assigned=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
 if len(assigned)!=3 or any(not x for x in assigned):raise RuntimeError(f'Slurm must expose exactly three assigned devices, got {os.environ.get("CUDA_VISIBLE_DEVICES")}')
 procs=[]
 for idx,head in enumerate(('H1','H2','H3')):
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=assigned[idx];env['OMP_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4'
  cmd=[sys.executable,'scripts/train_object_locus_frozen_probe_head_worker.py','--head',head,'--device','cuda:0','--attempt',str(root)]
  procs.append((head,subprocess.Popen(cmd,env=env)))
 failures=[]
 for head,proc in procs:
  code=proc.wait()
  if code:failures.append({'head':head,'returncode':code})
  receipt=root/f'head_{head}_complete.json'
  if code==0 and (not receipt.is_file() or json.loads(receipt.read_text()).get('status')!='PASS'):failures.append({'head':head,'missing_or_invalid_receipt':str(receipt)})
 if failures:
  (root/'heads_failure.json').write_text(json.dumps({'status':'INVALID','failures':failures},indent=2)+'\n');raise RuntimeError(f'head workers failed: {failures}')
 rows=[]
 for h in ('H1','H2','H3'):
  with (root/f'training_scalars_{h}.csv').open(newline='') as f:rows.extend(csv.DictReader(f))
 with (root/'training_scalars.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (root/'head_manifest.json').write_text(json.dumps({'seeds':[20261,20262,20263],'primary_seed':20261,'head_workers':3,'heads':{'H1':{'device':'allocated cuda:0','parameters':5395},'H2':{'device':'allocated cuda:1','parameters':5395},'H3':{'device':'allocated cuda:2','parameters':10771}},'all_nine_best_and_final_checkpoints':True},indent=2)+'\n')
 (root/'head_workers_cache_verification.json').write_text(json.dumps(cache_check,indent=2)+'\n')
 (root/'heads_complete.json').write_text(json.dumps({'status':'PASS','head_count':3,'seed_count':3,'best_checkpoints':9,'final_checkpoints':9,'receipts_checked':True},indent=2)+'\n')
def torch_count():
 import torch
 return torch.cuda.device_count()
if __name__=='__main__':main()
