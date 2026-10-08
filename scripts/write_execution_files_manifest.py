#!/usr/bin/env python3
import argparse,hashlib,json,subprocess
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--attempt',type=Path,required=True);x=a.parse_args();root=Path(__file__).resolve().parents[1]
 out=x.attempt/'execution_files_manifest.json'
 if out.exists():raise RuntimeError(f'refusing to overwrite execution files manifest: {out}')
 files=sorted([*root.glob('scripts/*.py'),*root.glob('tokengs/models/*.py'),*root.glob('tests/*.py'),*root.glob('tests/fixtures/*.json'),*root.glob('slurm/*.sbatch')])
 rows=[{'path':str(p.relative_to(root)),'sha256':sha(p),'size':p.stat().st_size} for p in files]
 siu=Path('/space/mawb/SIU3R');siu_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=siu,text=True).strip();task_repo=Path(__file__).resolve().parents[1]
 if siu_sha!='8ea80166be76854f938e90521f1a5b688b755c87':raise RuntimeError(f'locked SIU3R commit changed: {siu_sha}')
 for rel in ('src/evaluator.py','src/config.py','src/utils/scannet_constant.py'):
  p=siu/rel;rows.append({'path':str(p),'sha256':sha(p),'size':p.stat().st_size,'external_repository':'SIU3R','external_commit':siu_sha})
 for rel in ('scripts/export_object_locus_v3_set_official.py','scripts/eval_object_locus_v3_set.py','scripts/object_locus_v3_set_runtime.py','scripts/object_locus_gc_sweep_runtime.py','scripts/runtime_bootstrap.py','scripts/invoke_siu3r_official_evaluator.py'):
  p=task_repo/rel;blob=subprocess.check_output(['git','rev-parse',f'68e2376ffef5171206f458d82a063aac35e36891:{rel}'],cwd=task_repo,text=True).strip()
  rows.append({'path':str(p),'sha256':sha(p),'size':p.stat().st_size,'external_repository':'task-worktree-source','external_commit':'68e2376ffef5171206f458d82a063aac35e36891','git_blob':blob})
 code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 out.write_text(json.dumps({'code_sha':code,'siu3r_commit':siu_sha,'files':rows},indent=2)+'\n')
 print(json.dumps({'code_sha':code,'files':len(rows)}))
if __name__=='__main__':main()
