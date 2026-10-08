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
 expected=__import__('os').environ['TASK_CODE_SHA'];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 if head!=expected:raise RuntimeError(f'fixed execution SHA mismatch: {head} != {expected}')
 manifest=json.loads((x.attempt/'execution_files_manifest.json').read_text());fail=[]
 for r in manifest['files']:
  p=Path(r['path']) if r.get('external_repository') else root/r['path']
  if not p.is_file() or sha(p)!=r['sha256']:fail.append(r['path'])
 if fail:raise RuntimeError(f'execution files manifest mismatch: {fail}')
 print(json.dumps({'status':'PASS','code_sha':head,'verified_files':len(manifest['files'])}),flush=True)
if __name__=='__main__':main()
