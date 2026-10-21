"""Stream training output; retry only an Errno110 bootstrap with no STARTED."""
import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.io_resilience import startup_retry_allowed

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('command required')
    history=[]
    for attempt in range(1,4):
        tail=deque(maxlen=500)
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True);tail.append(line)
        code=process.wait();retry=startup_retry_allowed(code,''.join(tail),args.run_root,attempt)
        history.append(dict(attempt=attempt,exit_code=code,retry_bootstrap_only=retry))
        (args.run_root/'BOOTSTRAP_ATTEMPTS.json').write_text(json.dumps(history,indent=2)+'\n')
        if not retry:return code
        print(f'[bootstrap] pre-STARTED filesystem timeout, bounded retry {attempt}/3',flush=True)
        time.sleep(3*attempt)
    return code
if __name__=='__main__':sys.exit(main())
