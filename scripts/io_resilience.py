"""Bounded retries for transient shared-filesystem faults; never skip records."""
import errno
import functools
import sys
import time
from pathlib import Path

TRANSIENT_ERRNOS={errno.ETIMEDOUT,errno.ESTALE,errno.EIO,errno.ECONNRESET}

def retry_io(operation,*,label,attempts=3,delays=(1.,3.),sleep=time.sleep,log=None):
    if attempts<1:raise ValueError('attempts must be positive')
    if log is None:log=lambda message:print(message,file=sys.stderr,flush=True)
    for attempt in range(attempts):
        try:return operation()
        except OSError as error:
            if error.errno not in TRANSIENT_ERRNOS or attempt+1==attempts:raise
            delay=delays[min(attempt,len(delays)-1)] if delays else 0.
            log(f'[io-retry] {label}: errno={error.errno}; attempt {attempt+1}/{attempts}, retry after {delay}s')
            sleep(delay)

@functools.lru_cache(maxsize=2048)
def scene_depth_frames(scene_path):
    """Same full sorted directory listing as the provider; cache only successes."""
    depth=Path(scene_path)/'depth'
    # Batch-level retry covers directory enumeration and all subsequent reads.
    return tuple(sorted(int(x.stem) for x in depth.glob('*.png')))

def startup_retry_allowed(returncode,tail,run_root,attempt,attempts=3):
    """Only retry torch bootstrap timeouts before training/RNG/optimizer receipts."""
    root=Path(run_root)
    began=any((root/name).exists() for name in ('STARTED.json','checkpoint_latest.pt','FIRST_UPDATE_VERIFIED.json'))
    return bool(returncode and attempt<attempts and not began and 'TimeoutError' in tail and '[Errno 110]' in tail)
