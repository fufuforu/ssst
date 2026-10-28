"""Bounded reread for transient hash failures; immutable expected hash is mandatory."""
from pathlib import Path
from scripts.frozen_iou_head_core import digest


def verified_forward(cache, record):
    path=Path(cache)/record['official_forward_file'];expected=record['official_forward_sha256']
    observed=[]
    for attempt in range(2):
        actual=digest(path);observed.append(actual)
        if actual==expected:
            if attempt:
                print(f'[integrity] reread matched: {path}; expected={expected}; first={observed[0]}',flush=True)
            return path
    raise ValueError(f'Same-forward tensor file changed: path={path}; expected={expected}; observed={observed}')
