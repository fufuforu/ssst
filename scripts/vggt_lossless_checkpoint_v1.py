"""Lossless ZIP-deflated torch checkpoints; unchanged tensors/optimizer/RNG."""
from pathlib import Path
import os
import shutil
import tempfile
import zipfile
import torch


def save(payload, path):
    path = Path(path)
    # Stage ordinary serialization on node-local /tmp, not scarce shared disk.
    # Only one compressed shared temporary exists under the caller's lock.
    temporary = path.with_suffix('.compressed-tmp.pt')
    try:
        with tempfile.TemporaryDirectory(prefix='vggt-checkpoint-') as directory:
            plain = Path(directory) / 'checkpoint.pt'
            torch.save(payload, plain)
            with zipfile.ZipFile(plain, 'r') as source, zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as target:
                for record in source.infolist():
                    with source.open(record) as reader, target.open(record.filename, 'w', force_zip64=True) as writer:
                        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        with temporary.open('rb') as f:
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load(path):
    # Compressed storage cannot use mmap. Standard torch loading supports both
    # old ZIP_STORED and new ZIP_DEFLATED; all scientific bytes are unchanged.
    return torch.load(path, map_location='cpu', weights_only=False)
