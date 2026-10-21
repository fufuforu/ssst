"""Test-session runtime setup.

The fused-ssim CUDA extension is not importable on every node, and gsplat may
need a precompiled extension. `prepare_runtime` only supplies import stand-ins
when the real extensions are missing and logs what it substituted.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO_ROOT)
