# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime bootstrap shared by the SSST entry points.

Two cluster facts make this necessary and it must run before `tokengs.rendering`
is imported:

* the fused-ssim CUDA extension is built from an unreachable GitHub source, so
  `tokengs.rendering.fused_ssim` cannot import its symbol on CUDA machines. SSST
  trains with `lambda_ssim = 0`; the local stand-in only satisfies the import.
* gsplat JIT-compiles its CUDA extension when no build is cached, which fails
  on the compute nodes (CUDA headers are not visible to the JIT). If a
  precompiled `gsplat_cuda.so` is available it is loaded directly, as the other
  TokenGS runs on this cluster do.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

DEFAULT_GSPLAT_CANDIDATES = (
    "/space/mawb/tokengs_siu3r_joint_v1/.torch_extensions_v4/gsplat_cuda/gsplat_cuda.so",
    "/space/mawb/tokengs_siu3r_joint_v1/.torch_extensions_v3/gsplat_cuda/gsplat_cuda.so",
    # Built against an older glibc, so it also loads on the nodes whose system
    # image cannot load the _v4/_v3 extensions (the renderer is used unchanged).
    "/space/mawb/.cache/torch_extensions_backup/gsplat_cuda_partial_1406/gsplat_cuda.so",
)


def ensure_fused_ssim_importable(repo_root: Path, log=print) -> None:
    try:
        import fused_ssim_cuda  # noqa: F401
        return
    except ImportError:
        pass
    shim_dir = repo_root / "third_party" / "shims"
    if str(shim_dir) not in sys.path:
        sys.path.append(str(shim_dir))
    log(f"[runtime] fused-ssim extension unavailable; using import stand-in {shim_dir}")


def load_cached_gsplat_extension(log=print) -> str | None:
    if "gsplat_cuda" in sys.modules:
        return getattr(sys.modules["gsplat_cuda"], "__file__", None)
    # Importing torch first makes libc10/libtorch resolvable for the extension.
    import torch  # noqa: F401

    candidates = []
    env_path = os.environ.get("GSPLAT_PRECOMPILED_SO")
    if env_path:
        candidates.append(env_path)
    candidates.extend(DEFAULT_GSPLAT_CANDIDATES)
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("gsplat_cuda", path)
        if spec is None or spec.loader is None:
            continue
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as error:  # wrong libc/ABI on this host: fall through
            log(f"[runtime] cannot load {path} ({error}); trying the next candidate")
            continue
        sys.modules["gsplat_cuda"] = module
        import gsplat

        sys.modules.setdefault("gsplat.csrc", module)
        setattr(gsplat, "csrc", module)
        log(f"[runtime] preloaded precompiled gsplat extension from {path}")
        return str(path)
    log("[runtime] no precompiled gsplat extension found; gsplat will JIT-build on first use")
    return None


def prepare_runtime(repo_root: str | Path, log=print) -> None:
    repo_root = Path(repo_root)
    ensure_fused_ssim_importable(repo_root, log=log)
    load_cached_gsplat_extension(log=log)
