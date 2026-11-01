from pathlib import Path
import hashlib, importlib.util, sys, types

PIN = Path('/space/mawb/.cache/torch_extensions_backup/gsplat_cuda_partial_1406/gsplat_cuda.so')
PIN_SHA = '82403af8b36fb4b64774afc7462ea6611896fcfe7a02785098bce9ff50d8d751'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def imports():
    try:
        import fused_ssim_cuda
    except ImportError:
        module = types.ModuleType('fused_ssim_cuda')
        def unavailable(*a, **k):
            raise RuntimeError('Import-only shim called; ScanNet uses explicit differentiable SSIM.')
        module.fusedssim = module.fusedssim_backward = unavailable
        sys.modules['fused_ssim_cuda'] = module


def renderer():
    import torch
    if sha(PIN) != PIN_SHA:
        raise ValueError('Renderer SHA differs')
    if 'gsplat_cuda' not in sys.modules:
        spec = importlib.util.spec_from_file_location('gsplat_cuda', PIN)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules['gsplat_cuda'] = module
    module = sys.modules['gsplat_cuda']
    if Path(module.__file__).resolve() != PIN.resolve():
        raise ValueError('Different renderer already loaded')
    import gsplat
    sys.modules['gsplat.csrc'] = module
    gsplat.csrc = module
    for name in ('spherical_harmonics_fwd', 'spherical_harmonics_bwd'):
        if not hasattr(module, name):
            raise ValueError('Native SH kernel absent: '+name)
    return dict(path=str(PIN), sha256=PIN_SHA)
