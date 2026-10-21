"""CPU preflight for the semantic-resolution-256 snapshot.

1. SemanticMemory parameter names/shapes must match the A4 zero-step checkpoint.
2. Forward at 256 must produce (b, v, 256, 256, 256) on CPU.
"""
import sys
from pathlib import Path

ROOT = Path('/space/mawb/ssst_sem_res256_code_20261019')
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from tokengs.models.object_locus_dual_stream_v1 import SemanticMemory  # noqa: E402

CKPT = Path('/space/mawb/ssst/group_plus/unified_query_training_v1/'
            'attempt05_legacy_dual_superpoint_full_node13_8gpu/checkpoint_untrained_initial.pt')


def main():
    mod = SemanticMemory()
    local = {f'semantic_memory.{k}': tuple(v.shape) for k, v in mod.state_dict().items()}
    ck = torch.load(CKPT, map_location='cpu', weights_only=False, mmap=True)
    ref = {k: tuple(v.shape) for k, v in ck['model'].items() if k.startswith('core.semantic_memory.')}
    local_norm = {k.replace('semantic_memory.', 'core.semantic_memory.'): v for k, v in local.items()}
    missing = sorted(set(ref) - set(local_norm))
    extra = sorted(set(local_norm) - set(ref))
    shape_diff = {k: (ref[k], local_norm[k]) for k in set(ref) & set(local_norm) if ref[k] != local_norm[k]}
    print('ckpt tensors:', len(ref), 'module tensors:', len(local_norm),
          'missing:', len(missing), 'extra:', len(extra), 'shape_diff:', len(shape_diff))
    if missing or extra or shape_diff:
        print('MISSING', missing[:5], 'EXTRA', extra[:5], 'DIFF', list(shape_diff.items())[:3])
        return 1

    layers = [torch.randn(1, 2, 1024, 2048) for _ in range(4)]
    rgb = torch.rand(1, 2, 3, 256, 256)
    mod.eval()
    with torch.no_grad():
        out = mod(layers, rgb)
    print('forward output:', tuple(out.shape), 'finite:', bool(torch.isfinite(out).all()))
    ok = tuple(out.shape) == (1, 2, 256, 256, 256)
    print('PREFLIGHT', 'PASSED' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
