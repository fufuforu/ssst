#!/usr/bin/env python3
"""Generate ``PlainTokenGSCanonicalRecon`` initial states for a list of seeds.

The paired A/B needs both arms of a given replicate to start from the *same*
weights.  Generating the states up front removes any dependence on which arm
happens to run first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        opt = config_defaults["train_siu3r_plain_tokengs_canonical_recon"].evolve(seed=int(seed))
        torch.manual_seed(int(opt.seed))
        model = model_registry[opt.model_type](opt)
        path = out / f"init_seed{seed}.pt"
        torch.save({"model": model.state_dict(), "seed": int(seed)}, path)
        print(f"[init] seed {seed} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
