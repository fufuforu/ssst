#!/usr/bin/env python3
"""Diff the plain-TokenGS and LocusGS reconstruction recipes.

Dumps every Options field that differs between the two presets, the parameter
counts of both models, and the supervision/loss differences, so the comparison
is explicit rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO, log=lambda *a: None)

from tokengs.models import model_registry  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="train_siu3r_plain_tokengs_canonical_recon")
    parser.add_argument("--b", default="train_siu3r_locusgs_recon")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    opt_a = config_defaults[args.a].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=42, num_input_views=2, num_views=4,
    )
    opt_b = config_defaults[args.b].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=42, num_input_views=2, num_views=4,
    )
    da, db = asdict(opt_a), asdict(opt_b)
    diff = {}
    for key in sorted(set(da) | set(db)):
        if da.get(key) != db.get(key):
            diff[key] = {"plain_tokengs": da.get(key), "locusgs": db.get(key)}

    models = {}
    for name, opt in ((args.a, opt_a), (args.b, opt_b)):
        m = model_registry[opt.model_type](opt)
        models[name] = {
            "model_type": opt.model_type,
            "params": sum(p.numel() for p in m.parameters()),
            "params_trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
            "tensors": sum(1 for _ in m.parameters()),
        }
        del m

    print(f"=== distinct Options fields: {args.a}  vs  {args.b} ===")
    for key, val in diff.items():
        print(f"  {key:<34} {val['plain_tokengs']!r:<28} -> {val['locusgs']!r}")
    print("\n=== parameter counts ===")
    for name, info in models.items():
        print(f"  {name:<45} {info['params']:,} params ({info['tensors']} tensors)")

    payload = {"a": args.a, "b": args.b, "diff": diff, "models": models}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n[recipes] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
