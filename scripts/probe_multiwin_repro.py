#!/usr/bin/env python3
"""Read-only probe: is the token-contribution map computation reproducible?

`token_maps` accumulates the per-token pixel contributions with
`Tensor.index_add_` (and `alpha +=`), which on CUDA uses non-deterministic
atomics.  If two identical calls in ONE process already differ, then any two
training runs of this harness differ at the bit level and can only be compared
statistically.  Prints hashes; trains nothing.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_bootstrap import prepare_runtime  # noqa: E402

prepare_runtime(REPO)

from tokengs.models import model_registry  # noqa: E402
from tokengs.models.instance_query_head import InstanceQueryHead  # noqa: E402
from tokengs.options import config_defaults  # noqa: E402
from scripts.train_instance_query_fixed4 import (  # noqa: E402
    build_window, hungarian_match, query_losses,
)


class A:
    cell = 16
    min_instance_pixels = 200


def hsh(t):
    return hashlib.sha1(np.ascontiguousarray(t.detach().float().cpu().numpy()).tobytes()).hexdigest()[:16]


def main() -> int:
    device = torch.device("cuda")
    if os.environ.get("MW_DETERMINISTIC") == "1":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        print("[probe] torch.use_deterministic_algorithms(True) is ON", flush=True)
    # identical head-init sequence to scripts/train_instance_query_fixed4.py
    torch.manual_seed(42); np.random.seed(42)
    opt = config_defaults["train_siu3r_locusgs_recon_bounded_delta_frozen_radius"].evolve(
        dataset_kwargs={"data_root": "/space/mawb/SIU3R/data/scannet"},
        batch_size=1, num_workers=0, seed=42, num_input_views=2, num_views=4)
    model = model_registry[opt.model_type](opt)
    st = torch.load(REPO / "workspace_recon_diag/cross_scene/lgs_lr1e4/ckpt_step6000/model.pt",
                    map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model", st), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    head = InstanceQueryHead(dim=int(opt.enc_embed_dim), num_queries=100).to(device)
    init = {k: v.detach().clone() for k, v in head.state_dict().items()}
    print("[probe] head init hash:", hashlib.sha1(
        b"".join(np.ascontiguousarray(v.float().cpu().numpy()).tobytes()
                 for v in init.values())).hexdigest()[:16], flush=True)

    root = Path("/space/mawb/SIU3R/data/scannet/train")
    args = A()
    w1 = build_window(opt, model, root, "scene0012_02", [2043, 2075], [2045, 2055], args, device)
    w2 = build_window(opt, model, root, "scene0012_02", [2043, 2075], [2045, 2055], args, device)
    for v in range(4):
        d = float((w1["maps"][v] - w2["maps"][v]).abs().max())
        a = float(np.abs(w1["alpha"][v] - w2["alpha"][v]).max())
        print(f"[probe] view {v}: maps hash {hsh(w1['maps'][v])} vs {hsh(w2['maps'][v])} | "
              f"max|dmap| {d:.3e} max|dalpha| {a:.3e}", flush=True)
    print(f"[probe] tokens hash {hsh(w1['tokens'])} vs {hsh(w2['tokens'])}", flush=True)

    for name, w in (("w1", w1), ("w2", w2)):
        logits, obj = head(w["tokens"].unsqueeze(0))
        A_ = torch.softmax(logits[0], dim=-1)
        ms = [torch.einsum("tq,tp->qp", A_[:, :100], w["maps"][vv]).reshape(100, 256, 256)
              for vv in range(4)]
        matched = hungarian_match(ms, w)
        loss, bce, dice, obj_loss = query_losses(ms, obj[0], w, matched)
        print(f"[probe] {name}: matched {sorted(matched.items())[:6]} loss {float(loss):.8f} "
              f"bce {float(bce):.8f} dice {float(dice):.8f} obj {float(obj_loss):.8f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
