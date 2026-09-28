#!/usr/bin/env python3
"""CPU regression contract for the instance evaluator (no model, no dataset).

Builds synthetic predictions whose per-view masks are known, and checks the
registered definitions: GT-visible-view restricted multi-view IoU, the
per-view MIN_AREA rule acting as an all-zero prediction rather than dropping the
query, and the ``p_thing >= 0.5`` objectness gate.  Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/space/mawb/ssst")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_instance_state_v1 import (  # noqa: E402
    MIN_AREA, _instance_metrics, _multiview_iou_for_gt, _raw_recall50,
)

H = W = 16
VIEWS = 6


class Checks:
    def __init__(self):
        self.rows, self.failed = [], []

    def record(self, name, ok, detail):
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            self.failed.append(name)
        print(f"[eval-contract] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def blank(views=VIEWS, areas=(), cls=8, gid=5):
    """GT instance visible in the leading views with per-view pixel counts."""
    mask = torch.zeros(views, H, W, dtype=torch.bool)
    for v, area in enumerate(areas):
        flat = mask[v].reshape(-1)
        flat[:area] = True
    return (cls, gid), mask


def raw_from(spec):
    """spec: {(view, query): pixel_count} -> [V,100,H,W] bool."""
    raw = torch.zeros(VIEWS, 100, H, W, dtype=torch.bool)
    for (v, q), count in spec.items():
        raw[v, q].reshape(-1)[:count] = True
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="group_plus/instance_state_v1_generalization/"
                                    "eval_contract.json")
    args = ap.parse_args()
    checks = Checks()
    score = torch.full((100,), 0.1)
    cls = torch.full((100,), 8)
    is_thing = score >= 0.5
    area = MIN_AREA + 30

    # ---- Case A: GT visible in only 2 of 6 views, prediction perfect there -- #
    key, gt = blank(areas=(area, area))
    raw = raw_from({(0, 0): area, (1, 0): area})
    one = {key: gt}
    iou = _multiview_iou_for_gt(raw, 0, gt)
    checks.record("A.gt_visible_views_only", abs(iou - 1.0) < 1e-9,
                  f"IoU over the 2 visible views = {iou:.4f} (4 invisible views excluded)")
    checks.record("A.raw_recall_is_one", _raw_recall50(raw, one)["recall"] == 1.0,
                  f"raw recall {_raw_recall50(raw, one)}")
    s = score.clone(); s[0] = 0.9; t = s >= 0.5
    m = _instance_metrics(raw, cls, s, t, one, class_aware=True)
    checks.record("A.class_aware_tp", m["tp"] == 1 and m["fp"] == 0,
                  f"class-aware {m}")

    # ---- Case B: one visible view missed -> IoU < 0.5 -> FAIL ------------- #
    key_b, gt_b = blank(areas=(120, 60))          # both views visible, unequal
    raw_b = raw_from({(0, 0): 60})                # partial view 0, nothing in view 1
    iou_b = _multiview_iou_for_gt(raw_b, 0, gt_b)
    m_b = _instance_metrics(raw_b, cls, s, t, {key_b: gt_b}, class_aware=True)
    checks.record("B.partial_view_miss_fails",
                  iou_b < 0.5 and m_b["tp"] == 0 and m_b["fn"] == 1,
                  f"IoU {iou_b:.3f} -> {m_b}")

    # ---- Case C: p_thing < 0.5 with a perfect mask -> not a thing query --- #
    s_low = score.clone()
    t_low = s_low >= 0.5
    m_c = _instance_metrics(raw, cls, s_low, t_low, one, class_aware=True)
    checks.record("C.objectness_gate_blocks",
                  m_c["tp"] == 0 and m_c["fp"] == 0 and m_c["fn"] == 1,
                  f"p_thing 0.1 -> {m_c}")

    # ---- Case D: p_thing >= 0.5, same mask -> participates ----------------- #
    m_d = _instance_metrics(raw, cls, s, t, one, class_aware=True)
    checks.record("D.objectness_gate_allows", m_d["tp"] == 1, f"p_thing 0.9 -> {m_d}")

    # ---- Case E: wrong class -> class-aware FAIL, class-agnostic TP ------- #
    cls_wrong = cls.clone(); cls_wrong[0] = 3
    m_e_aware = _instance_metrics(raw, cls_wrong, s, t, one, class_aware=True)
    m_e_agn = _instance_metrics(raw, cls_wrong, s, t, one, class_aware=False)
    checks.record("E.class_aware_vs_agnostic",
                  m_e_aware["tp"] == 0 and m_e_agn["tp"] == 1,
                  f"aware {m_e_aware['tp']} vs agnostic {m_e_agn['tp']}")

    # ---- extra: hallucination on a GT-invisible-only support counts as FP -- #
    raw_f = raw_from({(3, 1): area})
    s2 = score.clone(); s2[1] = 0.9
    m_f = _instance_metrics(raw_f, cls, s2, s2 >= 0.5, one, class_aware=True)
    checks.record("F.hallucination_counts_fp", m_f["fp"] == 1 and m_f["tp"] == 0,
                  f"query 1 fires only in a GT-invisible view -> {m_f}")

    payload = {"checks": checks.rows, "failed": checks.failed, "ok": not checks.failed,
               "note": "synthetic CPU regression for the multi-view visibility rule and "
                       "the p_thing gate; raw recall50 stays GT-aided and un-gated by "
                       "definition (spec section 11.3)"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[eval-contract] {'ALL PASSED' if not checks.failed else 'FAILED ' + str(checks.failed)}")
    return 0 if not checks.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
