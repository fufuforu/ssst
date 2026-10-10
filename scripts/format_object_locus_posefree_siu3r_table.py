"""Format existing official scores as the 13 metric columns in SIU3R Table 1."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

PAPER = "https://arxiv.org/html/2507.02705v1#S3.T1"
FIELDS = (
    "AbsRel", "RMSE", "PSNR", "SSIM", "LPIPS",
    "context_mIoU_s", "context_mAP", "context_PQ", "context_mIoU_t",
    "novel_mIoU_s", "novel_mAP", "novel_PQ", "novel_mIoU_t",
)
HEADERS = (
    "AbsRel↓", "RMSE↓", "PSNR↑", "SSIM↑", "LPIPS↓",
    "输入 mIoUₛ↑", "输入 mAP↑", "输入 PQ↑", "输入 mIoUₜ↑",
    "新视图 mIoUₛ↑", "新视图 mAP↑", "新视图 PQ↑", "新视图 mIoUₜ↑",
)
PRECISION = (5, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4)
PROTOCOL_NOTE = (
    "按 SIU3R Table 1 的顺序输出 13 个指标列；前 5 列为重建，"
    "随后各 4 列为输入视图与新视图场景理解。两处 mIoUₜ 留空：文本分支未训练。\n\n"
    "取值严格对齐固定版本官方 Evaluator 的返回字段：重建用其原生图像/深度汇总，"
    "两组场景理解分别用 context_* 与 target_*。官方 val_pair.json 每条记录的 "
    "target_ids 含 2 张输入帧和 4 张额外帧；官方代码未剔除输入帧。因此表中“新视图”"
    "列沿用官方 target 集合，未替换为仅额外 4 帧的另一种汇总。"
)


def table_values(report):
    # The original official run's target set, not the separately filtered export.
    assert all(report["scopes"][scope]["mIoU_t"] is None
               for scope in ("context", "target-all"))
    recon = report["scopes"]["target-all"]
    seg = report["official_segmentation"]["all"]
    values = [recon[k] for k in ("absrel", "rmse", "psnr", "ssim", "lpips")]
    for view in ("context", "target"):
        values.extend((seg[f"{view}_miou"], seg[f"{view}_map"]["map"],
                       seg[f"{view}_pq"], None))
    assert len(values) == 13
    assert all(v is None or math.isfinite(v) for v in values)
    assert values[8] is None and values[12] is None
    return values


def markdown_table(report):
    values = table_values(report)
    cells = ["" if v is None else f"{v:.{p}f}" for v, p in zip(values, PRECISION)]
    return "\n".join((
        "| " + " | ".join(HEADERS) + " |",
        "| " + " | ".join(["---:"] * 13) + " |",
        "| " + " | ".join(cells) + " |",
    ))


def write_table(report, root):
    root = Path(root)
    epoch = report["training_checkpoint_metadata"]["epoch"]
    staged = bool(report['training_checkpoint_metadata'].get('recipe'))
    values = table_values(report)
    with (root / "siu3r_table1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerow(values)  # csv writes None as an empty cell.
    payload = {
        "method": 'object_locus_vggt_recon_adapt_freeze_v1 / adaptation2 + frozen joint4' if staged else f"object_locus_frozen_vggt_posefree_v1 / epoch {epoch:02d}",
        "status": report["status"], "windows": report["windows"],
        "unique_scenes": report["unique_scenes"], "paper_table": PAPER,
        "columns": list(FIELDS), "values": values,
        "source": "metrics.json",
        "official_commit": report["official_commit"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "source_mapping": {
            "reconstruction": "scopes.target-all (native official aggregate)",
            "context_understanding": "official_segmentation.all.context_*",
            "novel_understanding": "official_segmentation.all.target_*",
        },
        "protocol_note": PROTOCOL_NOTE,
    }
    (root / "siu3r_table1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (root / "siu3r_table1.md").write_text(
        ("# Adaptation 2 + frozen joint 4 — SIU3R Table 1\n\n" if staged else f"# Epoch {epoch:02d} — SIU3R Table 1\n\n")
        + f"{report['status']}; {report['windows']} windows / {report['unique_scenes']} scenes.\n\n"
        + markdown_table(report) + "\n\n" + PROTOCOL_NOTE
        + f"\n\n[SIU3R Table 1]({PAPER}) · [完整原始指标](metrics.json) · [CSV](siu3r_table1.csv)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    write_table(json.loads((args.root / "metrics.json").read_text()), args.root)
