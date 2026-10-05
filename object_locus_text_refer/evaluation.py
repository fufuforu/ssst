from __future__ import annotations

import torch


def binarize_probability(probability):
    return torch.as_tensor(probability) > 0.5


def masked_iou(pred_mask, gt_mask, valid_mask):
    valid=torch.as_tensor(valid_mask).bool()
    if not bool(valid.any()): return 0.0, "empty_valid_domain"
    pred=torch.as_tensor(pred_mask).bool() & valid
    gt=torch.as_tensor(gt_mask).bool() & valid
    union=int((pred|gt).sum())
    return (float((pred & gt).sum())/union if union else 1.0), None


def aggregate_expressions(expressions):
    """Aggregate one record per expression, averaging its two context-view IoUs."""
    rows=[]; values=[]; nulls=0
    for expression in expressions:
        failed=bool(expression.get("failure_reason"))
        view_ious=expression.get("view_ious") or []
        score=0.0 if failed or not view_ious else sum(float(v) for v in view_ious)/len(view_ious)
        row={**expression,"expression_iou":score}
        rows.append(row); values.append(score)
        nulls += int(expression.get("selected_slot")==100)
    count=len(rows)
    return {
        "context_refer_expression_count":count,
        "context_refer_mIoU":sum(values)/count if count else 0.0,
        "context_refer_Acc@0.25":sum(v>=.25 for v in values)/count if count else 0.0,
        "context_refer_Acc@0.50":sum(v>=.50 for v in values)/count if count else 0.0,
        "context_refer_null_rate":nulls/count if count else 0.0,
        "context_refer_failed_expressions":sum(bool(r.get("failure_reason")) for r in rows),
        "context_refer_records":rows,
        "protocol_note":"Context expression aggregation over the listed raw descriptions; not paper mIoU_t and not an aligned official novel benchmark.",
    }


def evaluate_records(rows):
    """Per-view CPU scalar summaries for smoke/contracts; failed rows are retained."""
    output=[]
    for source in rows:
        row={k:v for k,v in source.items() if k not in ("pred_mask","gt_mask","valid_mask")}
        if source.get("failure_reason"):
            iou=0.0
        elif source.get("pred_mask") is None or source.get("gt_mask") is None or source.get("valid_mask") is None:
            iou=0.0; row["failure_reason"]="missing_prediction_or_ground_truth"
        else:
            iou,reason=masked_iou(source["pred_mask"],source["gt_mask"],source["valid_mask"])
            if reason: row["failure_reason"]=reason
        row["iou"]=float(iou); output.append(row)
    return output
