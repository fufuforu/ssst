from collections import defaultdict

import torch


def evaluate_records(rows):
    """Keep every expression/view, including empty prediction and failed records."""
    by_text = defaultdict(list); all_ious=[]; nulls=0; total=0; serializable=[]
    for row in rows:
        pred = row.get("pred_mask")
        gt = row.get("gt_mask")
        valid = row.get("valid_mask")
        if pred is None or gt is None or valid is None:
            iou = 0.0
        else:
            p = torch.as_tensor(pred).bool() & torch.as_tensor(valid).bool()
            g = torch.as_tensor(gt).bool() & torch.as_tensor(valid).bool()
            union = int((p|g).sum()); iou = float((p&g).sum())/union if union else 1.0
        row["iou"] = iou; all_ious.append(iou); by_text[row.get("text_key")].append(iou)
        serializable.append({k:v for k,v in row.items() if k not in ("pred_mask","gt_mask","valid_mask")})
        total += 1; nulls += int(row.get("selected_slot") == 100)
    text_means = {key: sum(values)/len(values) for key, values in by_text.items()}
    return {"per_record": serializable, "per_text_context_mean_iou": text_means,
            "mean_iou_all_descriptions": sum(all_ious)/max(1,len(all_ious)),
            "acc_iou_025": sum(x>=.25 for x in all_ious)/max(1,len(all_ious)),
            "acc_iou_05": sum(x>=.5 for x in all_ious)/max(1,len(all_ious)),
            "null_selection_rate": nulls/max(1,total), "records": total,
            "aggregation_note": "All descriptions expanded in official val_refer_pair order; not official randomized-one-description aggregation and not paper mIoU_t."}
