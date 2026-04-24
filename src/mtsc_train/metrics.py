from __future__ import annotations

import torch


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if preds.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    flat = targets * num_classes + preds
    counts = torch.bincount(flat, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).to(dtype=torch.int64)


def summarize_confusion(
    conf: torch.Tensor,
    class_names: list[str],
) -> dict:
    conf = conf.to(dtype=torch.float64)
    tp = torch.diag(conf)
    support = conf.sum(dim=1)
    predicted = conf.sum(dim=0)
    total = conf.sum().clamp_min(1.0)

    precision = tp / predicted.clamp_min(1e-9)
    recall = tp / support.clamp_min(1e-9)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-9)

    accuracy = float(tp.sum() / total)
    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx].item()),
        }

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "samples": int(total.item()),
        "per_class": per_class,
    }
