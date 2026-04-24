from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        target_log_prob = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_prob = prob.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = torch.pow(1.0 - target_prob, self.gamma)
        loss = -focal_factor * target_log_prob

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t
        return loss


def compute_balanced_class_weights(counts: np.ndarray) -> np.ndarray:
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array")
    counts = counts.astype(np.float64)
    total = float(counts.sum())
    num_classes = int(len(counts))
    if total <= 0 or num_classes <= 0:
        raise ValueError("counts must contain at least one sample")

    weights = np.zeros_like(counts, dtype=np.float64)
    non_zero = counts > 0
    weights[non_zero] = total / (num_classes * counts[non_zero])
    if not np.any(non_zero):
        weights[:] = 1.0
    return weights.astype(np.float32)


def build_criterion(
    name: str,
    gamma: float,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    key = name.strip().lower()
    if key == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    if key == "focal":
        return FocalLoss(gamma=gamma, alpha=class_weights)
    raise ValueError(f"Unsupported loss function: {name}")


def weighted_mean(loss: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    weight_sum = sample_weight.sum().clamp_min(1e-6)
    return (loss * sample_weight).sum() / weight_sum
