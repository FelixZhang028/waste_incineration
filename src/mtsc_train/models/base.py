from __future__ import annotations

import torch
from torch import nn


class SequenceClassifier(nn.Module):
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
