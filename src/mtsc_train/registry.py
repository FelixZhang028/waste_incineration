from __future__ import annotations

from typing import Any

from .models import GRUClassifier, LSTMClassifier, SequenceClassifier


MODEL_REGISTRY: dict[str, type[SequenceClassifier]] = {
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
}


def create_model(
    name: str,
    input_size: int,
    num_classes: int,
    params: dict[str, Any] | None = None,
) -> SequenceClassifier:
    model_name = name.strip().lower()
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Supported models: {supported}")
    return model_cls(input_size=input_size, num_classes=num_classes, **(params or {}))
