from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def load_feature_list(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    features = payload.get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError(f"feature_list is empty or invalid: {path}")
    return [str(x) for x in features]


def load_label_map(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"label_map is empty or invalid: {path}")
    return {str(k): int(v) for k, v in payload.items()}


def _resolve_existing_table(path: str | Path) -> Path:
    raw = Path(path)
    candidates = [raw]
    if raw.suffix:
        alt = raw.with_suffix(".csv") if raw.suffix.lower() == ".parquet" else raw.with_suffix(".parquet")
        candidates.append(alt)
    else:
        candidates.extend([raw.with_suffix(".parquet"), raw.with_suffix(".csv")])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Table file not found for path: {raw}")


def maybe_existing_path(path: str | Path) -> Path | None:
    p = Path(path)
    return p if p.exists() else None


def _read_table(
    path: str | Path,
    usecols: list[str],
    nrows: int | None = None,
) -> pd.DataFrame:
    target = _resolve_existing_table(path)
    if target.suffix.lower() == ".parquet":
        return pd.read_parquet(target, columns=usecols)
    if target.suffix.lower() == ".csv":
        return pd.read_csv(target, usecols=usecols, nrows=nrows, low_memory=False)
    raise ValueError(f"Unsupported table format: {target.suffix}")


def _read_window_index(path: str | Path, max_windows: int | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Window index file not found: {path}")
    df = pd.read_csv(p, low_memory=False)
    if max_windows is not None:
        df = df.iloc[: max(0, int(max_windows))].copy()
    return df.reset_index(drop=True)


class WindowIndexDataset(Dataset):
    def __init__(
        self,
        table_path: str | Path,
        window_index_path: str | Path,
        feature_cols: list[str],
        max_windows: int | None = None,
    ) -> None:
        if not feature_cols:
            raise ValueError("feature_cols cannot be empty")

        index_df = _read_window_index(window_index_path, max_windows=max_windows)
        if index_df.empty:
            raise ValueError(f"No windows available in {window_index_path}")

        required = ["start_row_id", "end_row_id", "label_id", "sample_weight"]
        missing_required = [c for c in required if c not in index_df.columns]
        if missing_required:
            raise ValueError(
                f"Missing required columns in window index {window_index_path}: {missing_required}"
            )

        min_row = int(index_df["start_row_id"].min())
        max_row = int(index_df["end_row_id"].max())
        maybe_nrows = (max_row + 1) if min_row == 0 and max_windows is not None else None

        usecols = sorted(set(["source", "furnace_id", "timestamp", *feature_cols]))
        table_df = _read_table(table_path, usecols=usecols, nrows=maybe_nrows)
        table_df = table_df.sort_values(["source", "furnace_id", "timestamp"], kind="mergesort")
        table_df = table_df.reset_index(drop=True)

        if max_row >= len(table_df):
            raise ValueError(
                f"Window row id exceeds table size: max_row_id={max_row}, table_rows={len(table_df)}"
            )

        missing_features = [c for c in feature_cols if c not in table_df.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found in table: {missing_features}")

        self.features = table_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        self.start_row_ids = index_df["start_row_id"].to_numpy(dtype=np.int64, copy=True)
        self.end_row_ids = index_df["end_row_id"].to_numpy(dtype=np.int64, copy=True)
        self.label_ids = index_df["label_id"].to_numpy(dtype=np.int64, copy=True)
        if "furnace_id" in index_df.columns:
            self.furnace_ids = index_df["furnace_id"].to_numpy(dtype=np.int64, copy=True)
        else:
            # Backward compatibility for old index files without furnace_id.
            self.furnace_ids = table_df.iloc[self.end_row_ids]["furnace_id"].to_numpy(dtype=np.int64, copy=True)
        self.sample_weights = index_df["sample_weight"].to_numpy(dtype=np.float32, copy=True)
        self.window_steps = (
            index_df["window_steps"].to_numpy(dtype=np.int64, copy=True)
            if "window_steps" in index_df.columns
            else (self.end_row_ids - self.start_row_ids + 1)
        )
        self.feature_dim = len(feature_cols)

    def __len__(self) -> int:
        return int(len(self.start_row_ids))

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int | float]:
        start = int(self.start_row_ids[idx])
        end = int(self.end_row_ids[idx])
        window = self.features[start : end + 1]
        return {
            "x": window,
            "length": int(len(window)),
            "furnace_id": int(self.furnace_ids[idx]),
            "label_id": int(self.label_ids[idx]),
            "sample_weight": float(self.sample_weights[idx]),
        }

    def class_counts(self, num_classes: int) -> np.ndarray:
        counts = np.bincount(self.label_ids, minlength=num_classes).astype(np.int64)
        return counts


def collate_window_batch(
    batch: Iterable[dict[str, np.ndarray | int | float]],
) -> dict[str, torch.Tensor]:
    items = list(batch)
    if not items:
        raise ValueError("Empty batch received by collate_window_batch.")

    lengths = torch.tensor([int(item["length"]) for item in items], dtype=torch.long)
    max_len = int(lengths.max().item())
    feature_dim = int(items[0]["x"].shape[1])  # type: ignore[index]
    x = torch.zeros((len(items), max_len, feature_dim), dtype=torch.float32)

    labels = torch.tensor([int(item["label_id"]) for item in items], dtype=torch.long)
    furnace_ids = torch.tensor([int(item["furnace_id"]) for item in items], dtype=torch.long)
    weights = torch.tensor([float(item["sample_weight"]) for item in items], dtype=torch.float32)

    for i, item in enumerate(items):
        arr = np.asarray(item["x"], dtype=np.float32)  # type: ignore[arg-type]
        length = arr.shape[0]
        x[i, :length] = torch.from_numpy(arr)

    return {
        "x": x,
        "lengths": lengths,
        "furnace_id": furnace_ids,
        "label_id": labels,
        "sample_weight": weights,
    }


def build_loader(
    dataset: WindowIndexDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_window_batch,
        drop_last=False,
    )
