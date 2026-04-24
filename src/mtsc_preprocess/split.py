from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    date_boundaries: dict[str, str]


def split_by_date_ratio(
    df: pd.DataFrame,
    ratios: tuple[float, float, float],
) -> SplitResult:
    out = df.copy()
    out["date"] = out["timestamp"].dt.date
    unique_dates = sorted(out["date"].dropna().unique())
    if len(unique_dates) < 3:
        raise ValueError("Not enough unique dates to split by 7:1:2. Need at least 3 dates.")

    n = len(unique_dates)
    train_n = max(1, int(np.floor(n * ratios[0])))
    val_n = max(1, int(np.floor(n * ratios[1])))
    test_n = n - train_n - val_n
    if test_n <= 0:
        test_n = 1
        if train_n >= val_n:
            train_n -= 1
        else:
            val_n -= 1

    train_dates = set(unique_dates[:train_n])
    val_dates = set(unique_dates[train_n : train_n + val_n])
    test_dates = set(unique_dates[train_n + val_n :])

    train = out.loc[out["date"].isin(train_dates)].drop(columns=["date"]).reset_index(drop=True)
    val = out.loc[out["date"].isin(val_dates)].drop(columns=["date"]).reset_index(drop=True)
    test = out.loc[out["date"].isin(test_dates)].drop(columns=["date"]).reset_index(drop=True)

    boundaries = {
        "train_start": str(min(train_dates)),
        "train_end": str(max(train_dates)),
        "val_start": str(min(val_dates)),
        "val_end": str(max(val_dates)),
        "test_start": str(min(test_dates)),
        "test_end": str(max(test_dates)),
    }
    return SplitResult(train=train, val=val, test=test, date_boundaries=boundaries)


def split_full(df: pd.DataFrame) -> SplitResult:
    full = df.copy().reset_index(drop=True)
    empty = full.iloc[0:0].copy()
    start_date = str(full["timestamp"].dt.date.min())
    end_date = str(full["timestamp"].dt.date.max())
    boundaries = {
        "train_start": start_date,
        "train_end": end_date,
        "val_start": "",
        "val_end": "",
        "test_start": "",
        "test_end": "",
    }
    return SplitResult(train=full, val=empty, test=empty, date_boundaries=boundaries)


def split_by_source_lists(
    df: pd.DataFrame,
    train_sources: list[str],
    val_sources: list[str],
    test_sources: list[str],
    overlap_split_ratios: tuple[float, float] | None = None,
) -> SplitResult:
    out = df.copy()
    src_set = set(out["source"].dropna().astype(str).unique())
    expected = set(train_sources) | set(val_sources) | set(test_sources)
    missing = sorted(expected - src_set)
    if missing:
        raise ValueError(f"Configured sources not found in data: {missing}")

    train = out.loc[out["source"].isin(train_sources)].reset_index(drop=True)
    overlap_sources = sorted(set(val_sources) & set(test_sources))

    if overlap_sources:
        if overlap_split_ratios is None:
            raise ValueError(
                "overlap_split_ratios must be provided when val_sources and test_sources overlap."
            )
        val_ratio, test_ratio = overlap_split_ratios
        if val_ratio <= 0 or test_ratio <= 0:
            raise ValueError("overlap_split_ratios must contain positive values.")

        val_only_sources = set(val_sources) - set(overlap_sources)
        test_only_sources = set(test_sources) - set(overlap_sources)

        val_parts = [out.loc[out["source"].isin(val_only_sources)].copy()]
        test_parts = [out.loc[out["source"].isin(test_only_sources)].copy()]

        total = val_ratio + test_ratio
        val_share = val_ratio / total

        for src in overlap_sources:
            src_df = out.loc[out["source"] == src].copy()
            src_df["date"] = src_df["timestamp"].dt.date
            unique_dates = sorted(src_df["date"].dropna().unique())
            if len(unique_dates) < 2:
                raise ValueError(
                    f"Source '{src}' has fewer than 2 unique dates; cannot split into val/test."
                )

            val_date_n = int(np.floor(len(unique_dates) * val_share))
            val_date_n = max(1, min(val_date_n, len(unique_dates) - 1))
            val_dates = set(unique_dates[:val_date_n])
            test_dates = set(unique_dates[val_date_n:])

            val_parts.append(src_df.loc[src_df["date"].isin(val_dates)].drop(columns=["date"]))
            test_parts.append(src_df.loc[src_df["date"].isin(test_dates)].drop(columns=["date"]))

        val = pd.concat(val_parts, axis=0, ignore_index=True)
        test = pd.concat(test_parts, axis=0, ignore_index=True)
    else:
        val = out.loc[out["source"].isin(val_sources)].reset_index(drop=True)
        test = out.loc[out["source"].isin(test_sources)].reset_index(drop=True)

    if train.empty:
        raise ValueError("Train split is empty under source_holdout strategy.")
    if val.empty and test.empty:
        raise ValueError("Both val and test are empty under source_holdout strategy.")

    boundaries = {
        "train_start": str(train["timestamp"].dt.date.min()) if not train.empty else "",
        "train_end": str(train["timestamp"].dt.date.max()) if not train.empty else "",
        "val_start": str(val["timestamp"].dt.date.min()) if not val.empty else "",
        "val_end": str(val["timestamp"].dt.date.max()) if not val.empty else "",
        "test_start": str(test["timestamp"].dt.date.min()) if not test.empty else "",
        "test_end": str(test["timestamp"].dt.date.max()) if not test.empty else "",
    }
    return SplitResult(train=train, val=val, test=test, date_boundaries=boundaries)


def zscore_by_furnace(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, dict[str, float]]]]:
    train = train_df.copy()
    val = val_df.copy()
    test = test_df.copy()
    for frame in (train, val, test):
        if not frame.empty:
            frame[feature_cols] = frame[feature_cols].astype(float)

    stats: dict[str, dict[str, dict[str, float]]] = {}

    for furnace_id in sorted(train["furnace_id"].dropna().unique()):
        furnace_train_mask = train["furnace_id"] == furnace_id
        furnace_train = train.loc[furnace_train_mask, feature_cols]

        means = furnace_train.mean(axis=0)
        stds = furnace_train.std(axis=0).replace(0, 1.0).fillna(1.0)
        stats[str(int(furnace_id))] = {
            col: {"mean": float(means[col]), "std": float(stds[col])} for col in feature_cols
        }

        for frame in (train, val, test):
            mask = frame["furnace_id"] == furnace_id
            if mask.any():
                frame.loc[mask, feature_cols] = (frame.loc[mask, feature_cols] - means) / stds

    return train, val, test, stats
