from __future__ import annotations

import numpy as np
import pandas as pd


def clean_wide_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    outlier_quantiles: tuple[float, float] = (0.001, 0.999),
    impute_method: str = "ffill_bfill_median",
) -> tuple[pd.DataFrame, dict]:
    if impute_method != "ffill_bfill_median":
        raise ValueError(f"Unsupported impute method: {impute_method}")

    out = df.copy()
    q_low, q_high = outlier_quantiles

    missing_before = int(out[feature_cols].isna().sum().sum())
    clipped_total = 0
    clip_bounds: dict[str, dict[str, float]] = {}

    for col in feature_cols:
        series = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = series.dropna()
        if valid.empty:
            out[col] = series.fillna(0.0)
            clip_bounds[col] = {"low": 0.0, "high": 0.0}
            continue

        low = float(valid.quantile(q_low))
        high = float(valid.quantile(q_high))
        if high < low:
            low, high = high, low
        clipped = ((series < low) | (series > high)).fillna(False)
        clipped_total += int(clipped.sum())
        series = series.clip(lower=low, upper=high)
        out[col] = series
        clip_bounds[col] = {"low": low, "high": high}

    out = out.sort_values("timestamp").reset_index(drop=True)
    out[feature_cols] = out[feature_cols].ffill().bfill()
    medians = out[feature_cols].median(axis=0)
    out[feature_cols] = out[feature_cols].fillna(medians).fillna(0.0)

    missing_after = int(out[feature_cols].isna().sum().sum())
    diagnostics = {
        "missing_before": missing_before,
        "missing_after": missing_after,
        "outlier_clipped_total": clipped_total,
        "outlier_quantiles": [q_low, q_high],
        "impute_method": impute_method,
        "clip_bounds": clip_bounds,
    }
    return out, diagnostics
