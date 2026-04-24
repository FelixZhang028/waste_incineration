from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import PostRuleConfig


def load_scaler_stats(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _median_step_seconds(ts: pd.Series) -> float:
    d = ts.sort_values().diff().dt.total_seconds().dropna()
    if d.empty:
        return 60.0
    return float(d.median())


def _inverse_scale_column(
    out: pd.DataFrame,
    source_col: str,
    target_col: str,
    scaler_stats: dict,
) -> pd.Series:
    values = out[source_col].astype(float).copy()
    for furnace_id, idx in out.groupby("furnace_id", sort=False).groups.items():
        furnace_stats = scaler_stats.get(str(int(furnace_id)), {})
        col_stats = furnace_stats.get(source_col)
        if not isinstance(col_stats, dict):
            continue
        mean = float(col_stats.get("mean", 0.0))
        std = float(col_stats.get("std", 1.0))
        values.loc[idx] = values.loc[idx] * std + mean
    values.name = target_col
    return values


def _rolling_linear_slope(values: np.ndarray, step_minutes: float, win_steps: int) -> np.ndarray:
    n = int(len(values))
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    step = max(float(step_minutes), 1e-6)
    for i in range(n):
        start = max(0, i - win_steps + 1)
        y = values[start : i + 1]
        k = int(len(y))
        if k < 2:
            out[i] = 0.0
            continue

        x = np.arange(k, dtype=np.float64) * step
        x_center = x - x.mean()
        denom = float(np.dot(x_center, x_center))
        if denom <= 0:
            out[i] = 0.0
            continue
        y_center = y - y.mean()
        out[i] = float(np.dot(x_center, y_center) / denom)

    return out


def build_rule_features(
    pred_df: pd.DataFrame,
    cfg: PostRuleConfig,
    scaler_stats: dict,
) -> pd.DataFrame:
    out = pred_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    fm = cfg.feature_mapping
    required_cols = {
        fm.temperature_upper_col,
        fm.temperature_middle_col,
        fm.oxygen_col,
        fm.pusher1_col,
        fm.pusher2_col,
        *fm.run_zero_cols,
        *fm.shutdown_zero_cols,
    }
    missing = sorted([c for c in required_cols if c not in out.columns])
    if missing:
        raise KeyError(f"Rule feature columns missing in prediction dataframe: {missing}")

    for col in sorted(required_cols):
        phys_col = f"{col}__phys"
        if cfg.inverse_scale:
            out[phys_col] = _inverse_scale_column(out, col, phys_col, scaler_stats)
        else:
            out[phys_col] = out[col].astype(float)

    out["temp_upper"] = out[f"{fm.temperature_upper_col}__phys"].astype(float)
    out["temp_middle"] = out[f"{fm.temperature_middle_col}__phys"].astype(float)
    out["temp_avg"] = (out["temp_upper"] + out["temp_middle"]) / 2.0
    out["o2"] = out[f"{fm.oxygen_col}__phys"].astype(float)
    out["pusher1"] = out[f"{fm.pusher1_col}__phys"].astype(float)
    out["pusher2"] = out[f"{fm.pusher2_col}__phys"].astype(float)

    run_phys_cols = [f"{c}__phys" for c in fm.run_zero_cols]
    shutdown_phys_cols = [f"{c}__phys" for c in fm.shutdown_zero_cols]

    run_zero_ok = np.ones(len(out), dtype=bool)
    for col in run_phys_cols:
        run_zero_ok &= out[col].to_numpy(dtype=np.float64) == 0.0
    out["run_zero_ok"] = run_zero_ok

    shutdown_zero_ok = np.ones(len(out), dtype=bool)
    for col in shutdown_phys_cols:
        shutdown_zero_ok &= out[col].to_numpy(dtype=np.float64) == 0.0
    out["shutdown_zero_ok"] = shutdown_zero_ok

    out["slope"] = 0.0
    win_minutes = int(cfg.timing.slope_window_minutes)

    for (_, _), idx in out.groupby(["source", "furnace_id"], sort=False).groups.items():
        g = out.loc[idx].sort_values("timestamp")
        step_sec = _median_step_seconds(g["timestamp"])
        step_minutes = step_sec / 60.0
        win_steps = max(2, int(round(win_minutes * 60.0 / max(step_sec, 1e-6))))
        slopes = _rolling_linear_slope(
            g["temp_avg"].to_numpy(dtype=np.float64),
            step_minutes=step_minutes,
            win_steps=win_steps,
        )
        out.loc[g.index, "slope"] = slopes

    return out.sort_values(["source", "furnace_id", "timestamp"]).reset_index(drop=True)
