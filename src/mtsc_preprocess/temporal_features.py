from __future__ import annotations

import pandas as pd


def add_temporal_features(
    df: pd.DataFrame,
    base_feature_cols: list[str],
    diff_lags: list[int],
    rolling_windows_minutes: list[int],
) -> tuple[pd.DataFrame, list[str], dict]:
    out = df.copy().sort_values(["source", "furnace_id", "timestamp"]).reset_index(drop=True)
    generated: list[str] = []
    new_cols: dict[str, pd.Series] = {}

    group_keys = ["source", "furnace_id"]
    grouped = out.groupby(group_keys, sort=False)

    for lag in diff_lags:
        for col in base_feature_cols:
            new_col = f"{col}__diff_{lag}"
            new_cols[new_col] = grouped[col].diff(lag)
            generated.append(new_col)

    # Time-based rolling statistics by source+furnace.
    for win_min in rolling_windows_minutes:
        for col in base_feature_cols:
            mean_col = f"{col}__roll_mean_{win_min}m"
            std_col = f"{col}__roll_std_{win_min}m"
            parts_mean: list[pd.Series] = []
            parts_std: list[pd.Series] = []

            for (_, _), idx in grouped.indices.items():
                g = out.loc[idx, ["timestamp", col]].sort_values("timestamp")
                s = g.set_index("timestamp")[col]
                r = s.rolling(f"{win_min}min", min_periods=1)
                parts_mean.append(r.mean().reset_index(drop=True).set_axis(g.index))
                parts_std.append(r.std().reset_index(drop=True).set_axis(g.index))

            mean_series = pd.concat(parts_mean).sort_index()
            std_series = pd.concat(parts_std).sort_index()
            new_cols[mean_col] = mean_series
            new_cols[std_col] = std_series
            generated.extend([mean_col, std_col])

    if generated:
        feature_df = pd.DataFrame(new_cols, index=out.index)
        feature_df = feature_df.fillna(0.0)
        out = pd.concat([out, feature_df], axis=1)
    diagnostics = {
        "base_feature_count": len(base_feature_cols),
        "generated_feature_count": len(generated),
        "diff_lags": diff_lags,
        "rolling_windows_minutes": rolling_windows_minutes,
    }
    return out, generated, diagnostics
