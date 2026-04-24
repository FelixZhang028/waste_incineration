from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class WindowBuildResult:
    index_df: pd.DataFrame
    dense_data: dict[str, np.ndarray] | None
    diagnostics: dict


def _median_step_seconds(df: pd.DataFrame) -> float:
    ts = df["timestamp"].sort_values()
    d = ts.diff().dt.total_seconds().dropna()
    if d.empty:
        return 60.0
    return float(d.median())


def build_window_samples(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_minutes: int,
    export_mode: str = "index",
) -> WindowBuildResult:
    if df.empty:
        return WindowBuildResult(
            index_df=df.iloc[0:0].copy(),
            dense_data=None,
            diagnostics={"rows": 0, "windows": 0, "window_minutes": window_minutes},
        )

    meta_cols = ["source", "furnace_id", "timestamp", "label_id", "sample_weight"]
    needed_cols = meta_cols + feature_cols if export_mode == "dense" else meta_cols
    out = df[needed_cols].sort_values(["source", "furnace_id", "timestamp"]).reset_index(drop=True)
    out["row_id"] = np.arange(len(out), dtype=np.int64)

    index_rows = []
    dense_x = []
    dense_y = []
    dense_w = []
    dropped_small_groups = 0

    for (source, furnace_id), g in out.groupby(["source", "furnace_id"], sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        step_sec = _median_step_seconds(g)
        win_steps = max(2, int(round(window_minutes * 60.0 / step_sec)))
        if len(g) < win_steps:
            dropped_small_groups += 1
            continue

        for end_idx in range(win_steps - 1, len(g)):
            start_idx = end_idx - win_steps + 1
            row_start = int(g.loc[start_idx, "row_id"])
            row_end = int(g.loc[end_idx, "row_id"])
            label_id = int(g.loc[end_idx, "label_id"])
            sample_weight = float(g.loc[end_idx, "sample_weight"])
            index_rows.append(
                {
                    "source": source,
                    "furnace_id": int(furnace_id),
                    "timestamp": g.loc[end_idx, "timestamp"],
                    "start_row_id": row_start,
                    "end_row_id": row_end,
                    "window_steps": win_steps,
                    "label_id": label_id,
                    "sample_weight": sample_weight,
                }
            )

            if export_mode == "dense":
                arr = g.loc[start_idx : end_idx, feature_cols].to_numpy(dtype=np.float32)
                dense_x.append(arr)
                dense_y.append(label_id)
                dense_w.append(sample_weight)

    index_df = pd.DataFrame(index_rows)
    dense_data: dict[str, np.ndarray] | None = None
    if export_mode == "dense":
        if dense_x:
            dense_data = {
                "X": np.stack(dense_x, axis=0),
                "y": np.asarray(dense_y, dtype=np.int64),
                "sample_weight": np.asarray(dense_w, dtype=np.float32),
                "feature_names": np.asarray(feature_cols),
            }
        else:
            dense_data = {
                "X": np.empty((0, 0, len(feature_cols)), dtype=np.float32),
                "y": np.empty((0,), dtype=np.int64),
                "sample_weight": np.empty((0,), dtype=np.float32),
                "feature_names": np.asarray(feature_cols),
            }

    diagnostics = {
        "rows": int(len(out)),
        "windows": int(len(index_df)),
        "window_minutes": window_minutes,
        "export_mode": export_mode,
        "dropped_small_groups": dropped_small_groups,
    }
    return WindowBuildResult(index_df=index_df, dense_data=dense_data, diagnostics=diagnostics)
