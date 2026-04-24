from __future__ import annotations

import pandas as pd


def apply_labels_with_transition(
    long_df: pd.DataFrame,
    label_df: pd.DataFrame,
    monitor_to_furnace: dict[str, int],
    normal_label: str,
    transition_buffer_minutes: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = long_df.copy()
    out["label"] = normal_label
    out["is_transition"] = False

    unresolved_points: set[str] = set()
    buffer_delta = pd.Timedelta(minutes=transition_buffer_minutes)

    for row in label_df.itertuples(index=False):
        monitor = str(row.monitor_point).strip()
        status = str(row.status).strip()
        start_time = pd.Timestamp(row.start_time)
        end_time = pd.Timestamp(row.end_time)
        furnace_id = monitor_to_furnace.get(monitor)
        if furnace_id is None:
            unresolved_points.add(monitor)
            continue

        furnace_mask = out["furnace_id"] == int(furnace_id)
        interval_mask = furnace_mask & (out["timestamp"] >= start_time) & (out["timestamp"] <= end_time)
        out.loc[interval_mask, "label"] = status

        start_win = (
            furnace_mask
            & (out["timestamp"] >= (start_time - buffer_delta))
            & (out["timestamp"] <= (start_time + buffer_delta))
        )
        end_win = (
            furnace_mask
            & (out["timestamp"] >= (end_time - buffer_delta))
            & (out["timestamp"] <= (end_time + buffer_delta))
        )
        out.loc[start_win | end_win, "is_transition"] = True

    diagnostics: dict[str, object] = {
        "unresolved_monitor_points": sorted(unresolved_points),
        "transition_buffer_minutes": transition_buffer_minutes,
    }
    return out, diagnostics
