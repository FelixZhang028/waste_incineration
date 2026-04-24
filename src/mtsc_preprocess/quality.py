from __future__ import annotations

import pandas as pd


def summarize_dataframe(df: pd.DataFrame, name: str) -> dict:
    if df.empty:
        return {
            "name": name,
            "rows": 0,
            "start_time": "",
            "end_time": "",
            "label_distribution": {},
            "transition_ratio": 0.0,
        }

    label_dist = (
        df["label"].value_counts(normalize=True).round(6).to_dict()
        if "label" in df.columns
        else {}
    )
    transition_ratio = (
        float(df["is_transition"].mean()) if "is_transition" in df.columns else 0.0
    )
    return {
        "name": name,
        "rows": int(len(df)),
        "start_time": str(df["timestamp"].min()),
        "end_time": str(df["timestamp"].max()),
        "label_distribution": label_dist,
        "transition_ratio": transition_ratio,
        "sources": df["source"].value_counts().to_dict() if "source" in df.columns else {},
        "furnaces": (
            df["furnace_id"].value_counts().sort_index().to_dict()
            if "furnace_id" in df.columns
            else {}
        ),
    }
