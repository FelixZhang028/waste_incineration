from __future__ import annotations

import re
import pandas as pd


FURNACE_PATTERN = re.compile(r"^([123])#(.+)$")


def select_feature_columns(
    df: pd.DataFrame,
    point_table_features: list[str],
    exclude_features: list[str] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    exclude = set(exclude_features or [])
    available = [c for c in df.columns if c != "timestamp"]
    white = set(point_table_features)
    selected = [c for c in available if c in white and c not in exclude]
    dropped_by_whitelist = [c for c in available if c not in white]
    dropped_by_exclude = [c for c in available if c in exclude]
    if not selected:
        raise ValueError("No feature selected after point-table whitelist filtering.")

    info = {
        "selected_features": selected,
        "dropped_by_whitelist": dropped_by_whitelist,
        "dropped_by_exclude": dropped_by_exclude,
    }
    return selected, info


def _build_furnace_map(feature_columns: list[str]) -> dict[int, dict[str, str]]:
    mapping: dict[int, dict[str, str]] = {1: {}, 2: {}, 3: {}}
    for col in feature_columns:
        m = FURNACE_PATTERN.match(col)
        if not m:
            continue
        furnace_id = int(m.group(1))
        base_name = m.group(2).strip()
        mapping[furnace_id][base_name] = col
    return mapping


def build_long_table(
    wide_df: pd.DataFrame,
    feature_columns: list[str],
    source_name: str,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    mapping = _build_furnace_map(feature_columns)
    base_sets = [set(v.keys()) for v in mapping.values() if v]
    if len(base_sets) != 3:
        raise ValueError("Feature set must contain usable columns for all 3 furnaces.")
    common_base = sorted(set.intersection(*base_sets))
    if not common_base:
        raise ValueError("No common base feature across the 3 furnaces.")

    long_parts: list[pd.DataFrame] = []
    for furnace_id in [1, 2, 3]:
        selected_cols = [mapping[furnace_id][base] for base in common_base]
        part = wide_df[["timestamp", *selected_cols]].copy()
        part = part.rename(columns={mapping[furnace_id][base]: base for base in common_base})
        part["furnace_id"] = furnace_id
        part["source"] = source_name
        long_parts.append(part)

    long_df = pd.concat(long_parts, axis=0, ignore_index=True)
    diagnostics = {
        "common_base_features": common_base,
        "raw_feature_columns": feature_columns,
    }
    return long_df, common_base, diagnostics
