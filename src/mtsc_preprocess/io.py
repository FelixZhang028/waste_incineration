from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import DataSourceConfig


def _find_column(columns: list[str], keyword: str) -> str:
    for col in columns:
        if keyword in col:
            return col
    raise KeyError(f"Column containing keyword '{keyword}' not found in: {columns}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def load_point_table(path: str | Path) -> list[str]:
    xls = pd.ExcelFile(path)
    df = _normalize_columns(pd.read_excel(path, sheet_name=xls.sheet_names[0]))
    name_col = _find_column(list(df.columns), "名称")
    features = (
        df[name_col]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    return sorted(features)


def load_raw_data(source: DataSourceConfig) -> pd.DataFrame:
    data_path = Path(source.data_path)
    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path, encoding=source.encoding or "utf-8")
    elif data_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    df = _normalize_columns(df)
    time_col = _find_column(list(df.columns), "时间")
    df = df.rename(columns={time_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")

    non_time_cols = [c for c in df.columns if c != "timestamp"]
    for col in non_time_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_label_data(source: DataSourceConfig) -> pd.DataFrame:
    df = _normalize_columns(pd.read_excel(source.label_path))
    cols = list(df.columns)
    monitor_col = _find_column(cols, "监控点")
    status_col = _find_column(cols, "状态")
    start_col = _find_column(cols, "开始")
    end_col = _find_column(cols, "结束")

    out = df.rename(
        columns={
            monitor_col: "monitor_point",
            status_col: "status",
            start_col: "start_time",
            end_col: "end_time",
        }
    )
    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
    out["status"] = out["status"].astype(str).str.strip()
    out["monitor_point"] = out["monitor_point"].astype(str).str.strip()
    out = out.dropna(subset=["start_time", "end_time"])
    out = out.loc[out["end_time"] >= out["start_time"]].reset_index(drop=True)
    return out[["monitor_point", "status", "start_time", "end_time"]]
