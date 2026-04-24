from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np


def save_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_table(path_without_suffix: str | Path, df: pd.DataFrame) -> str:
    base = Path(path_without_suffix)
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        target = base.with_suffix(".parquet")
        df.to_parquet(target, index=False)
        return str(target)
    except Exception:
        target = base.with_suffix(".csv")
        df.to_csv(target, index=False, encoding="utf-8-sig")
        return str(target)


def save_npz(path_without_suffix: str | Path, arrays: dict[str, np.ndarray]) -> str:
    base = Path(path_without_suffix)
    base.parent.mkdir(parents=True, exist_ok=True)
    target = base.with_suffix(".npz")
    np.savez_compressed(target, **arrays)
    return str(target)
