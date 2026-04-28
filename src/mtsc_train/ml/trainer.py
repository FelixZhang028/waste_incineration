from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import random
from typing import Any

import numpy as np
import pandas as pd
import torch

from mtsc_train.config import TrainConfig
from mtsc_train.data import load_feature_list, load_label_map
from mtsc_train.metrics import confusion_matrix, summarize_confusion


WINDOW_STATS = ("mean", "std", "min", "max", "first", "last", "delta")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("mtsc-train-ml")
    p.add_argument("--config", default="configs/train.json", help="Training config path")
    p.add_argument(
        "--model",
        default="lightgbm",
        choices=["lightgbm", "extratrees"],
        help="Machine-learning model to train",
    )
    p.add_argument("--out-dir", default="artifacts/train_ml", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--n-estimators", type=int, default=500, help="Number of trees")
    p.add_argument("--learning-rate", type=float, default=0.05, help="LightGBM learning rate")
    p.add_argument("--num-leaves", type=int, default=31, help="LightGBM num_leaves")
    p.add_argument(
        "--class-weight",
        default="none",
        choices=["none", "balanced"],
        help="Class weighting for the ML model",
    )
    p.add_argument(
        "--no-sample-weight",
        action="store_true",
        help="Do not pass window sample_weight to the ML model",
    )
    p.add_argument(
        "--max-train-windows",
        type=int,
        default=None,
        help="Optional cap for training windows",
    )
    p.add_argument(
        "--max-eval-windows",
        type=int,
        default=None,
        help="Optional cap for val/test windows",
    )
    return p.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _resolve_existing_table(path: str | Path) -> Path:
    raw = Path(path)
    candidates = [raw]
    if raw.suffix:
        alt = raw.with_suffix(".csv") if raw.suffix.lower() == ".parquet" else raw.with_suffix(".parquet")
        candidates.append(alt)
    else:
        candidates.extend([raw.with_suffix(".parquet"), raw.with_suffix(".csv")])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Table file not found for path: {raw}")


def _read_table(path: str | Path, usecols: list[str]) -> pd.DataFrame:
    target = _resolve_existing_table(path)
    if target.suffix.lower() == ".parquet":
        return pd.read_parquet(target, columns=usecols)
    if target.suffix.lower() == ".csv":
        return pd.read_csv(target, usecols=usecols, low_memory=False)
    raise ValueError(f"Unsupported table format: {target.suffix}")


def _read_window_index(path: str | Path, max_windows: int | None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Window index file not found: {path}")
    df = pd.read_csv(p, low_memory=False)
    if max_windows is not None:
        df = df.iloc[: max(0, int(max_windows))].copy()
    return df.reset_index(drop=True)


def _label_names_by_id(label_map: dict[str, int]) -> list[str]:
    pairs = sorted(((idx, name) for name, idx in label_map.items()), key=lambda x: x[0])
    if not pairs:
        raise ValueError("label_map cannot be empty")
    ids = [idx for idx, _ in pairs]
    if ids != list(range(len(pairs))):
        raise ValueError(f"label ids must be contiguous from 0, got {ids}")
    return [name for _, name in pairs]


def _split_paths(cfg: TrainConfig, split: str) -> tuple[str, str, int | None]:
    if split == "train":
        return cfg.data.train_table, cfg.data.train_window_index, cfg.data.max_train_windows
    if split == "val":
        return cfg.data.val_table, cfg.data.val_window_index, cfg.data.max_eval_windows
    if split == "test":
        return cfg.data.test_table, cfg.data.test_window_index, cfg.data.max_eval_windows
    raise ValueError(f"Unsupported split: {split}")


def _build_stat_names(feature_cols: list[str]) -> tuple[list[str], list[str], list[str]]:
    stat_features: list[str] = []
    base_features: list[str] = []
    stats: list[str] = []
    for base in feature_cols:
        for stat in WINDOW_STATS:
            stat_features.append(f"{base}__{stat}")
            base_features.append(base)
            stats.append(stat)
    return stat_features, base_features, stats


def _window_to_stats(values: np.ndarray) -> np.ndarray:
    first = values[0]
    last = values[-1]
    parts = [
        values.mean(axis=0),
        values.std(axis=0),
        values.min(axis=0),
        values.max(axis=0),
        first,
        last,
        last - first,
    ]
    return np.stack(parts, axis=1).reshape(-1)


def _load_window_matrix(
    table_path: str | Path,
    index_path: str | Path,
    feature_cols: list[str],
    max_windows: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_df = _read_window_index(index_path, max_windows=max_windows)
    if index_df.empty:
        raise ValueError(f"No windows available in {index_path}")

    required = ["start_row_id", "end_row_id", "label_id", "sample_weight"]
    missing = [c for c in required if c not in index_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in window index {index_path}: {missing}")

    usecols = sorted(set(["source", "furnace_id", "timestamp", *feature_cols]))
    table_df = _read_table(table_path, usecols=usecols)
    table_df = table_df.sort_values(["source", "furnace_id", "timestamp"], kind="mergesort")
    table_df = table_df.reset_index(drop=True)

    max_row = int(index_df["end_row_id"].max())
    if max_row >= len(table_df):
        raise ValueError(
            f"Window row id exceeds table size: max_row_id={max_row}, table_rows={len(table_df)}"
        )

    raw_features = table_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    starts = index_df["start_row_id"].to_numpy(dtype=np.int64, copy=True)
    ends = index_df["end_row_id"].to_numpy(dtype=np.int64, copy=True)
    labels = index_df["label_id"].to_numpy(dtype=np.int64, copy=True)
    sample_weight = index_df["sample_weight"].to_numpy(dtype=np.float32, copy=True)

    x = np.empty((len(index_df), len(feature_cols) * len(WINDOW_STATS)), dtype=np.float32)
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        x[i] = _window_to_stats(raw_features[int(start) : int(end) + 1])

    return x, labels, sample_weight


def _create_model(
    name: str,
    n_estimators: int,
    seed: int,
    class_weight: str,
    learning_rate: float,
    num_leaves: int,
):
    class_weight_arg = None if class_weight == "none" else "balanced"
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is not installed. Install it with: "
                'python -m pip install "lightgbm[scikit-learn]"'
            ) from exc
        return LGBMClassifier(
            objective="multiclass",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight=class_weight_arg,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )

    try:
        from sklearn.ensemble import ExtraTreesClassifier
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is not installed. Install it with: python -m pip install scikit-learn"
        ) from exc
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight=class_weight_arg,
    )


def _compute_metrics(pred: np.ndarray, target: np.ndarray, class_names: list[str]) -> dict:
    conf = confusion_matrix(
        preds=torch.from_numpy(pred.astype(np.int64)),
        targets=torch.from_numpy(target.astype(np.int64)),
        num_classes=len(class_names),
    )
    return summarize_confusion(conf, class_names)


def _evaluate_split(
    model,
    x: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
) -> tuple[dict, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    pred = np.asarray(model.predict(x_arr), dtype=np.int64)
    return _compute_metrics(pred=pred, target=y_arr, class_names=class_names), pred


def _raw_importance(model, model_name: str, feature_count: int) -> tuple[np.ndarray, np.ndarray | None]:
    if model_name == "lightgbm":
        booster = model.booster_
        gain = booster.feature_importance(importance_type="gain").astype(np.float64)
        split = booster.feature_importance(importance_type="split").astype(np.float64)
        return gain, split

    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        raise ValueError(f"Model {model_name} does not expose feature_importances_.")
    raw = np.asarray(importance, dtype=np.float64)
    if raw.shape[0] != feature_count:
        raise ValueError(f"Importance size mismatch: got {raw.shape[0]}, expected {feature_count}")
    return raw, None


def _normalize_importance(raw: np.ndarray) -> np.ndarray:
    total = float(raw.sum())
    if total <= 0:
        return np.zeros_like(raw, dtype=np.float64)
    return raw.astype(np.float64) / total


def _build_importance_frames(
    raw_importance: np.ndarray,
    split_importance: np.ndarray | None,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stat_features, base_features, stats = _build_stat_names(feature_cols)
    importance = _normalize_importance(raw_importance)

    stat_df = pd.DataFrame(
        {
            "stat_feature": stat_features,
            "base_feature": base_features,
            "stat": stats,
            "importance": importance,
            "importance_pct": importance * 100.0,
            "raw_importance": raw_importance,
        }
    )
    if split_importance is not None:
        stat_df["split_importance"] = split_importance

    stat_df = stat_df.sort_values("importance", ascending=False).reset_index(drop=True)
    stat_df.insert(0, "rank", np.arange(1, len(stat_df) + 1))

    base_df = (
        stat_df.groupby("base_feature", as_index=False)
        .agg(
            importance=("importance", "sum"),
            raw_importance=("raw_importance", "sum"),
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    base_df["importance_pct"] = base_df["importance"] * 100.0
    base_df.insert(0, "rank", np.arange(1, len(base_df) + 1))
    base_df = base_df.rename(columns={"base_feature": "feature"})
    return stat_df, base_df


def _save_model(model, model_name: str, out_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    pkl_path = out_dir / "model.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(model, f)
    outputs["model_pickle"] = str(pkl_path.resolve())

    if model_name == "lightgbm":
        txt_path = out_dir / "model.txt"
        model.booster_.save_model(str(txt_path))
        outputs["model_text"] = str(txt_path.resolve())
    return outputs


def _fit_model(
    model,
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    use_sample_weight: bool,
) -> None:
    x_train_arr = np.asarray(x_train, dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    x_val_arr = None if x_val is None else np.asarray(x_val, dtype=np.float32)
    y_val_arr = None if y_val is None else np.asarray(y_val, dtype=np.int64)

    fit_kwargs: dict[str, Any] = {}
    if use_sample_weight:
        fit_kwargs["sample_weight"] = np.asarray(w_train, dtype=np.float32)

    if model_name == "lightgbm" and x_val_arr is not None and y_val_arr is not None:
        fit_kwargs["eval_set"] = [(x_val_arr, y_val_arr)]
        fit_kwargs["eval_metric"] = "multi_logloss"

    model.fit(x_train_arr, y_train_arr, **fit_kwargs)


def main() -> None:
    args = _parse_args()
    _seed_everything(args.seed)

    root = Path(".").resolve()
    cfg = TrainConfig.from_json(args.config).resolve_paths(base_dir=root)
    feature_cols = load_feature_list(cfg.data.feature_list)
    label_map = load_label_map(cfg.data.label_map)
    class_names = _label_names_by_id(label_map)

    train_table, train_index, train_max = _split_paths(cfg, "train")
    val_table, val_index, val_max = _split_paths(cfg, "val")
    test_table, test_index, test_max = _split_paths(cfg, "test")
    if args.max_train_windows is not None:
        train_max = args.max_train_windows
    if args.max_eval_windows is not None:
        val_max = args.max_eval_windows
        test_max = args.max_eval_windows

    print("Loading train windows...")
    x_train, y_train, w_train = _load_window_matrix(
        table_path=train_table,
        index_path=train_index,
        feature_cols=feature_cols,
        max_windows=train_max,
    )
    print("Loading val windows...")
    x_val, y_val, _w_val = _load_window_matrix(
        table_path=val_table,
        index_path=val_index,
        feature_cols=feature_cols,
        max_windows=val_max,
    )
    print("Loading test windows...")
    x_test, y_test, _w_test = _load_window_matrix(
        table_path=test_table,
        index_path=test_index,
        feature_cols=feature_cols,
        max_windows=test_max,
    )

    model = _create_model(
        name=args.model,
        n_estimators=args.n_estimators,
        seed=args.seed,
        class_weight=args.class_weight,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
    )
    use_sample_weight = not args.no_sample_weight
    _fit_model(
        model=model,
        model_name=args.model,
        x_train=x_train,
        y_train=y_train,
        w_train=w_train,
        x_val=x_val,
        y_val=y_val,
        use_sample_weight=use_sample_weight,
    )

    train_metrics, _ = _evaluate_split(model, x_train, y_train, class_names)
    val_metrics, _ = _evaluate_split(model, x_val, y_val, class_names)
    test_metrics, _ = _evaluate_split(model, x_test, y_test, class_names)

    raw_importance, split_importance = _raw_importance(
        model=model,
        model_name=args.model,
        feature_count=x_train.shape[1],
    )
    stat_df, base_df = _build_importance_frames(
        raw_importance=raw_importance,
        split_importance=split_importance,
        feature_cols=feature_cols,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_outputs = _save_model(model, args.model, out_dir)

    stat_path = out_dir / "stat_feature_importance.csv"
    base_path = out_dir / "base_feature_importance.csv"
    schema_path = out_dir / "feature_schema.json"
    summary_path = out_dir / "summary.json"

    stat_df.to_csv(stat_path, index=False, encoding="utf-8-sig")
    base_df.to_csv(base_path, index=False, encoding="utf-8-sig")

    stat_feature_names, _, _ = _build_stat_names(feature_cols)
    schema = {
        "model": args.model,
        "base_features": feature_cols,
        "window_stats": list(WINDOW_STATS),
        "stat_features": stat_feature_names,
        "label_map": label_map,
        "label_names": class_names,
        "num_base_features": int(len(feature_cols)),
        "num_stat_features": int(len(stat_feature_names)),
    }
    _save_json(schema_path, schema)

    summary = {
        "model": args.model,
        "class_weight": args.class_weight,
        "use_sample_weight": use_sample_weight,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate if args.model == "lightgbm" else None,
        "num_leaves": args.num_leaves if args.model == "lightgbm" else None,
        "num_train_samples": int(len(y_train)),
        "num_val_samples": int(len(y_val)),
        "num_test_samples": int(len(y_test)),
        "num_base_features": int(len(feature_cols)),
        "num_stat_features": int(x_train.shape[1]),
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "outputs": {
            **model_outputs,
            "feature_schema": str(schema_path.resolve()),
            "stat_feature_importance": str(stat_path.resolve()),
            "base_feature_importance": str(base_path.resolve()),
        },
    }
    _save_json(summary_path, summary)
    _save_json(out_dir / "metrics_train.json", train_metrics)
    _save_json(out_dir / "metrics_val.json", val_metrics)
    _save_json(out_dir / "metrics_test.json", test_metrics)

    print(f"saved outputs in: {out_dir.resolve()}")
    print(
        "test "
        f"macro_f1={test_metrics['macro_f1']:.5f} "
        f"accuracy={test_metrics['accuracy']:.5f}"
    )


if __name__ == "__main__":
    main()
