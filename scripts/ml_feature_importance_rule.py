from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mtsc_train.config import TrainConfig
from mtsc_train.data import load_feature_list, load_label_map
from mtsc_train.metrics import confusion_matrix, summarize_confusion
from mtsc_postprocess import PostRuleConfig, load_scaler_stats


WINDOW_STATS = ("mean", "std", "min", "max", "first", "last", "delta")
RULE_FEATURES = (
    "rule_temp_upper_med5",
    "rule_temp_middle_med5",
    "rule_temp_avg_med5",
    "rule_temp_slope_10m",
    "rule_temp_high_both",
    "rule_temp_low_both",
    "rule_o2_high",
    "rule_feed_zero",
    "rule_run_zero_ok",
    "rule_shutdown_zero_ok",
    "rule_stop_hit",
    "rule_cooldown_hit",
    "rule_shutdown_hit",
    "rule_bake_hit",
    "rule_startup_hit",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ml-feature-importance-rule")
    p.add_argument("--config", default="configs/train.json", help="Training config path")
    p.add_argument("--rules-config", default="configs/post_rules.json", help="Rule config path")
    p.add_argument("--split", default="val", choices=["train", "val", "test"], help="Eval split")
    p.add_argument(
        "--model",
        default="lightgbm",
        choices=["lightgbm", "extratrees"],
        help="Machine-learning model used for feature importance",
    )
    p.add_argument("--out-dir", default="artifacts/ml_feature_importance", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--n-estimators", type=int, default=500, help="Number of trees")
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
        help="Optional analysis-only cap for training windows",
    )
    p.add_argument(
        "--max-eval-windows",
        type=int,
        default=None,
        help="Optional analysis-only cap for eval windows",
    )
    return p.parse_args()


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


def _rolling_median(values: np.ndarray, win_steps: int) -> np.ndarray:
    n = int(len(values))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - win_steps + 1)
        out[i] = float(np.median(values[start : i + 1]))
    return out


def _rolling_linear_slope(values: np.ndarray, step_minutes: float, win_steps: int) -> np.ndarray:
    n = int(len(values))
    out = np.zeros(n, dtype=np.float64)
    step = max(float(step_minutes), 1e-6)
    for i in range(n):
        start = max(0, i - win_steps + 1)
        y = values[start : i + 1]
        k = int(len(y))
        if k < 2:
            continue
        x = np.arange(k, dtype=np.float64) * step
        x_center = x - x.mean()
        denom = float(np.dot(x_center, x_center))
        if denom <= 0:
            continue
        y_center = y - y.mean()
        out[i] = float(np.dot(x_center, y_center) / denom)
    return out


def _add_rule_features(
    table_df: pd.DataFrame,
    cfg: PostRuleConfig,
    scaler_stats: dict,
) -> pd.DataFrame:
    out = table_df.copy()
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
        raise KeyError(f"Rule feature columns missing in table: {missing}")

    for col in sorted(required_cols):
        phys_col = f"{col}__rule_phys"
        if cfg.inverse_scale:
            out[phys_col] = _inverse_scale_column(out, col, phys_col, scaler_stats)
        else:
            out[phys_col] = out[col].astype(float)

    upper_col = f"{fm.temperature_upper_col}__rule_phys"
    middle_col = f"{fm.temperature_middle_col}__rule_phys"
    oxygen_col = f"{fm.oxygen_col}__rule_phys"
    pusher1_col = f"{fm.pusher1_col}__rule_phys"
    pusher2_col = f"{fm.pusher2_col}__rule_phys"
    run_phys_cols = [f"{c}__rule_phys" for c in fm.run_zero_cols]
    shutdown_phys_cols = [f"{c}__rule_phys" for c in fm.shutdown_zero_cols]

    out["rule_temp_upper_med5"] = 0.0
    out["rule_temp_middle_med5"] = 0.0
    out["rule_temp_avg_med5"] = 0.0
    out["rule_temp_slope_10m"] = 0.0

    for (_, _), idx in out.groupby(["source", "furnace_id"], sort=False).groups.items():
        g = out.loc[idx].sort_values("timestamp")
        step_sec = _median_step_seconds(g["timestamp"])
        med_steps = max(1, int(round(5 * 60.0 / max(step_sec, 1e-6))))
        slope_steps = max(2, int(round(int(cfg.timing.slope_window_minutes) * 60.0 / max(step_sec, 1e-6))))

        upper_med = _rolling_median(g[upper_col].to_numpy(dtype=np.float64), med_steps)
        middle_med = _rolling_median(g[middle_col].to_numpy(dtype=np.float64), med_steps)
        avg_med = (upper_med + middle_med) / 2.0
        slope = _rolling_linear_slope(
            avg_med,
            step_minutes=step_sec / 60.0,
            win_steps=slope_steps,
        )

        out.loc[g.index, "rule_temp_upper_med5"] = upper_med
        out.loc[g.index, "rule_temp_middle_med5"] = middle_med
        out.loc[g.index, "rule_temp_avg_med5"] = avg_med
        out.loc[g.index, "rule_temp_slope_10m"] = slope

    thr = cfg.thresholds
    feed_zero = (
        (out[pusher1_col].to_numpy(dtype=np.float64) == 0.0)
        & (out[pusher2_col].to_numpy(dtype=np.float64) == 0.0)
    )

    run_zero_ok = np.ones(len(out), dtype=bool)
    for col in run_phys_cols:
        run_zero_ok &= out[col].to_numpy(dtype=np.float64) == 0.0

    shutdown_zero_ok = np.ones(len(out), dtype=bool)
    for col in shutdown_phys_cols:
        shutdown_zero_ok &= out[col].to_numpy(dtype=np.float64) == 0.0

    high_both = (
        (out["rule_temp_upper_med5"].to_numpy(dtype=np.float64) > float(thr.temp_high))
        & (out["rule_temp_middle_med5"].to_numpy(dtype=np.float64) > float(thr.temp_high))
    )
    low_both = (
        (out["rule_temp_upper_med5"].to_numpy(dtype=np.float64) < float(thr.temp_low))
        & (out["rule_temp_middle_med5"].to_numpy(dtype=np.float64) < float(thr.temp_low))
    )
    o2_high = out[oxygen_col].to_numpy(dtype=np.float64) > float(thr.o2_high)
    slope = out["rule_temp_slope_10m"].to_numpy(dtype=np.float64)

    out["rule_temp_high_both"] = high_both.astype(np.float32)
    out["rule_temp_low_both"] = low_both.astype(np.float32)
    out["rule_o2_high"] = o2_high.astype(np.float32)
    out["rule_feed_zero"] = feed_zero.astype(np.float32)
    out["rule_run_zero_ok"] = run_zero_ok.astype(np.float32)
    out["rule_shutdown_zero_ok"] = shutdown_zero_ok.astype(np.float32)

    normal_like_ref = float(thr.stop_normal_like_fallback_abs)
    stop_slope_band = normal_like_ref * float(thr.stop_normal_like_factor) + float(
        thr.stop_normal_like_margin
    )
    out["rule_stop_hit"] = (high_both & feed_zero & (np.abs(slope) <= stop_slope_band)).astype(
        np.float32
    )
    out["rule_cooldown_hit"] = (feed_zero & (slope < float(thr.cooldown_slope_lt))).astype(
        np.float32
    )
    out["rule_shutdown_hit"] = (
        low_both & o2_high & run_zero_ok & shutdown_zero_ok
    ).astype(np.float32)
    out["rule_bake_hit"] = (feed_zero & (slope > float(thr.bake_slope_gt))).astype(np.float32)
    out["rule_startup_hit"] = high_both.astype(np.float32)

    return out


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
    return cfg.data.test_table, cfg.data.test_window_index, cfg.data.max_eval_windows


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
    rules_cfg: PostRuleConfig,
    scaler_stats: dict,
    max_windows: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_df = _read_window_index(index_path, max_windows=max_windows)
    if index_df.empty:
        raise ValueError(f"No windows available in {index_path}")

    required = ["start_row_id", "end_row_id", "label_id", "sample_weight"]
    missing = [c for c in required if c not in index_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in window index {index_path}: {missing}")

    fm = rules_cfg.feature_mapping
    rule_source_cols = [
        fm.temperature_upper_col,
        fm.temperature_middle_col,
        fm.oxygen_col,
        fm.pusher1_col,
        fm.pusher2_col,
        *fm.run_zero_cols,
        *fm.shutdown_zero_cols,
    ]
    usecols = sorted(set(["source", "furnace_id", "timestamp", *feature_cols, *rule_source_cols]))
    table_df = _read_table(table_path, usecols=usecols)
    table_df = table_df.sort_values(["source", "furnace_id", "timestamp"], kind="mergesort")
    table_df = table_df.reset_index(drop=True)
    table_df = _add_rule_features(table_df, cfg=rules_cfg, scaler_stats=scaler_stats)

    max_row = int(index_df["end_row_id"].max())
    if max_row >= len(table_df):
        raise ValueError(
            f"Window row id exceeds table size: max_row_id={max_row}, table_rows={len(table_df)}"
        )

    all_feature_cols = [*feature_cols, *RULE_FEATURES]
    raw_features = table_df[all_feature_cols].to_numpy(dtype=np.float32, copy=True)
    starts = index_df["start_row_id"].to_numpy(dtype=np.int64, copy=True)
    ends = index_df["end_row_id"].to_numpy(dtype=np.int64, copy=True)
    labels = index_df["label_id"].to_numpy(dtype=np.int64, copy=True)
    sample_weight = index_df["sample_weight"].to_numpy(dtype=np.float32, copy=True)

    x = np.empty((len(index_df), len(all_feature_cols) * len(WINDOW_STATS)), dtype=np.float32)
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        x[i] = _window_to_stats(raw_features[int(start) : int(end) + 1])

    return x, labels, sample_weight


def _create_model(name: str, n_estimators: int, seed: int, class_weight: str):
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
            learning_rate=0.05,
            num_leaves=31,
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


def _raw_importance(model, model_name: str, feature_count: int) -> tuple[np.ndarray, np.ndarray | None]:
    if model_name == "lightgbm":
        booster = model.booster_
        gain = booster.feature_importance(importance_type="gain").astype(np.float64) # 这个特征被模型用了多少次，次数越多，说明这个特征越重要（缺点：用得多不代表每次都很关键）
        split = booster.feature_importance(importance_type="split").astype(np.float64) # 这个特征对模型效果提升有多大

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


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _plot_base_importance(base_df: pd.DataFrame, path: Path, title: str, top_k: int | None = None) -> None:
    plot_df = base_df.copy()
    if top_k is not None:
        plot_df = plot_df.head(top_k).copy()
    plot_df = plot_df.sort_values("importance", ascending=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    height = max(5.0, 0.36 * len(plot_df) + 1.4)
    fig, ax = plt.subplots(figsize=(11, height), dpi=150)
    ax.barh(plot_df["feature"], plot_df["importance_pct"], color="#3b82f6")
    ax.set_xlabel("Importance (%)")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    cfg = TrainConfig.from_json(args.config).resolve_paths(base_dir=ROOT)
    rules_cfg = PostRuleConfig.from_json(args.rules_config)
    scaler_stats = load_scaler_stats(Path(cfg.data.processed_dir) / "scaler_stats.json")
    feature_cols = load_feature_list(cfg.data.feature_list)
    all_feature_cols = [*feature_cols, *RULE_FEATURES]
    label_map = load_label_map(cfg.data.label_map)
    class_names = _label_names_by_id(label_map)

    train_table, train_index, train_max = _split_paths(cfg, "train")
    eval_table, eval_index, eval_max = _split_paths(cfg, args.split)
    if args.max_train_windows is not None:
        train_max = args.max_train_windows
    if args.max_eval_windows is not None:
        eval_max = args.max_eval_windows

    x_train, y_train, w_train = _load_window_matrix(
        table_path=train_table,
        index_path=train_index,
        feature_cols=feature_cols,
        rules_cfg=rules_cfg,
        scaler_stats=scaler_stats,
        max_windows=train_max,
    )
    x_eval, y_eval, _w_eval = _load_window_matrix(
        table_path=eval_table,
        index_path=eval_index,
        feature_cols=feature_cols,
        rules_cfg=rules_cfg,
        scaler_stats=scaler_stats,
        max_windows=eval_max,
    )

    model = _create_model(
        name=args.model,
        n_estimators=args.n_estimators,
        seed=args.seed,
        class_weight=args.class_weight,
    )
    fit_kwargs = {}
    if not args.no_sample_weight:
        fit_kwargs["sample_weight"] = w_train
    model.fit(x_train, y_train, **fit_kwargs)

    pred = model.predict(x_eval)
    metrics = _compute_metrics(pred=np.asarray(pred), target=y_eval, class_names=class_names)

    raw_importance, split_importance = _raw_importance(
        model=model,
        model_name=args.model,
        feature_count=x_train.shape[1],
    )
    stat_df, base_df = _build_importance_frames(
        raw_importance=raw_importance,
        split_importance=split_importance,
        feature_cols=all_feature_cols,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stat_path = out_dir / f"stat_feature_importance_{args.split}.csv"
    base_path = out_dir / f"base_feature_importance_{args.split}.csv"
    all_png_path = out_dir / f"base_feature_importance_{args.split}.png"
    top_png_path = out_dir / f"top15_feature_importance_{args.split}.png"
    metrics_path = out_dir / f"metrics_{args.split}.json"

    stat_df.to_csv(stat_path, index=False, encoding="utf-8-sig")
    base_df.to_csv(base_path, index=False, encoding="utf-8-sig")
    _plot_base_importance(
        base_df,
        all_png_path,
        title=f"{args.split} {args.model} base feature importance",
        top_k=None,
    )
    _plot_base_importance(
        base_df,
        top_png_path,
        title=f"{args.split} {args.model} top 15 base feature importance",
        top_k=min(15, len(base_df)),
    )

    metrics_payload = {
        "split": args.split,
        "model": args.model,
        "importance_type": "gain" if args.model == "lightgbm" else "feature_importances_",
        "class_weight": args.class_weight,
        "use_sample_weight": not args.no_sample_weight,
        "n_estimators": args.n_estimators,
        "num_train_samples": int(len(y_train)),
        "num_eval_samples": int(len(y_eval)),
        "num_original_features": int(len(feature_cols)),
        "num_rule_features": int(len(RULE_FEATURES)),
        "num_base_features": int(len(all_feature_cols)),
        "num_stat_features": int(x_train.shape[1]),
        "window_stats": list(WINDOW_STATS),
        "rule_features": list(RULE_FEATURES),
        "metrics": metrics,
        "outputs": {
            "stat_feature_importance": str(stat_path.resolve()),
            "base_feature_importance": str(base_path.resolve()),
            "base_feature_importance_plot": str(all_png_path.resolve()),
            "top15_feature_importance_plot": str(top_png_path.resolve()),
        },
    }
    _save_json(metrics_path, metrics_payload)

    print(f"saved outputs in: {out_dir.resolve()}")
    print(f"macro_f1={metrics['macro_f1']:.5f} accuracy={metrics['accuracy']:.5f}")


if __name__ == "__main__":
    main()
