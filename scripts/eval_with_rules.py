from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mtsc_postprocess import PostRuleConfig, apply_rules, build_rule_features, load_scaler_stats
from mtsc_train.config import TrainConfig
from mtsc_train.data import WindowIndexDataset, build_loader, load_feature_list, load_label_map
from mtsc_train.metrics import confusion_matrix, summarize_confusion
from mtsc_train.registry import create_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("eval-with-rules")
    p.add_argument("--config", default="configs/train.json", help="Training config path")
    p.add_argument("--checkpoint", default="artifacts/train/best.pt", help="Model checkpoint")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="Eval split")
    p.add_argument("--out-dir", default="artifacts/eval", help="Output directory")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    p.add_argument("--with-rules", action="store_true", help="Enable post rules")
    p.add_argument(
        "--rules-config",
        default="configs/post_rules.json",
        help="Rule config path (used when --with-rules)",
    )
    return p.parse_args()


def _select_device(raw: str) -> torch.device:
    key = raw.lower().strip()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


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


def _read_table(path: str | Path, usecols: list[str], nrows: int | None = None) -> pd.DataFrame:
    target = _resolve_existing_table(path)
    if target.suffix.lower() == ".parquet":
        return pd.read_parquet(target, columns=usecols)
    if target.suffix.lower() == ".csv":
        return pd.read_csv(target, usecols=usecols, nrows=nrows, low_memory=False)
    raise ValueError(f"Unsupported table format: {target.suffix}")


def _load_window_index(path: str | Path, max_windows: int | None) -> pd.DataFrame:
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
    got = [idx for idx, _ in pairs]
    if got != list(range(len(pairs))):
        raise ValueError(f"label ids must be contiguous from 0, got {got}")
    return [name for _, name in pairs]


def _split_paths(cfg: TrainConfig, split: str) -> tuple[str, str, int | None]:
    if split == "train":
        return cfg.data.train_table, cfg.data.train_window_index, cfg.data.max_train_windows
    if split == "val":
        return cfg.data.val_table, cfg.data.val_window_index, cfg.data.max_eval_windows
    return cfg.data.test_table, cfg.data.test_window_index, cfg.data.max_eval_windows


def _infer_raw_predictions(
    cfg: TrainConfig,
    feature_cols: list[str],
    num_classes: int,
    class_names: list[str],
    split: str,
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table_path, index_path, max_windows = _split_paths(cfg, split)
    dataset = WindowIndexDataset(
        table_path=table_path,
        window_index_path=index_path,
        feature_cols=feature_cols,
        max_windows=max_windows,
    )
    loader = build_loader(
        dataset=dataset,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    model = create_model(
        name=cfg.model.name,
        input_size=len(feature_cols),
        num_classes=num_classes,
        params=cfg.model.params,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pred_ids: list[np.ndarray] = []
    pred_conf: list[np.ndarray] = []
    true_ids: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label_id"].cpu().numpy()

            logits = model(x, lengths)
            prob = torch.softmax(logits, dim=1)
            conf, pred = prob.max(dim=1)

            pred_ids.append(pred.cpu().numpy())
            pred_conf.append(conf.cpu().numpy())
            true_ids.append(labels)

    return (
        np.concatenate(pred_ids, axis=0),
        np.concatenate(pred_conf, axis=0),
        np.concatenate(true_ids, axis=0),
    )


def _build_prediction_frame(
    cfg: TrainConfig,
    split: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    table_path, index_path, max_windows = _split_paths(cfg, split)
    index_df = _load_window_index(index_path, max_windows=max_windows)

    usecols = sorted(set(["source", "furnace_id", "timestamp", *feature_cols]))
    table_df = _read_table(table_path, usecols=usecols)
    table_df = table_df.sort_values(["source", "furnace_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    max_row = int(index_df["end_row_id"].max()) if not index_df.empty else -1
    if max_row >= len(table_df):
        raise ValueError(
            f"Window end_row_id exceeds table size: max_row_id={max_row}, table_rows={len(table_df)}"
        )

    out = index_df[
        ["source", "furnace_id", "timestamp", "label_id", "sample_weight", "start_row_id", "end_row_id"]
    ].copy()
    end_rows = index_df["end_row_id"].to_numpy(dtype=np.int64, copy=True)

    for col in feature_cols:
        out[col] = table_df.iloc[end_rows][col].to_numpy(copy=True)

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out


def _compute_metrics(
    pred_ids: np.ndarray,
    true_ids: np.ndarray,
    num_classes: int,
    class_names: list[str],
    split: str,
) -> tuple[dict, np.ndarray]:
    preds_t = torch.from_numpy(pred_ids.astype(np.int64))
    trues_t = torch.from_numpy(true_ids.astype(np.int64))
    conf = confusion_matrix(preds=preds_t, targets=trues_t, num_classes=num_classes)
    metrics = summarize_confusion(conf, class_names)
    metrics["split"] = split
    return metrics, conf.numpy()


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _save_confusion(
    out_dir: Path,
    split: str,
    tag: str,
    conf: np.ndarray,
    class_names: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame(conf, index=class_names, columns=class_names)
    row_sum = conf.sum(axis=1, keepdims=True).astype(np.float64)
    norm = np.divide(
        conf.astype(np.float64),
        np.clip(row_sum, 1.0, None),
        out=np.zeros_like(conf, dtype=np.float64),
        where=row_sum != 0,
    )
    norm_df = pd.DataFrame(norm, index=class_names, columns=class_names)

    raw_path = out_dir / f"confusion_matrix_{split}_{tag}.csv"
    norm_path = out_dir / f"confusion_matrix_{split}_{tag}_normalized.csv"
    png_path = out_dir / f"confusion_matrix_{split}_{tag}.png"

    raw_df.to_csv(raw_path, encoding="utf-8-sig")
    norm_df.to_csv(norm_path, encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(norm, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{split.title()} Confusion Matrix ({tag}, row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    cfg = TrainConfig.from_json(args.config).resolve_paths(base_dir=ROOT)
    feature_cols = load_feature_list(cfg.data.feature_list)
    label_map = load_label_map(cfg.data.label_map)
    class_names = _label_names_by_id(label_map)
    num_classes = len(class_names)
    id_to_name = {idx: name for idx, name in enumerate(class_names)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device(args.device)

    raw_pred_ids, raw_conf, true_ids = _infer_raw_predictions(
        cfg=cfg,
        feature_cols=feature_cols,
        num_classes=num_classes,
        class_names=class_names,
        split=args.split,
        checkpoint=args.checkpoint,
        device=device,
    )

    pred_df = _build_prediction_frame(cfg=cfg, split=args.split, feature_cols=feature_cols)
    if len(pred_df) != len(raw_pred_ids):
        raise ValueError(
            f"Prediction count mismatch: frame_rows={len(pred_df)}, pred_rows={len(raw_pred_ids)}"
        )

    pred_df["raw_pred_id"] = raw_pred_ids
    pred_df["raw_pred"] = [id_to_name[int(x)] for x in raw_pred_ids]
    pred_df["raw_confidence"] = raw_conf

    raw_metrics, raw_conf_mat = _compute_metrics(
        pred_ids=raw_pred_ids,
        true_ids=true_ids,
        num_classes=num_classes,
        class_names=class_names,
        split=args.split,
    )

    _save_json(out_dir / f"{args.split}_metrics_raw.json", raw_metrics)
    _save_confusion(out_dir=out_dir, split=args.split, tag="raw", conf=raw_conf_mat, class_names=class_names)
    pred_df.to_csv(out_dir / f"pred_{args.split}_raw.csv", index=False, encoding="utf-8-sig")

    summary: dict[str, object] = {
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "raw_metrics": raw_metrics,
    }

    if args.with_rules:
        rules_cfg = PostRuleConfig.from_json(args.rules_config)
        if not rules_cfg.enabled:
            print("rules config is disabled (enabled=false), skipping rule postprocess.")
        else:
            scaler_stats_path = Path(cfg.data.processed_dir) / "scaler_stats.json"
            scaler_stats = load_scaler_stats(scaler_stats_path)
            rule_df = build_rule_features(pred_df=pred_df, cfg=rules_cfg, scaler_stats=scaler_stats)
            rule_pred = apply_rules(rule_df, rules_cfg).to_numpy(dtype=np.int64, copy=True)
            rule_df["rule_pred_id"] = rule_pred
            rule_df["rule_pred"] = [id_to_name[int(x)] for x in rule_pred]

            rule_metrics, rule_conf_mat = _compute_metrics(
                pred_ids=rule_pred,
                true_ids=true_ids,
                num_classes=num_classes,
                class_names=class_names,
                split=args.split,
            )

            _save_json(out_dir / f"{args.split}_metrics_rule.json", rule_metrics)
            _save_confusion(
                out_dir=out_dir,
                split=args.split,
                tag="rule",
                conf=rule_conf_mat,
                class_names=class_names,
            )
            rule_df.to_csv(out_dir / f"pred_{args.split}_rule.csv", index=False, encoding="utf-8-sig")
            summary["rule_metrics"] = rule_metrics

    _save_json(out_dir / f"{args.split}_summary.json", summary)
    print(f"saved outputs in: {out_dir}")


if __name__ == "__main__":
    main()
