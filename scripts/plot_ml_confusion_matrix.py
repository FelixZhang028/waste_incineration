from __future__ import annotations

import argparse
from pathlib import Path
import pickle
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
from mtsc_train.metrics import confusion_matrix
from mtsc_train.ml.trainer import _label_names_by_id, _load_window_matrix, _split_paths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("plot-ml-confusion-matrix")
    p.add_argument("--config", default="configs/train.json", help="Training config path")
    p.add_argument("--model-path", default="artifacts/train_ml_catboost/model.pkl", help="Pickled ML model path")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    p.add_argument("--out-dir", default=None, help="Output directory; defaults to model path parent")
    p.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap for windows, useful for quick checks",
    )
    return p.parse_args()


def _save_confusion_plot(conf_norm: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(conf_norm, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir) if args.out_dir is not None else model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig.from_json(args.config).resolve_paths(base_dir=ROOT)
    feature_cols = load_feature_list(cfg.data.feature_list)
    label_map = load_label_map(cfg.data.label_map)
    class_names = _label_names_by_id(label_map)

    table_path, index_path, max_windows = _split_paths(cfg, args.split)
    if args.max_windows is not None:
        max_windows = args.max_windows

    x, y, _sample_weight = _load_window_matrix(
        table_path=table_path,
        index_path=index_path,
        feature_cols=feature_cols,
        max_windows=max_windows,
    )

    with model_path.open("rb") as f:
        model = pickle.load(f)

    pred = np.asarray(model.predict(x), dtype=np.int64).reshape(-1)
    conf = confusion_matrix(
        preds=torch.from_numpy(pred),
        targets=torch.from_numpy(y.astype(np.int64)),
        num_classes=len(class_names),
    ).numpy()

    conf_df = pd.DataFrame(conf, index=class_names, columns=class_names)
    row_sum = conf.sum(axis=1, keepdims=True).astype(np.float64)
    conf_norm = np.divide(
        conf.astype(np.float64),
        np.clip(row_sum, 1.0, None),
        out=np.zeros_like(conf, dtype=np.float64),
        where=row_sum != 0,
    )
    conf_norm_df = pd.DataFrame(conf_norm, index=class_names, columns=class_names)

    raw_path = out_dir / f"confusion_matrix_{args.split}.csv"
    norm_path = out_dir / f"confusion_matrix_{args.split}_normalized.csv"
    png_path = out_dir / f"confusion_matrix_{args.split}.png"

    conf_df.to_csv(raw_path, encoding="utf-8-sig")
    conf_norm_df.to_csv(norm_path, encoding="utf-8-sig")
    _save_confusion_plot(
        conf_norm=conf_norm,
        class_names=class_names,
        path=png_path,
        title=f"{args.split.title()} Confusion Matrix (ML, row-normalized)",
    )

    print(f"saved: {raw_path}")
    print(f"saved: {norm_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
