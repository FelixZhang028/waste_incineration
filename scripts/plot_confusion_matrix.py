from __future__ import annotations

import argparse
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
from mtsc_train.data import WindowIndexDataset, build_loader, load_feature_list, load_label_map
from mtsc_train.metrics import confusion_matrix
from mtsc_train.registry import create_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("plot-confusion-matrix")
    p.add_argument("--config", default="configs/train.json", help="Training config path")
    p.add_argument("--checkpoint", default="artifacts/train/best.pt", help="Model checkpoint path")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    p.add_argument("--out-dir", default="artifacts/train", help="Output directory")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
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


def _label_names_by_id(label_map: dict[str, int]) -> list[str]:
    pairs = sorted(((idx, name) for name, idx in label_map.items()), key=lambda x: x[0])
    return [name for _, name in pairs]


def _dataset_paths(cfg: TrainConfig, split: str) -> tuple[str, str, int | None]:
    if split == "train":
        return cfg.data.train_table, cfg.data.train_window_index, cfg.data.max_train_windows
    if split == "val":
        return cfg.data.val_table, cfg.data.val_window_index, cfg.data.max_eval_windows
    return cfg.data.test_table, cfg.data.test_window_index, cfg.data.max_eval_windows


def main() -> None:
    args = _parse_args()

    cfg = TrainConfig.from_json(args.config).resolve_paths(base_dir=ROOT)
    feature_cols = load_feature_list(cfg.data.feature_list)
    label_map = load_label_map(cfg.data.label_map)
    class_names = _label_names_by_id(label_map)
    num_classes = len(class_names)

    table_path, index_path, max_windows = _dataset_paths(cfg, args.split)
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

    device = _select_device(args.device)
    model = create_model(
        name=cfg.model.name,
        input_size=len(feature_cols),
        num_classes=num_classes,
        params=cfg.model.params,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label_id"].to(device)
            logits = model(x, lengths)
            preds = torch.argmax(logits, dim=1)
            conf += confusion_matrix(preds=preds.cpu(), targets=labels.cpu(), num_classes=num_classes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_np = conf.numpy()
    conf_df = pd.DataFrame(conf_np, index=class_names, columns=class_names)

    row_sum = conf_np.sum(axis=1, keepdims=True).astype(np.float64)
    conf_norm = np.divide(
        conf_np.astype(np.float64),
        np.clip(row_sum, 1.0, None),
        out=np.zeros_like(conf_np, dtype=np.float64),
        where=row_sum != 0,
    )
    conf_norm_df = pd.DataFrame(conf_norm, index=class_names, columns=class_names)

    csv_raw = out_dir / f"confusion_matrix_{args.split}.csv"
    csv_norm = out_dir / f"confusion_matrix_{args.split}_normalized.csv"
    png_path = out_dir / f"confusion_matrix_{args.split}.png"

    conf_df.to_csv(csv_raw, encoding="utf-8-sig")
    conf_norm_df.to_csv(csv_norm, encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(conf_norm, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{args.split.title()} Confusion Matrix (Row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

    print(f"saved: {csv_raw}")
    print(f"saved: {csv_norm}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
