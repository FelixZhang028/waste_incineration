from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import random

import numpy as np
import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from .config import SchedulerConfig, TrainConfig
from .data import (
    WindowIndexDataset,
    build_loader,
    load_feature_list,
    load_label_map,
    maybe_existing_path,
)
from .losses import build_criterion, compute_balanced_class_weights
from .registry import create_model
from .trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("mtsc-train")
    parser.add_argument(
        "--config",
        default="configs/train.json",
        help="Path to training config JSON.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Override device in config.",
    )
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="Override max training epochs.",
    )
    parser.add_argument(
        "--max-train-windows",
        default=None,
        type=int,
        help="Limit number of train windows for debugging.",
    )
    parser.add_argument(
        "--max-eval-windows",
        default=None,
        type=int,
        help="Limit number of val/test windows for debugging.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint (.pt) for resuming training.",
    )
    return parser


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(raw: str) -> torch.device:
    key = raw.lower().strip()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if key == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {raw}")


def _build_optimizer(cfg: TrainConfig, params) -> torch.optim.Optimizer:
    key = cfg.optimizer.name.lower().strip()
    if key == "adam":
        return Adam(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    if key == "adamw":
        return AdamW(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    if key == "sgd":
        return SGD(
            params,
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")


def _build_scheduler(
    scheduler_cfg: SchedulerConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    key = scheduler_cfg.name.lower().strip()
    if key == "none":
        return None
    if key == "step":
        return StepLR(optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma)
    if key == "cosine":
        return CosineAnnealingLR(optimizer, T_max=scheduler_cfg.t_max, eta_min=scheduler_cfg.min_lr)
    raise ValueError(f"Unsupported scheduler: {scheduler_cfg.name}")


def _make_dataset_if_possible(
    table_path: str,
    index_path: str,
    feature_cols: list[str],
    max_windows: int | None,
) -> WindowIndexDataset | None:
    if maybe_existing_path(table_path) is None or maybe_existing_path(index_path) is None:
        return None
    try:
        return WindowIndexDataset(
            table_path=table_path,
            window_index_path=index_path,
            feature_cols=feature_cols,
            max_windows=max_windows,
        )
    except ValueError:
        return None


def _label_names_by_id(label_map: dict[str, int]) -> list[str]:
    pairs = sorted(((idx, name) for name, idx in label_map.items()), key=lambda x: x[0])
    if not pairs:
        raise ValueError("label_map cannot be empty")
    expected = list(range(len(pairs)))
    got = [idx for idx, _ in pairs]
    if got != expected:
        raise ValueError(f"label ids must be contiguous from 0, got {got}")
    return [name for _, name in pairs]


def _is_better(metric_name: str, current: float, best: float) -> bool:
    is_loss = metric_name.endswith("loss")
    if math.isnan(current):
        return False
    if math.isinf(best):
        return True
    return current < best if is_loss else current > best


def _load_history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def _extract_monitor(row: dict[str, float], monitor: str) -> float | None:
    value = row.get(monitor)
    if value is None:
        value = row.get("train_loss")
    try:
        val = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if math.isnan(val):
        try:
            fallback = float(row["train_loss"])
        except Exception:
            return None
        return fallback
    return val


def _infer_best_from_history(
    history: list[dict[str, float]],
    monitor: str,
) -> tuple[float, int]:
    best_metric = math.inf if monitor.endswith("loss") else -math.inf
    best_epoch = 0
    for row in history:
        metric = _extract_monitor(row, monitor)
        if metric is None:
            continue
        try:
            epoch = int(float(row.get("epoch", 0)))
        except Exception:
            continue
        if _is_better(monitor, metric, best_metric):
            best_metric = metric
            best_epoch = epoch
    return best_metric, best_epoch


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = TrainConfig.from_json(args.config).resolve_paths(base_dir=Path("."))
    if args.device:
        cfg.trainer.device = args.device
    if args.epochs is not None:
        cfg.trainer.epochs = args.epochs
    if args.max_train_windows is not None:
        cfg.data.max_train_windows = args.max_train_windows
    if args.max_eval_windows is not None:
        cfg.data.max_eval_windows = args.max_eval_windows
    cfg.validate()

    _seed_everything(cfg.trainer.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    feature_cols = load_feature_list(cfg.data.feature_list)
    label_map = load_label_map(cfg.data.label_map)
    class_names = _label_names_by_id(label_map)
    num_classes = len(class_names)

    train_dataset = WindowIndexDataset(
        table_path=cfg.data.train_table,
        window_index_path=cfg.data.train_window_index,
        feature_cols=feature_cols,
        max_windows=cfg.data.max_train_windows,
    )
    val_dataset = _make_dataset_if_possible(
        table_path=cfg.data.val_table,
        index_path=cfg.data.val_window_index,
        feature_cols=feature_cols,
        max_windows=cfg.data.max_eval_windows,
    )
    test_dataset = _make_dataset_if_possible(
        table_path=cfg.data.test_table,
        index_path=cfg.data.test_window_index,
        feature_cols=feature_cols,
        max_windows=cfg.data.max_eval_windows,
    )

    train_loader = build_loader(
        dataset=train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    val_loader = (
        build_loader(
            dataset=val_dataset,
            batch_size=cfg.data.eval_batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
        if val_dataset is not None
        else None
    )
    test_loader = (
        build_loader(
            dataset=test_dataset,
            batch_size=cfg.data.eval_batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
        if test_dataset is not None
        else None
    )

    device = _select_device(cfg.trainer.device)
    model = create_model(
        name=cfg.model.name,
        input_size=len(feature_cols),
        num_classes=num_classes,
        params=cfg.model.params,
    ).to(device)

    class_weights = None
    if cfg.loss.class_weight == "balanced":
        counts = train_dataset.class_counts(num_classes=num_classes)
        class_weights = torch.from_numpy(compute_balanced_class_weights(counts)).to(device)

    criterion = build_criterion(
        name=cfg.loss.name,
        gamma=cfg.loss.gamma,
        class_weights=class_weights,
    ).to(device)

    optimizer = _build_optimizer(cfg, model.parameters())
    scheduler = _build_scheduler(cfg.scheduler, optimizer)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        output_dir=cfg.trainer.output_dir,
        monitor=cfg.trainer.monitor,
        grad_clip_norm=cfg.trainer.grad_clip_norm,
        early_stopping_patience=cfg.trainer.early_stopping_patience,
        log_every_steps=cfg.trainer.log_every_steps,
        use_amp=cfg.trainer.use_amp,
    )

    resume_path = None
    start_epoch = 0
    resume_history: list[dict[str, float]] = []
    best_metric = math.inf if cfg.trainer.monitor.endswith("loss") else -math.inf
    best_epoch = 0
    no_improve_epochs = 0

    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_absolute():
            resume_path = (Path(".") / resume_path).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = trainer.load_checkpoint(resume_path)
        start_epoch = int(checkpoint.get("epoch", 0))
        resume_history = _load_history(trainer.history_path)

        if resume_history:
            best_metric, best_epoch = _infer_best_from_history(resume_history, cfg.trainer.monitor)
        elif trainer.best_checkpoint_path.exists():
            best_state = torch.load(trainer.best_checkpoint_path, map_location=device)
            best_metric = float(best_state.get("monitor_metric", best_metric))
            best_epoch = int(best_state.get("epoch", start_epoch))
        else:
            if checkpoint.get("monitor_metric") is not None:
                best_metric = float(checkpoint["monitor_metric"])
            best_epoch = start_epoch

        if best_epoch <= 0 and start_epoch > 0:
            best_epoch = start_epoch
        no_improve_epochs = max(0, start_epoch - best_epoch)
        print(f"Resuming from checkpoint: {resume_path} (start epoch: {start_epoch})")

    run_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.trainer.epochs,
        run_config=cfg.to_dict(),
        start_epoch=start_epoch,
        history=resume_history,
        best_metric=best_metric,
        best_epoch=best_epoch,
        no_improve_epochs=no_improve_epochs,
    )

    test_metrics = {}
    if test_loader is not None:
        test_metrics = trainer.evaluate(test_loader, split="test")
        test_report_path = Path(cfg.trainer.output_dir) / "test_metrics.json"
        with test_report_path.open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)
        by_furnace = test_metrics.get("by_furnace")
        if isinstance(by_furnace, dict):
            for fid in sorted(by_furnace.keys(), key=lambda x: int(x)):
                item = by_furnace[fid]
                if not isinstance(item, dict):
                    continue
                print(
                    f"test furnace {fid}: "
                    f"acc={float(item.get('accuracy', float('nan'))):.5f} "
                    f"macro_f1={float(item.get('macro_f1', float('nan'))):.5f} "
                    f"samples={int(item.get('samples', 0))}"
                )

    summary = {
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "start_epoch": run_result.start_epoch,
        "final_epoch": run_result.final_epoch,
        "best_epoch": run_result.best_epoch,
        "best_metric": run_result.best_metric,
        "best_checkpoint": run_result.best_checkpoint,
        "last_checkpoint": run_result.last_checkpoint,
        "test_metrics": test_metrics,
    }
    summary_path = Path(cfg.trainer.output_dir) / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training completed.")
    print(f"best epoch: {run_result.best_epoch}")
    print(f"best metric: {run_result.best_metric:.5f}")
    print(f"output dir: {cfg.trainer.output_dir}")


if __name__ == "__main__":
    main()
