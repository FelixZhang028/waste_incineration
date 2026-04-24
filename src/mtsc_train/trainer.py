from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import time
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .losses import weighted_mean
from .metrics import confusion_matrix, summarize_confusion
from .models import SequenceClassifier


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_metric: float
    best_checkpoint: str
    last_checkpoint: str
    start_epoch: int
    final_epoch: int


class Trainer:
    def __init__(
        self,
        model: SequenceClassifier,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: torch.device,
        num_classes: int,
        class_names: list[str],
        output_dir: str | Path,
        monitor: str = "val_macro_f1",
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 5,
        log_every_steps: int = 100,
        use_amp: bool = False,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = int(num_classes)
        self.class_names = class_names
        self.monitor = monitor
        self.grad_clip_norm = float(grad_clip_norm)
        self.early_stopping_patience = int(early_stopping_patience)
        self.log_every_steps = int(log_every_steps)
        self.use_amp = bool(use_amp and device.type == "cuda")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "history.json"
        self.best_checkpoint_path = self.output_dir / "best.pt"
        self.last_checkpoint_path = self.output_dir / "last.pt"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        per_sample_loss = self.criterion(logits, labels)
        loss = weighted_mean(per_sample_loss, sample_weight)
        return loss, per_sample_loss

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        correct = 0
        sample_count = 0

        start_time = time.time()
        for step, batch in enumerate(loader, start=1):
            batch = self._batch_to_device(batch)
            x = batch["x"]
            lengths = batch["lengths"]
            labels = batch["label_id"]
            sample_weight = batch["sample_weight"]

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x, lengths)
                loss, per_sample_loss = self._compute_loss(logits, labels, sample_weight)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            weighted_loss_sum += float((per_sample_loss * sample_weight).sum().item())
            weight_sum += float(sample_weight.sum().item())

            preds = torch.argmax(logits.detach(), dim=1)
            correct += int((preds == labels).sum().item())
            sample_count += int(labels.numel())

            if step % self.log_every_steps == 0:
                running_loss = weighted_loss_sum / max(weight_sum, 1e-6)
                running_acc = correct / max(sample_count, 1)
                print(
                    f"[epoch {epoch}] step {step}/{len(loader)} "
                    f"loss={running_loss:.5f} acc={running_acc:.5f}"
                )

        elapsed = time.time() - start_time
        avg_loss = weighted_loss_sum / max(weight_sum, 1e-6)
        avg_acc = correct / max(sample_count, 1)
        throughput = sample_count / max(elapsed, 1e-6)
        return {"loss": avg_loss, "accuracy": avg_acc, "samples_per_sec": throughput}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str) -> dict[str, Any]:
        self.model.eval()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        conf = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        conf_by_furnace: dict[int, torch.Tensor] = {}

        for batch in loader:
            batch = self._batch_to_device(batch)
            x = batch["x"]
            lengths = batch["lengths"]
            furnace_ids = batch["furnace_id"]
            labels = batch["label_id"]
            sample_weight = batch["sample_weight"]

            logits = self.model(x, lengths)
            _, per_sample_loss = self._compute_loss(logits, labels, sample_weight)

            weighted_loss_sum += float((per_sample_loss * sample_weight).sum().item())
            weight_sum += float(sample_weight.sum().item())

            preds = torch.argmax(logits, dim=1).detach().cpu()
            labels_cpu = labels.detach().cpu()
            furnace_ids_cpu = furnace_ids.detach().cpu()

            conf += confusion_matrix(preds=preds, targets=labels_cpu, num_classes=self.num_classes)

            unique_furnaces = torch.unique(furnace_ids_cpu)
            for fid in unique_furnaces.tolist():
                mask = furnace_ids_cpu == int(fid)
                if not torch.any(mask):
                    continue
                if int(fid) not in conf_by_furnace:
                    conf_by_furnace[int(fid)] = torch.zeros(
                        (self.num_classes, self.num_classes), dtype=torch.int64
                    )
                conf_by_furnace[int(fid)] += confusion_matrix(
                    preds=preds[mask],
                    targets=labels_cpu[mask],
                    num_classes=self.num_classes,
                )

        metrics = summarize_confusion(conf, self.class_names)
        if conf_by_furnace:
            metrics["by_furnace"] = {}
            for fid in sorted(conf_by_furnace.keys()):
                furnace_metrics = summarize_confusion(conf_by_furnace[fid], self.class_names)
                furnace_metrics["split"] = split
                furnace_metrics["furnace_id"] = int(fid)
                metrics["by_furnace"][str(int(fid))] = furnace_metrics
        metrics["loss"] = weighted_loss_sum / max(weight_sum, 1e-6)
        metrics["split"] = split
        return metrics

    def _is_better(self, metric_name: str, current: float, best: float) -> bool:
        is_loss = metric_name.endswith("loss")
        if math.isnan(current):
            return False
        if math.isinf(best):
            return True
        return current < best if is_loss else current > best

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metric: float,
        run_config: dict[str, Any],
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "monitor_metric": metric,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict(),
                "run_config": run_config,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])

        optimizer_state = state.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = state.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = state.get("scaler_state_dict")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)
        return state

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epochs: int,
        run_config: dict[str, Any],
        start_epoch: int = 0,
        history: list[dict[str, float]] | None = None,
        best_metric: float | None = None,
        best_epoch: int = 0,
        no_improve_epochs: int = 0,
    ) -> TrainResult:
        if start_epoch < 0:
            raise ValueError("start_epoch cannot be negative")

        base_history = history or []
        history = [dict(row) for row in base_history]
        if best_metric is None:
            best_metric = math.inf if self.monitor.endswith("loss") else -math.inf
        best_epoch = int(best_epoch)
        no_improve_epochs = int(no_improve_epochs)

        if start_epoch >= epochs:
            print(
                f"Resume epoch {start_epoch} is already >= target epochs {epochs}, "
                "skipping training loop."
            )

        for epoch in range(start_epoch + 1, epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, epoch=epoch)
            row: dict[str, float] = {f"train_{k}": float(v) for k, v in train_metrics.items()}

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, split="val")
                row.update(
                    {
                        "val_loss": float(val_metrics["loss"]),
                        "val_accuracy": float(val_metrics["accuracy"]),
                        "val_macro_f1": float(val_metrics["macro_f1"]),
                    }
                )
            else:
                row["val_loss"] = float("nan")
                row["val_accuracy"] = float("nan")
                row["val_macro_f1"] = float("nan")

            monitor_key = self.monitor
            monitor_value = row.get(monitor_key)
            if monitor_value is None or math.isnan(monitor_value):
                monitor_key = "train_loss"
                monitor_value = float(row[monitor_key])

            row["epoch"] = float(epoch)
            row["monitor"] = float(monitor_value)
            history.append(row)

            self._save_checkpoint(
                path=self.last_checkpoint_path,
                epoch=epoch,
                metric=monitor_value,
                run_config=run_config,
            )

            if self._is_better(monitor_key, monitor_value, best_metric):
                best_metric = monitor_value
                best_epoch = epoch
                no_improve_epochs = 0
                self._save_checkpoint(
                    path=self.best_checkpoint_path,
                    epoch=epoch,
                    metric=monitor_value,
                    run_config=run_config,
                )
            else:
                no_improve_epochs += 1

            if self.scheduler is not None:
                self.scheduler.step()

            with self.history_path.open("w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            print(
                f"[epoch {epoch}] train_loss={row['train_loss']:.5f} "
                f"train_acc={row['train_accuracy']:.5f} "
                f"monitor({monitor_key})={monitor_value:.5f} best={best_metric:.5f}"
            )

            if self.early_stopping_patience > 0 and no_improve_epochs >= self.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch}, no improvement for "
                    f"{self.early_stopping_patience} epochs."
                )
                break

        if not self.best_checkpoint_path.exists():
            fallback_epoch = int(history[-1]["epoch"]) if history else int(start_epoch)
            fallback_metric = (
                float(best_metric) if not (math.isinf(best_metric) or math.isnan(best_metric)) else float("nan")
            )
            self._save_checkpoint(
                path=self.best_checkpoint_path,
                epoch=fallback_epoch,
                metric=fallback_metric,
                run_config=run_config,
            )

        best_state = torch.load(self.best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(best_state["model_state_dict"])
        final_epoch = int(history[-1]["epoch"]) if history else int(start_epoch)

        return TrainResult(
            history=history,
            best_epoch=best_epoch,
            best_metric=float(best_metric),
            best_checkpoint=str(self.best_checkpoint_path),
            last_checkpoint=str(self.last_checkpoint_path),
            start_epoch=int(start_epoch),
            final_epoch=final_epoch,
        )
