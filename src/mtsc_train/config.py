from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any


def _resolve_path(base_dir: Path, raw_path: str) -> str:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((base_dir / candidate).resolve())


@dataclass
class DataConfig:
    processed_dir: str = "data/processed"
    train_table: str = "dataset_train.csv"
    val_table: str = "dataset_val.csv"
    test_table: str = "dataset_test.csv"
    train_window_index: str = "window_index_train.csv"
    val_window_index: str = "window_index_val.csv"
    test_window_index: str = "window_index_test.csv"
    feature_list: str = "feature_list.json"
    label_map: str = "label_map.json"
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = True
    max_train_windows: int | None = None
    max_eval_windows: int | None = None

    def resolve_paths(self, base_dir: str | Path) -> "DataConfig":
        root = Path(base_dir)
        processed_root = Path(_resolve_path(root, self.processed_dir))

        def from_processed(raw_path: str) -> str:
            raw = Path(raw_path)
            if raw.is_absolute():
                return str(raw.resolve())
            return str((processed_root / raw).resolve())

        return DataConfig(
            processed_dir=str(processed_root),
            train_table=from_processed(self.train_table),
            val_table=from_processed(self.val_table),
            test_table=from_processed(self.test_table),
            train_window_index=from_processed(self.train_window_index),
            val_window_index=from_processed(self.val_window_index),
            test_window_index=from_processed(self.test_window_index),
            feature_list=from_processed(self.feature_list),
            label_map=from_processed(self.label_map),
            batch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            max_train_windows=self.max_train_windows,
            max_eval_windows=self.max_eval_windows,
        )


@dataclass
class ModelConfig:
    name: str = "lstm"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    name: str = "cross_entropy"  # cross_entropy | focal
    gamma: float = 2.0
    class_weight: str = "none"  # none | balanced


@dataclass
class OptimizerConfig:
    name: str = "adam"  # adam | adamw | sgd
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    name: str = "none"  # none | step | cosine
    step_size: int = 5
    gamma: float = 0.5
    t_max: int = 20
    min_lr: float = 1e-6


@dataclass
class TrainerConfig:
    epochs: int = 20
    device: str = "auto"  # auto | cpu | cuda
    seed: int = 42
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    monitor: str = "val_macro_f1"
    log_every_steps: int = 100
    output_dir: str = "artifacts/train"
    use_amp: bool = False

    def resolve_paths(self, base_dir: str | Path) -> "TrainerConfig":
        root = Path(base_dir)
        return TrainerConfig(
            epochs=self.epochs,
            device=self.device,
            seed=self.seed,
            grad_clip_norm=self.grad_clip_norm,
            early_stopping_patience=self.early_stopping_patience,
            monitor=self.monitor,
            log_every_steps=self.log_every_steps,
            output_dir=_resolve_path(root, self.output_dir),
            use_amp=self.use_amp,
        )


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def validate(self) -> None:
        if self.model.name.strip() == "":
            raise ValueError("model.name cannot be empty.")
        if self.loss.name not in {"cross_entropy", "focal"}:
            raise ValueError("loss.name must be one of: cross_entropy, focal")
        if self.loss.class_weight not in {"none", "balanced"}:
            raise ValueError("loss.class_weight must be one of: none, balanced")
        if self.optimizer.name not in {"adam", "adamw", "sgd"}:
            raise ValueError("optimizer.name must be one of: adam, adamw, sgd")
        if self.scheduler.name not in {"none", "step", "cosine"}:
            raise ValueError("scheduler.name must be one of: none, step, cosine")
        if self.data.batch_size <= 0 or self.data.eval_batch_size <= 0:
            raise ValueError("batch_size and eval_batch_size must be positive")
        if self.data.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.trainer.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.trainer.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        if self.trainer.log_every_steps <= 0:
            raise ValueError("log_every_steps must be positive")
        if self.trainer.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")

    def resolve_paths(self, base_dir: str | Path) -> "TrainConfig":
        return TrainConfig(
            data=self.data.resolve_paths(base_dir),
            model=self.model,
            loss=self.loss,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            trainer=self.trainer.resolve_paths(base_dir),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainConfig":
        cfg = cls(
            data=DataConfig(**payload.get("data", {})),
            model=ModelConfig(**payload.get("model", {})),
            loss=LossConfig(**payload.get("loss", {})),
            optimizer=OptimizerConfig(**payload.get("optimizer", {})),
            scheduler=SchedulerConfig(**payload.get("scheduler", {})),
            trainer=TrainerConfig(**payload.get("trainer", {})),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
