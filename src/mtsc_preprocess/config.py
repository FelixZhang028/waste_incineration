from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any


@dataclass
class DataSourceConfig:
    name: str
    data_path: str
    label_path: str
    encoding: str | None = None


@dataclass
class PipelineConfig:
    point_table_path: str
    output_dir: str
    data_sources: list[DataSourceConfig]
    split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)
    split_strategy: str = "date_ratio"  # date_ratio | source_holdout
    train_sources: list[str] = field(default_factory=list)
    val_sources: list[str] = field(default_factory=list)
    test_sources: list[str] = field(default_factory=list)
    mode: str = "split"  # split | full
    transition_buffer_minutes: int = 10
    transition_strategy: str = "drop"  # drop | down_weight | keep
    transition_weight: float = 0.2
    drop_labels: list[str] = field(default_factory=lambda: ["故障"])
    exclude_features: list[str] = field(default_factory=list)
    outlier_quantiles: tuple[float, float] = (0.001, 0.999)
    impute_method: str = "ffill_bfill_median"
    diff_lags: list[int] = field(default_factory=lambda: [1])
    rolling_windows_minutes: list[int] = field(default_factory=lambda: [5, 15])
    build_window_samples: bool = True
    window_minutes: int = 30
    window_export_mode: str = "index"  # index | dense
    normal_label: str = "正常运行"
    monitor_to_furnace: dict[str, int] = field(
        default_factory=lambda: {"FQA70081": 1, "FQA70082": 2, "FQA70083": 3}
    )

    def validate(self) -> None:
        if self.mode not in {"split", "full"}:
            raise ValueError("mode must be one of: split, full")
        if self.split_strategy not in {"date_ratio", "source_holdout"}:
            raise ValueError("split_strategy must be one of: date_ratio, source_holdout")
        if self.transition_strategy not in {"drop", "down_weight", "keep"}:
            raise ValueError("transition_strategy must be one of: drop, down_weight, keep")
        if self.window_export_mode not in {"index", "dense"}:
            raise ValueError("window_export_mode must be one of: index, dense")
        if len(self.split_ratios) != 3:
            raise ValueError("split_ratios must contain exactly 3 numbers")
        if any(x < 0 for x in self.split_ratios):
            raise ValueError("split_ratios cannot contain negative values")
        if self.mode == "split" and self.split_strategy == "date_ratio":
            ratio_sum = sum(self.split_ratios)
            if abs(ratio_sum - 1.0) > 1e-9:
                raise ValueError(f"split_ratios must sum to 1.0, got {ratio_sum}")
        if self.split_strategy == "source_holdout" and self.mode == "split":
            if not self.train_sources:
                raise ValueError("train_sources cannot be empty when split_strategy=source_holdout")
            if not self.test_sources and not self.val_sources:
                raise ValueError(
                    "At least one of test_sources or val_sources must be set when split_strategy=source_holdout"
                )
        q_low, q_high = self.outlier_quantiles
        if not (0 <= q_low < q_high <= 1):
            raise ValueError("outlier_quantiles must satisfy 0 <= low < high <= 1")
        if self.impute_method not in {"ffill_bfill_median"}:
            raise ValueError("impute_method must be ffill_bfill_median")
        if any(x <= 0 for x in self.diff_lags):
            raise ValueError("diff_lags must be positive integers")
        if any(x <= 0 for x in self.rolling_windows_minutes):
            raise ValueError("rolling_windows_minutes must be positive integers")
        if self.window_minutes <= 0:
            raise ValueError("window_minutes must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        raw_sources = data.get("data_sources", [])
        if not raw_sources:
            raise ValueError("data_sources cannot be empty")
        sources = [DataSourceConfig(**item) for item in raw_sources]

        split = tuple(data.get("split_ratios", (0.7, 0.1, 0.2)))
        cfg = cls(
            point_table_path=data["point_table_path"],
            output_dir=data["output_dir"],
            data_sources=sources,
            split_ratios=split,  # type: ignore[arg-type]
            split_strategy=data.get("split_strategy", "date_ratio"),
            train_sources=list(data.get("train_sources", [])),
            val_sources=list(data.get("val_sources", [])),
            test_sources=list(data.get("test_sources", [])),
            mode=data.get("mode", "split"),
            transition_buffer_minutes=int(data.get("transition_buffer_minutes", 10)),
            transition_strategy=data.get("transition_strategy", "drop"),
            transition_weight=float(data.get("transition_weight", 0.2)),
            drop_labels=list(data.get("drop_labels", ["故障"])),
            exclude_features=list(data.get("exclude_features", [])),
            outlier_quantiles=tuple(data.get("outlier_quantiles", [0.001, 0.999])),
            impute_method=data.get("impute_method", "ffill_bfill_median"),
            diff_lags=list(data.get("diff_lags", [1])),
            rolling_windows_minutes=list(data.get("rolling_windows_minutes", [5, 15])),
            build_window_samples=bool(data.get("build_window_samples", True)),
            window_minutes=int(data.get("window_minutes", 30)),
            window_export_mode=data.get("window_export_mode", "index"),
            normal_label=data.get("normal_label", "正常运行"),
            monitor_to_furnace=dict(
                data.get(
                    "monitor_to_furnace",
                    {"FQA70081": 1, "FQA70082": 2, "FQA70083": 3},
                )
            ),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def resolve_paths(self, base_dir: str | Path) -> "PipelineConfig":
        root = Path(base_dir)
        resolved_sources = []
        for source in self.data_sources:
            resolved_sources.append(
                DataSourceConfig(
                    name=source.name,
                    data_path=str((root / source.data_path).resolve()),
                    label_path=str((root / source.label_path).resolve()),
                    encoding=source.encoding,
                )
            )

        return PipelineConfig(
            point_table_path=str((root / self.point_table_path).resolve()),
            output_dir=str((root / self.output_dir).resolve()),
            data_sources=resolved_sources,
            split_ratios=self.split_ratios,
            split_strategy=self.split_strategy,
            train_sources=self.train_sources,
            val_sources=self.val_sources,
            test_sources=self.test_sources,
            mode=self.mode,
            transition_buffer_minutes=self.transition_buffer_minutes,
            transition_strategy=self.transition_strategy,
            transition_weight=self.transition_weight,
            drop_labels=self.drop_labels,
            exclude_features=self.exclude_features,
            outlier_quantiles=self.outlier_quantiles,
            impute_method=self.impute_method,
            diff_lags=self.diff_lags,
            rolling_windows_minutes=self.rolling_windows_minutes,
            build_window_samples=self.build_window_samples,
            window_minutes=self.window_minutes,
            window_export_mode=self.window_export_mode,
            normal_label=self.normal_label,
            monitor_to_furnace=self.monitor_to_furnace,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
