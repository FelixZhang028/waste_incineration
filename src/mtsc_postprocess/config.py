from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any


@dataclass
class FeatureMapping:
    temperature_upper_col: str = "炉膛上部温度"
    temperature_middle_col: str = "炉膛中部温度"
    oxygen_col: str = "炉氧量"
    pusher1_col: str = "炉推料器一"
    pusher2_col: str = "炉推料器二"
    run_zero_cols: list[str] = field(
        default_factory=lambda: ["辅燃运行信号", "炉引风机频率", "炉鼓风机频率"]
    )
    shutdown_zero_cols: list[str] = field(
        default_factory=lambda: ["炉活性炭喷射量", "炉小苏打喷射量", "氨水流量"]
    )


@dataclass
class ThresholdConfig:
    temp_low: float = 400.0
    temp_high: float = 850.0
    o2_high: float = 19.0
    cooldown_slope_lt: float = 0.0
    bake_slope_gt: float = 0.0
    stop_normal_like_factor: float = 2.0
    stop_normal_like_margin: float = 0.2
    stop_normal_like_fallback_abs: float = 1.0


@dataclass
class TimingConfig:
    slope_window_minutes: int = 10


@dataclass
class StateIdConfig:
    stop: int = 0
    cooldown: int = 1
    shutdown: int = 2
    bake: int = 3
    startup: int = 4
    normal: int = 5

    def to_dict(self) -> dict[str, int]:
        return {
            "stop": int(self.stop),
            "cooldown": int(self.cooldown),
            "shutdown": int(self.shutdown),
            "bake": int(self.bake),
            "startup": int(self.startup),
            "normal": int(self.normal),
        }


@dataclass
class RuntimeRuleConfig:
    confidence_max_for_override: float = 0.8
    hard_override_state_ids: list[int] = field(default_factory=lambda: [2])
    startup_max_minutes: int = 240
    min_duration_minutes: dict[str, int] = field(
        default_factory=lambda: {
            "stop": 6,
            "cooldown": 6,
            "shutdown": 6,
            "bake": 6,
            "startup": 6,
            "normal": 6,
        }
    )
    transition_whitelist: list[list[int]] = field(
        default_factory=lambda: [
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 3],
            [2, 3],
            [2, 5],
            [3, 4],
            [3, 5],
            [4, 5],
            [5, 0],
            [5, 2],
        ]
    )


@dataclass
class PostRuleConfig:
    enabled: bool = True
    inverse_scale: bool = True
    feature_mapping: FeatureMapping = field(default_factory=FeatureMapping)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    states: StateIdConfig = field(default_factory=StateIdConfig)
    runtime: RuntimeRuleConfig = field(default_factory=RuntimeRuleConfig)

    def validate(self) -> None:
        if self.timing.slope_window_minutes <= 0:
            raise ValueError("timing.slope_window_minutes must be positive")
        if self.runtime.confidence_max_for_override < 0 or self.runtime.confidence_max_for_override > 1:
            raise ValueError("runtime.confidence_max_for_override must be in [0, 1]")
        if self.runtime.startup_max_minutes <= 0:
            raise ValueError("runtime.startup_max_minutes must be positive")
        state_ids = set(self.states.to_dict().values())
        if len(state_ids) != 6:
            raise ValueError("states contains duplicate ids")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PostRuleConfig":
        cfg = cls(
            enabled=bool(payload.get("enabled", True)),
            inverse_scale=bool(payload.get("inverse_scale", True)),
            feature_mapping=FeatureMapping(**payload.get("feature_mapping", {})),
            thresholds=ThresholdConfig(**payload.get("thresholds", {})),
            timing=TimingConfig(**payload.get("timing", {})),
            states=StateIdConfig(**payload.get("states", {})),
            runtime=RuntimeRuleConfig(**payload.get("runtime", {})),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "PostRuleConfig":
        with Path(path).open("r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        return cls.from_dict(payload)
