from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import PostRuleConfig


@dataclass
class Segment:
    start: int
    end: int
    state_id: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _to_segments(states: np.ndarray) -> list[Segment]:
    if states.size == 0:
        return []
    segments: list[Segment] = []
    start = 0
    current = int(states[0])
    for i in range(1, len(states)):
        value = int(states[i])
        if value != current:
            segments.append(Segment(start=start, end=i, state_id=current))
            start = i
            current = value
    segments.append(Segment(start=start, end=len(states), state_id=current))
    return segments


def _minutes_to_points(ts: pd.Series, minutes: int) -> int:
    d = ts.diff().dt.total_seconds().dropna()
    step = float(d.median()) if not d.empty else 60.0
    return max(1, int(round(minutes * 60.0 / step)))


def _build_allowed_transitions(cfg: PostRuleConfig) -> set[tuple[int, int]]:
    state_ids = set(cfg.states.to_dict().values())
    allowed = {(s, s) for s in state_ids}
    for pair in cfg.runtime.transition_whitelist:
        if len(pair) != 2:
            continue
        allowed.add((int(pair[0]), int(pair[1])))
    return allowed


def _min_duration_points(ts: pd.Series, cfg: PostRuleConfig) -> dict[int, int]:
    points: dict[int, int] = {}
    state_map = cfg.states.to_dict()
    for name, state_id in state_map.items():
        mins = int(cfg.runtime.min_duration_minutes.get(name, 0))
        points[int(state_id)] = _minutes_to_points(ts, mins) if mins > 0 else 1
    return points


def _propose_state(
    row: pd.Series,
    normal_abs_slope_ref: float,
    cfg: PostRuleConfig,
) -> int | None:
    state = cfg.states
    thr = cfg.thresholds

    upper = float(row["temp_upper"])
    middle = float(row["temp_middle"])
    o2 = float(row["o2"])
    pusher1 = float(row["pusher1"])
    pusher2 = float(row["pusher2"])
    slope = float(row["slope"])

    feed_zero = (pusher1 == 0.0) and (pusher2 == 0.0)
    high_temp_both = (upper > thr.temp_high) and (middle > thr.temp_high)
    low_temp_both = (upper < thr.temp_low) and (middle < thr.temp_low)

    stop_slope_band = (
        abs(normal_abs_slope_ref) * float(thr.stop_normal_like_factor)
        + float(thr.stop_normal_like_margin)
    )

    shutdown_cond = (
        low_temp_both
        and (o2 > thr.o2_high)
        and bool(row["run_zero_ok"])
        and bool(row["shutdown_zero_ok"])
    )
    if shutdown_cond:
        return state.shutdown

    stop_cond = high_temp_both and feed_zero and (abs(slope) <= stop_slope_band)
    if stop_cond:
        return state.stop

    cooldown_cond = feed_zero and (slope < thr.cooldown_slope_lt)
    if cooldown_cond:
        return state.cooldown

    bake_cond = feed_zero and (slope > thr.bake_slope_gt)
    if bake_cond:
        return state.bake

    startup_cond = high_temp_both
    if startup_cond:
        return state.startup

    return None


def _apply_min_duration(states: np.ndarray, min_points: dict[int, int]) -> np.ndarray:
    out = states.copy()
    for _ in range(2):
        segments = _to_segments(out)
        changed = False
        for i, seg in enumerate(segments):
            threshold = int(min_points.get(seg.state_id, 1))
            if seg.length >= threshold:
                continue

            prev_state = segments[i - 1].state_id if i > 0 else None
            next_state = segments[i + 1].state_id if i < len(segments) - 1 else None

            if prev_state is not None and next_state is not None and prev_state == next_state:
                replacement = prev_state
            elif prev_state is not None:
                replacement = prev_state
            elif next_state is not None:
                replacement = next_state
            else:
                replacement = seg.state_id

            out[seg.start : seg.end] = replacement
            changed = True

        if not changed:
            break
    return out


def _apply_startup_max(states: np.ndarray, timestamps: pd.Series, cfg: PostRuleConfig) -> np.ndarray:
    out = states.copy()
    max_points = _minutes_to_points(timestamps, int(cfg.runtime.startup_max_minutes))
    startup_id = int(cfg.states.startup)
    normal_id = int(cfg.states.normal)

    for seg in _to_segments(out):
        if seg.state_id != startup_id:
            continue
        if seg.length <= max_points:
            continue
        out[seg.start + max_points : seg.end] = normal_id
    return out


def apply_rules(pred_df: pd.DataFrame, cfg: PostRuleConfig) -> pd.Series:
    required = {
        "source",
        "furnace_id",
        "timestamp",
        "raw_pred_id",
        "raw_confidence",
        "temp_upper",
        "temp_middle",
        "o2",
        "pusher1",
        "pusher2",
        "slope",
        "run_zero_ok",
        "shutdown_zero_ok",
    }
    missing = sorted([c for c in required if c not in pred_df.columns])
    if missing:
        raise KeyError(f"Missing required columns for rule engine: {missing}")

    allowed_transitions = _build_allowed_transitions(cfg)
    hard_override = set(int(x) for x in cfg.runtime.hard_override_state_ids)
    conf_limit = float(cfg.runtime.confidence_max_for_override)

    out = pred_df.copy().sort_values(["source", "furnace_id", "timestamp"]).reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    rule_pred = out["raw_pred_id"].astype(int).to_numpy(copy=True)

    for (_, _), idx in out.groupby(["source", "furnace_id"], sort=False).groups.items():
        g = out.loc[idx].sort_values("timestamp").copy()
        indices = g.index.to_numpy()

        group_pred = g["raw_pred_id"].astype(int).to_numpy(copy=True)

        normal_mask = g["raw_pred_id"].astype(int) == int(cfg.states.normal)
        if normal_mask.any():
            normal_abs_ref = float(g.loc[normal_mask, "slope"].abs().median())
        else:
            normal_abs_ref = float(cfg.thresholds.stop_normal_like_fallback_abs)

        prev_state: int | None = None
        for local_i, row in enumerate(g.itertuples(index=False)):
            raw_state = int(getattr(row, "raw_pred_id"))
            raw_conf = float(getattr(row, "raw_confidence"))

            cand = _propose_state(
                pd.Series(row._asdict()),
                normal_abs_slope_ref=normal_abs_ref,
                cfg=cfg,
            )

            chosen = raw_state
            if cand is not None and (cand in hard_override or raw_conf <= conf_limit):
                chosen = int(cand)

            if prev_state is not None and (prev_state, chosen) not in allowed_transitions:
                if (prev_state, raw_state) in allowed_transitions:
                    chosen = raw_state
                else:
                    chosen = prev_state

            group_pred[local_i] = chosen
            prev_state = chosen

        group_pred = _apply_startup_max(group_pred, g["timestamp"], cfg)
        group_pred = _apply_min_duration(group_pred, _min_duration_points(g["timestamp"], cfg))
        rule_pred[indices] = group_pred

    return pd.Series(rule_pred, index=out.index, name="rule_pred_id")
