from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import pandas as pd

from .cleaning import clean_wide_features
from .config import PipelineConfig
from .features import build_long_table, select_feature_columns
from .io import load_label_data, load_point_table, load_raw_data
from .labels import apply_labels_with_transition
from .output import save_json, save_npz, save_table
from .quality import summarize_dataframe
from .split import split_by_date_ratio, split_by_source_lists, split_full, zscore_by_furnace
from .temporal_features import add_temporal_features
from .windowing import build_window_samples


DEFAULT_LABEL_ORDER = ["停炉", "停炉降温", "停运", "烘炉", "启炉", "故障", "正常运行"]


def _file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_label_map(labels: list[str]) -> dict[str, int]:
    seen = set(labels)
    ordered = [x for x in DEFAULT_LABEL_ORDER if x in seen]
    tail = sorted([x for x in seen if x not in ordered])
    all_labels = ordered + tail
    return {name: idx for idx, name in enumerate(all_labels)}


def run_pipeline(cfg: PipelineConfig) -> dict:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    point_features = load_point_table(cfg.point_table_path)
    source_diagnostics: dict[str, dict] = {}
    all_frames: list[pd.DataFrame] = []
    common_features_per_source: dict[str, list[str]] = {}

    for source in cfg.data_sources:
        raw_df = load_raw_data(source)
        label_df = load_label_data(source)

        selected_features, feature_diag = select_feature_columns(
            raw_df,
            point_table_features=point_features,
            exclude_features=cfg.exclude_features,
        )
        cleaned_df, cleaning_diag = clean_wide_features(
            raw_df,
            feature_cols=selected_features,
            outlier_quantiles=cfg.outlier_quantiles,
            impute_method=cfg.impute_method,
        )

        long_df, base_features, long_diag = build_long_table(
            wide_df=cleaned_df,
            feature_columns=selected_features,
            source_name=source.name,
        )

        labeled_df, label_diag = apply_labels_with_transition(
            long_df=long_df,
            label_df=label_df,
            monitor_to_furnace=cfg.monitor_to_furnace,
            normal_label=cfg.normal_label,
            transition_buffer_minutes=cfg.transition_buffer_minutes,
        )
        featured_df, generated_features, temporal_diag = add_temporal_features(
            labeled_df,
            base_feature_cols=base_features,
            diff_lags=cfg.diff_lags,
            rolling_windows_minutes=cfg.rolling_windows_minutes,
        )

        source_diagnostics[source.name] = {
            "raw_rows": int(len(raw_df)),
            "long_rows": int(len(long_df)),
            "featured_rows": int(len(featured_df)),
            "selected_feature_count": int(len(selected_features)),
            "base_feature_count": int(len(base_features)),
            "generated_feature_count": int(len(generated_features)),
            "feature_diagnostics": feature_diag,
            "cleaning_diagnostics": cleaning_diag,
            "long_diagnostics": long_diag,
            "label_diagnostics": label_diag,
            "temporal_diagnostics": temporal_diag,
            "raw_time_start": str(raw_df["timestamp"].min()),
            "raw_time_end": str(raw_df["timestamp"].max()),
        }
        common_features_per_source[source.name] = [*base_features, *generated_features]
        all_frames.append(featured_df)

    combined = pd.concat(all_frames, axis=0, ignore_index=True).sort_values(
        ["timestamp", "furnace_id"]
    )
    combined = combined.reset_index(drop=True)

    if cfg.drop_labels:
        combined = combined.loc[~combined["label"].isin(cfg.drop_labels)].reset_index(drop=True)

    combined_with_transition = combined.copy()
    combined_with_transition["sample_weight"] = 1.0
    combined = combined_with_transition.copy()
    if cfg.transition_strategy == "drop":
        combined = combined.loc[~combined["is_transition"]].reset_index(drop=True)
    elif cfg.transition_strategy == "down_weight":
        combined.loc[combined["is_transition"], "sample_weight"] = cfg.transition_weight

    feature_sets = list(common_features_per_source.values())
    if not feature_sets:
        raise ValueError("No source data loaded.")
    common_features = sorted(set.intersection(*[set(x) for x in feature_sets]))
    if not common_features:
        raise ValueError("No common features across sources after processing.")

    keep_cols = [
        "timestamp",
        "source",
        "furnace_id",
        *common_features,
        "label",
        "sample_weight",
        "is_transition",
    ]
    combined = combined[keep_cols]

    if cfg.mode == "full":
        split_result = split_full(combined)
    elif cfg.split_strategy == "source_holdout":
        split_result = split_by_source_lists(
            combined,
            train_sources=cfg.train_sources,
            val_sources=cfg.val_sources,
            test_sources=cfg.test_sources,
            overlap_split_ratios=(cfg.split_ratios[1], cfg.split_ratios[2]),
        )
    else:
        split_result = split_by_date_ratio(combined, cfg.split_ratios)

    train_df, val_df, test_df, scaler_stats = zscore_by_furnace(
        train_df=split_result.train,
        val_df=split_result.val,
        test_df=split_result.test,
        feature_cols=common_features,
    )

    all_labels = train_df["label"].tolist()
    if not val_df.empty:
        all_labels += val_df["label"].tolist()
    if not test_df.empty:
        all_labels += test_df["label"].tolist()
    label_map = _build_label_map(all_labels)

    for frame in (train_df, val_df, test_df):
        if not frame.empty:
            frame["label_id"] = frame["label"].map(label_map).astype("Int64")
        else:
            frame["label_id"] = pd.Series(dtype="Int64")

    saved_paths = defaultdict(str)
    saved_paths["train"] = save_table(output_dir / "dataset_train", train_df)
    saved_paths["val"] = save_table(output_dir / "dataset_val", val_df)
    saved_paths["test"] = save_table(output_dir / "dataset_test", test_df)
    saved_paths["transition_mask"] = save_table(
        output_dir / "transition_mask",
        combined_with_transition[["timestamp", "source", "furnace_id", "is_transition"]],
    )
    if cfg.mode == "full":
        saved_paths["full"] = save_table(output_dir / "dataset_full", combined)

    window_diag = {}
    if cfg.build_window_samples:
        window_train = build_window_samples(
            train_df,
            feature_cols=common_features,
            window_minutes=cfg.window_minutes,
            export_mode=cfg.window_export_mode,
        )
        saved_paths["window_index_train"] = save_table(
            output_dir / "window_index_train", window_train.index_df
        )
        if window_train.dense_data is not None:
            saved_paths["window_dense_train"] = save_npz(
                output_dir / "window_dense_train", window_train.dense_data
            )

        window_diag["train"] = window_train.diagnostics

        if not val_df.empty:
            window_val = build_window_samples(
                val_df,
                feature_cols=common_features,
                window_minutes=cfg.window_minutes,
                export_mode=cfg.window_export_mode,
            )
            saved_paths["window_index_val"] = save_table(
                output_dir / "window_index_val", window_val.index_df
            )
            if window_val.dense_data is not None:
                saved_paths["window_dense_val"] = save_npz(
                    output_dir / "window_dense_val", window_val.dense_data
                )
            window_diag["val"] = window_val.diagnostics

        if not test_df.empty:
            window_test = build_window_samples(
                test_df,
                feature_cols=common_features,
                window_minutes=cfg.window_minutes,
                export_mode=cfg.window_export_mode,
            )
            saved_paths["window_index_test"] = save_table(
                output_dir / "window_index_test", window_test.index_df
            )
            if window_test.dense_data is not None:
                saved_paths["window_dense_test"] = save_npz(
                    output_dir / "window_dense_test", window_test.dense_data
                )
            window_diag["test"] = window_test.diagnostics

    quality_report = {
        "combined": summarize_dataframe(combined, "combined"),
        "combined_with_transition": summarize_dataframe(
            combined_with_transition, "combined_with_transition"
        ),
        "train": summarize_dataframe(train_df, "train"),
        "val": summarize_dataframe(val_df, "val"),
        "test": summarize_dataframe(test_df, "test"),
        "source_diagnostics": source_diagnostics,
        "window_diagnostics": window_diag,
    }
    save_json(output_dir / "quality_report.json", quality_report)
    save_json(output_dir / "feature_list.json", {"features": common_features})
    save_json(output_dir / "label_map.json", label_map)
    save_json(output_dir / "scaler_stats.json", scaler_stats)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input_files": {
            "point_table": {
                "path": cfg.point_table_path,
                "sha256": _file_sha256(cfg.point_table_path),
            },
            "sources": [
                {
                    "name": src.name,
                    "data_path": src.data_path,
                    "data_sha256": _file_sha256(src.data_path),
                    "label_path": src.label_path,
                    "label_sha256": _file_sha256(src.label_path),
                }
                for src in cfg.data_sources
            ],
        },
        "split_boundaries": split_result.date_boundaries,
        "saved_paths": dict(saved_paths),
        "row_counts": {
            "combined": int(len(combined)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }
    save_json(output_dir / "run_manifest.json", manifest)
    return manifest
