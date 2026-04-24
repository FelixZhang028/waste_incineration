from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("mtsc-preprocess")
    parser.add_argument(
        "--config",
        default="configs/preprocess.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--mode",
        choices=["split", "full"],
        default=None,
        help="Override mode in config.",
    )
    parser.add_argument(
        "--transition-strategy",
        choices=["drop", "down_weight", "keep"],
        default=None,
        help="Override transition strategy.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = PipelineConfig.from_json(args.config).resolve_paths(base_dir=Path("."))
    if args.mode:
        cfg.mode = args.mode
    if args.transition_strategy:
        cfg.transition_strategy = args.transition_strategy
    cfg.validate()

    manifest = run_pipeline(cfg)
    print("Preprocess completed.")
    print(f"train rows: {manifest['row_counts']['train']}")
    print(f"val rows: {manifest['row_counts']['val']}")
    print(f"test rows: {manifest['row_counts']['test']}")
    print(f"output dir: {cfg.output_dir}")


if __name__ == "__main__":
    main()
