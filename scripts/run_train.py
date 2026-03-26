"""CLI script to run training and generate submission."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E.SUN AI Cup 2025 training pipeline")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path containing csv files")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config path")
    return parser.parse_args()


def main() -> None:
    from esun_aicup_2025.pipeline import run_pipeline
    from esun_aicup_2025.utils.config import load_config

    args = parse_args()
    config = load_config(args.config)
    run_pipeline(data_dir=args.data_dir, output_dir=args.output_dir, config=config)


if __name__ == "__main__":
    main()
