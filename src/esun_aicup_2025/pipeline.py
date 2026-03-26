"""Project pipeline entry for training and prediction."""

from __future__ import annotations

from pathlib import Path

from esun_aicup_2025.features.feature_engineering import get_all_features_and_dataframes
from esun_aicup_2025.models.lgbm_trainer import train_and_predict


REQUIRED_FILES = ["acct_transaction.csv", "acct_alert.csv", "acct_predict.csv"]


def validate_data_dir(data_dir: str | Path) -> Path:
    """Validate dataset folder and required csv files."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    missing = [name for name in REQUIRED_FILES if not (data_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {data_path}: {missing}")

    return data_path


def run_pipeline(data_dir: str | Path, output_dir: str | Path, config: dict):
    """Execute end-to-end feature engineering + model training pipeline."""
    valid_data_dir = validate_data_dir(data_dir)
    df_train, df_predict, feature_cols = get_all_features_and_dataframes(str(valid_data_dir))
    return train_and_predict(
        df_final_pd=df_train,
        df_predict_pd=df_predict,
        feature_cols=feature_cols,
        output_path=output_dir,
        model_cfg=config["model"],
    )
