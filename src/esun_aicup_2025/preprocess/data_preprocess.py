"""Data preprocessing utilities for E.SUN AI Cup 2025."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import polars as pl

MAX_WINDOW = 30
OBS_OFFSETS = [1, 3, 7]


def load_and_define_global_data(data_path: str | Path) -> Tuple[pl.LazyFrame, pl.DataFrame, pl.DataFrame, int]:
    """Load transaction/alert/predict datasets and compute max transaction date."""
    data_dir = Path(data_path)
    acc_tran_path = data_dir / "acct_transaction.csv"
    acc_alert_path = data_dir / "acct_alert.csv"
    acc_predict_path = data_dir / "acct_predict.csv"

    scan_txn = pl.scan_csv(
        acc_tran_path,
        has_header=True,
        schema_overrides={
            "txn_date": pl.Int64,
            "txn_time": pl.Utf8,
            "txn_amt": pl.Float64,
            "is_self_txn": pl.Utf8,
            "currency_type": pl.Utf8,
            "to_acct_type": pl.Utf8,
            "from_acct_type": pl.Utf8,
        },
    )

    acc_alert = pl.read_csv(acc_alert_path)
    acc_predict = pl.read_csv(acc_predict_path)

    max_date_overall = scan_txn.select(pl.col("txn_date").max()).collect().item(0, 0)
    return scan_txn, acc_alert, acc_predict, max_date_overall


def create_observation_points(
    acc_alert: pl.DataFrame,
    acc_predict: pl.DataFrame,
    max_date: int,
    offsets: List[int],
) -> pl.DataFrame:
    """Build train/predict observation points with multi-PIT for positive samples."""
    df_obs_1 = acc_alert.with_columns(
        [
            pl.lit(1).cast(pl.Int64).alias("label"),
            pl.col("event_date").cast(pl.Int64),
        ]
    ).select("acct", "event_date", "label")

    df_obs_0 = (
        acc_predict.filter(pl.col("label") == 0)
        .select("acct", "label")
        .with_columns(
            pl.lit(max_date).cast(pl.Int64).alias("event_date"),
            pl.col("label").cast(pl.Int64),
        )
    )

    multi_obs = []
    for offset in offsets:
        temp = (
            df_obs_1.with_columns((pl.col("event_date") - offset).alias("event_date"))
            .filter(pl.col("event_date") > 0)
        )
        multi_obs.append(temp)

    df_obs_1_multi = pl.concat(multi_obs + [df_obs_1]).unique(subset=["acct", "event_date"])

    return pl.concat(
        [
            df_obs_1_multi.select(["acct", "event_date", "label"]),
            df_obs_0.select(["acct", "event_date", "label"]),
        ]
    )


def preprocess_transaction_data(scan_txn: pl.LazyFrame) -> pl.LazyFrame:
    """Apply base transaction cleaning and derive binary flags."""
    return (
        scan_txn.with_columns(
            [
                pl.col("is_self_txn").cast(pl.Utf8).str.to_uppercase(),
                pl.col("txn_time").str.slice(0, 2).cast(pl.Int64).alias("txn_hour"),
                pl.col("currency_type").cast(pl.Utf8),
                pl.col("to_acct_type").cast(pl.Utf8),
                pl.col("from_acct_type").cast(pl.Utf8),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("is_self_txn") == "Y").then(1).otherwise(0).alias("is_self_flag"),
                pl.when((pl.col("txn_hour") >= 23) | (pl.col("txn_hour") <= 6)).then(1).otherwise(0).alias("is_night"),
                pl.when(pl.col("currency_type") == "TWD").then(1).otherwise(0).alias("is_twd"),
                pl.when((pl.col("from_acct_type") == "01") & (pl.col("to_acct_type") == "02")).then(1).otherwise(0).alias("is_cross_bank_out"),
                pl.when((pl.col("from_acct_type") == "02") & (pl.col("to_acct_type") == "01")).then(1).otherwise(0).alias("is_cross_bank_in"),
            ]
        )
    )
