"""Feature engineering module for E.SUN AI Cup 2025."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import polars as pl

from esun_aicup_2025.preprocess.data_preprocess import (
    MAX_WINDOW,
    OBS_OFFSETS,
    create_observation_points,
    load_and_define_global_data,
    preprocess_transaction_data,
)

epsilon = 1e-6


def calculate_pit_features(
    scan_tx_processed: pl.LazyFrame, df_observation_points: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Calculate PIT outflow/inflow features."""
    tx_with_obs_out = scan_tx_processed.join(
        df_observation_points.lazy(), left_on="from_acct", right_on="acct", how="inner"
    )
    tx_filtered_out = tx_with_obs_out.with_columns(
        (pl.col("event_date") - pl.col("txn_date")).alias("days_before_alert")
    ).filter((pl.col("days_before_alert") > 0) & (pl.col("days_before_alert") <= MAX_WINDOW))

    feat_pit_out = (
        tx_filtered_out.group_by("from_acct", "event_date")
        .agg(
            [
                pl.len().alias("txn_cnt_out_30d"),
                pl.col("txn_amt").sum().alias("sum_out_30d"),
                pl.col("txn_amt").mean().alias("mean_out_30d"),
                pl.col("txn_amt").max().alias("max_out_30d"),
                pl.n_unique("to_acct").alias("unique_to_30d"),
                pl.col("is_self_flag").mean().alias("self_ratio_out_30d"),
                pl.col("is_night").mean().alias("night_ratio_out_30d"),
                pl.col("is_twd").mean().alias("twd_ratio_out_30d"),
                pl.col("is_cross_bank_out").mean().alias("cross_bank_ratio_30d"),
                pl.when(pl.col("days_before_alert") <= 7).then(1).otherwise(0).sum().alias("txn_cnt_out_7d"),
                pl.when(pl.col("days_before_alert") <= 7).then(pl.col("txn_amt")).otherwise(None).sum().alias("sum_out_7d"),
                pl.when(pl.col("days_before_alert") <= 7).then(pl.col("txn_amt")).otherwise(None).max().alias("max_out_7d"),
                pl.when(pl.col("days_before_alert") <= 7).then(pl.col("to_acct")).otherwise(None).n_unique().alias("unique_to_7d"),
                pl.when(pl.col("days_before_alert") <= 7).then(pl.col("is_self_flag")).otherwise(None).mean().alias("self_ratio_out_7d"),
                pl.when(pl.col("days_before_alert") <= 7).then(pl.col("is_night")).otherwise(None).mean().alias("night_ratio_out_7d"),
                pl.when(pl.col("days_before_alert") <= 7).then(pl.col("is_twd")).otherwise(None).mean().alias("twd_ratio_out_7d"),
                pl.when(pl.col("days_before_alert") <= 7).then(pl.col("is_cross_bank_out")).otherwise(None).mean().alias("cross_bank_ratio_7d"),
                pl.when(pl.col("days_before_alert") <= 1).then(1).otherwise(0).sum().alias("txn_cnt_out_1d"),
                pl.when(pl.col("days_before_alert") <= 1).then(pl.col("txn_amt")).otherwise(None).sum().alias("sum_out_1d"),
            ]
        )
        .with_columns(pl.col("from_acct").alias("acct"))
        .select(pl.all().exclude("from_acct"))
        .collect()
    )

    tx_with_obs_in = scan_tx_processed.join(
        df_observation_points.lazy(), left_on="to_acct", right_on="acct", how="inner"
    )
    tx_filtered_in = tx_with_obs_in.with_columns(
        (pl.col("event_date") - pl.col("txn_date")).alias("days_before_alert")
    ).filter((pl.col("days_before_alert") > 0) & (pl.col("days_before_alert") <= MAX_WINDOW))

    feat_pit_in = (
        tx_filtered_in.group_by("to_acct", "event_date")
        .agg(
            [
                pl.len().alias("txn_cnt_in_30d"),
                pl.col("txn_amt").sum().alias("sum_in_30d"),
                pl.n_unique("from_acct").alias("unique_from_30d"),
                pl.col("is_night").mean().alias("night_ratio_in_30d"),
                pl.col("is_twd").mean().alias("twd_ratio_in_30d"),
                pl.col("is_cross_bank_in").mean().alias("cross_bank_ratio_in_30d"),
                pl.when(pl.col("days_before_alert") <= 7).then(1).otherwise(0).sum().alias("txn_cnt_in_7d"),
                pl.when(pl.col("days_before_alert") <= 7).then(pl.col("txn_amt")).otherwise(None).sum().alias("sum_in_7d"),
                pl.when(pl.col("days_before_alert") <= 7).then(pl.col("from_acct")).otherwise(None).n_unique().alias("unique_from_7d"),
                pl.when(pl.col("days_before_alert") <= 7).then(pl.col("is_night")).otherwise(None).mean().alias("night_ratio_in_7d"),
                pl.when(pl.col("days_before_alert") <= 7).then(pl.col("is_twd")).otherwise(None).mean().alias("twd_ratio_in_7d"),
                pl.when(pl.col("days_before_alert") <= 7).then(pl.col("is_cross_bank_in")).otherwise(None).mean().alias("cross_bank_ratio_in_7d"),
                pl.when(pl.col("days_before_alert") <= 1).then(1).otherwise(0).sum().alias("txn_cnt_in_1d"),
                pl.when(pl.col("days_before_alert") <= 1).then(pl.col("txn_amt")).otherwise(None).sum().alias("sum_in_1d"),
            ]
        )
        .with_columns(pl.col("to_acct").alias("acct"))
        .select(pl.all().exclude("to_acct"))
        .collect()
    )

    return feat_pit_out, feat_pit_in


def calculate_history_features(scan_tx_processed: pl.LazyFrame, df_observation_points: pl.DataFrame) -> pl.DataFrame:
    """Calculate account history features."""
    return (
        scan_tx_processed.join(
            df_observation_points.lazy(), left_on="from_acct", right_on="acct", how="inner"
        )
        .filter(pl.col("txn_date") < pl.col("event_date"))
        .group_by("from_acct", "event_date")
        .agg(
            [
                pl.col("txn_date").n_unique().alias("history_active_days"),
                (pl.col("event_date") - pl.col("txn_date").max()).fill_null(0).cast(pl.Float64).alias("days_since_last_txn_out"),
                (pl.col("event_date") - pl.col("txn_date").min()).fill_null(0).cast(pl.Float64).alias("days_from_first_txn"),
                pl.col("txn_date").min().alias("first_txn_date_out"),
            ]
        )
        .with_columns(pl.col("from_acct").alias("acct"))
        .select(pl.all().exclude("from_acct"))
        .collect()
    )


def calculate_stability_and_graph_features(
    scan_tx_processed: pl.LazyFrame, df_observation_points: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Calculate stability and graph features."""
    tx_out_30d = (
        scan_tx_processed.join(
            df_observation_points.lazy(), left_on="from_acct", right_on="acct", how="inner"
        )
        .with_columns(
            [
                (pl.col("event_date") - pl.col("txn_date")).alias("days_before_alert"),
                (pl.col("txn_date") * 24 + pl.col("txn_hour")).cast(pl.Int64).alias("time_idx"),
            ]
        )
        .filter((pl.col("days_before_alert") > 0) & (pl.col("days_before_alert") <= 30))
    )

    stab_interval = (
        tx_out_30d.sort(["from_acct", "time_idx"])
        .with_columns(pl.col("time_idx").diff().over("from_acct", "event_date").alias("interval_hours"))
        .group_by("from_acct", "event_date")
        .agg(
            [
                pl.col("interval_hours").mean().fill_null(0.0).alias("avg_interval_out_30d"),
                (pl.col("interval_hours").std(ddof=1) / (pl.col("interval_hours").mean() + epsilon)).fill_null(0.0).alias("cv_interval_out_30d"),
            ]
        )
    )

    stab_conc = (
        tx_out_30d.group_by(["from_acct", "event_date", "to_acct"])
        .agg(pl.len().alias("cnt_to"))
        .group_by("from_acct", "event_date")
        .agg(
            [
                (pl.col("cnt_to") / (pl.col("cnt_to").sum() + epsilon)).pow(2).sum().alias("hhi_out_30d"),
                (pl.col("cnt_to").max() / (pl.col("cnt_to").sum() + epsilon)).alias("max_share_out_30d"),
            ]
        )
        .select(["from_acct", "event_date", "hhi_out_30d", "max_share_out_30d"])
    )

    stab_burst = (
        tx_out_30d.group_by(["from_acct", "event_date", "txn_date"])
        .agg(pl.len().alias("cnt_per_day"))
        .group_by("from_acct", "event_date")
        .agg([pl.col("cnt_per_day").max().alias("max_day_cnt"), pl.col("cnt_per_day").mean().alias("avg_day_cnt")])
        .with_columns((pl.col("max_day_cnt") / (pl.col("avg_day_cnt") + epsilon)).alias("burstiness_out_30d"))
        .select(["from_acct", "event_date", "burstiness_out_30d"])
    )

    feat_stability = (
        stab_interval.join(stab_conc, on=["from_acct", "event_date"], how="left")
        .join(stab_burst, on=["from_acct", "event_date"], how="left")
        .with_columns(pl.col("from_acct").alias("acct"))
        .select(pl.all().exclude("from_acct"))
        .fill_null(0.0)
        .with_columns(pl.col("event_date").cast(pl.Int64))
        .collect()
    )

    graph_feat = (
        scan_tx_processed.group_by("from_acct")
        .agg(pl.n_unique("to_acct").alias("out_degree"))
        .join(
            scan_tx_processed.group_by("to_acct").agg(pl.n_unique("from_acct").alias("in_degree")),
            left_on="from_acct",
            right_on="to_acct",
            how="outer",
        )
        .select(
            [
                pl.col("from_acct").fill_null(pl.col("to_acct")).alias("acct"),
                pl.col("out_degree").fill_null(0),
                pl.col("in_degree").fill_null(0),
            ]
        )
        .with_columns(((pl.col("out_degree") + 1) / (pl.col("in_degree") + 1)).alias("degree_ratio"))
        .collect()
    )

    return feat_stability, graph_feat


def get_all_features_and_dataframes(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Build train/predict feature matrices and return aligned feature columns."""
    scan_txn, acc_alert, acc_predict, max_date = load_and_define_global_data(data_path)
    scan_tx_processed = preprocess_transaction_data(scan_txn)

    df_observation_points_train = create_observation_points(acc_alert, acc_predict, max_date, OBS_OFFSETS)
    df_observation_points_predict = (
        acc_predict.select("acct")
        .unique(subset=["acct"])
        .with_columns([pl.lit(max_date).cast(pl.Int64).alias("event_date"), pl.lit(0).alias("label")])
        .select(["acct", "event_date", "label"])
    )

    feat_pit_out_tr, feat_pit_in_tr = calculate_pit_features(scan_tx_processed, df_observation_points_train)
    feat_hist_tr = calculate_history_features(scan_tx_processed, df_observation_points_train)
    feat_stab_tr, graph_feat = calculate_stability_and_graph_features(scan_tx_processed, df_observation_points_train)

    df_final_train = (
        df_observation_points_train.join(feat_pit_out_tr, on=["acct", "event_date"], how="left")
        .join(feat_pit_in_tr, on=["acct", "event_date"], how="left")
        .join(feat_hist_tr, on=["acct", "event_date"], how="left")
        .join(feat_stab_tr, on=["acct", "event_date"], how="left")
        .join(graph_feat, on="acct", how="left")
        .fill_null(0)
    )

    feat_pit_out_pr, feat_pit_in_pr = calculate_pit_features(scan_tx_processed, df_observation_points_predict)
    feat_hist_pr = calculate_history_features(scan_tx_processed, df_observation_points_predict)
    feat_stab_pr, _ = calculate_stability_and_graph_features(scan_tx_processed, df_observation_points_predict)

    df_final_predict = (
        df_observation_points_predict.join(feat_pit_out_pr, on=["acct", "event_date"], how="left")
        .join(feat_pit_in_pr, on=["acct", "event_date"], how="left")
        .join(feat_hist_pr, on=["acct", "event_date"], how="left")
        .join(feat_stab_pr, on=["acct", "event_date"], how="left")
        .join(graph_feat, on="acct", how="left")
        .fill_null(0)
    )

    derived_exprs = [
        (pl.col("sum_out_7d") / (pl.col("sum_out_30d") + epsilon)).alias("ratio_sum_out_7d_vs_30d"),
        (pl.col("txn_cnt_out_7d") / (pl.col("txn_cnt_out_30d") + epsilon)).alias("ratio_cnt_out_7d_vs_30d"),
        (pl.col("sum_in_7d") / (pl.col("sum_in_30d") + epsilon)).alias("ratio_sum_in_7d_vs_30d"),
        (pl.col("unique_from_7d") / (pl.col("unique_from_30d") + epsilon)).alias("ratio_unique_from_7d_vs_30d"),
        (pl.col("sum_in_7d") - pl.col("sum_out_7d")).alias("net_flow_7d"),
        (pl.col("sum_in_30d") - pl.col("sum_out_30d")).alias("net_flow_30d"),
        ((pl.col("sum_out_7d") - (pl.col("sum_out_30d") / 4)) / (pl.col("sum_out_30d") / 4 + epsilon)).alias("growth_sum_out_7d"),
        ((pl.col("txn_cnt_out_7d") - (pl.col("txn_cnt_out_30d") / 4)) / (pl.col("txn_cnt_out_30d") / 4 + epsilon)).alias("growth_cnt_out_7d"),
        ((pl.col("sum_in_7d") - (pl.col("sum_in_30d") / 4)) / (pl.col("sum_in_30d") / 4 + epsilon)).alias("growth_sum_in_7d"),
        (pl.col("sum_in_7d") / (pl.col("sum_in_7d") + pl.col("sum_out_7d") + epsilon)).alias("inflow_ratio_7d"),
        (pl.col("sum_out_7d") / (pl.col("sum_in_7d") + pl.col("sum_out_7d") + epsilon)).alias("outflow_ratio_7d"),
        (pl.col("unique_to_7d") / (pl.col("txn_cnt_out_7d") + epsilon)).alias("unique_ratio_out_7d"),
        (pl.col("unique_from_7d") / (pl.col("txn_cnt_in_7d") + epsilon)).alias("unique_ratio_in_7d"),
        (pl.col("self_ratio_out_7d") * pl.col("night_ratio_out_7d")).alias("night_self_interact"),
    ]

    df_final_train = df_final_train.with_columns(derived_exprs)
    df_final_predict = df_final_predict.with_columns(derived_exprs)

    global_means = df_final_train.select(
        [
            pl.col("sum_out_30d").mean().alias("global_sum_out_30d_mean"),
            pl.col("txn_cnt_out_30d").mean().alias("global_cnt_out_30d_mean"),
            pl.col("sum_in_30d").mean().alias("global_sum_in_30d_mean"),
            pl.col("txn_cnt_in_30d").mean().alias("global_cnt_in_30d_mean"),
        ]
    ).to_dict(as_series=False)

    rel_exprs = [
        (pl.col("sum_out_30d") / (global_means["global_sum_out_30d_mean"][0] + epsilon)).alias("rel_sum_out_vs_global"),
        (pl.col("txn_cnt_out_30d") / (global_means["global_cnt_out_30d_mean"][0] + epsilon)).alias("rel_cnt_out_vs_global"),
        (pl.col("sum_in_30d") / (global_means["global_sum_in_30d_mean"][0] + epsilon)).alias("rel_sum_in_vs_global"),
        (pl.col("txn_cnt_in_30d") / (global_means["global_cnt_in_30d_mean"][0] + epsilon)).alias("rel_cnt_in_vs_global"),
    ]

    df_final_train = df_final_train.with_columns(rel_exprs)
    df_final_predict = df_final_predict.with_columns(rel_exprs)

    exclude_cols = ["acct", "event_date", "label", "first_txn_date_out"]
    feature_cols = [
        col for col in df_final_train.columns if col not in exclude_cols and col in df_final_predict.columns
    ]

    return df_final_train.to_pandas(), df_final_predict.to_pandas(), feature_cols
