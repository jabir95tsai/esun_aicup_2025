# Preprocess/data_preprocess.py
"""
AI CUP 2025 玉山挑戰賽 - 資料前處理模組

本模組負責資料的載入 (使用 Polars Lazy 模式)、全局參數定義，
以及訓練集與預測集的觀測點 (Point-in-Time, PIT) 建立。
同時也包含交易資料的基礎清理與衍生二元 (Binary) 特徵。
"""
import os
import polars as pl
import pandas as pd
import numpy as np
from typing import Tuple, List

# --- 0. 全局參數定義 ---
# 設定時間窗大小與多觀測點偏移量
MAX_WINDOW = 30
OBS_OFFSETS = [1, 3, 7] 

# --- 1. 載入資料 (Lazy) 與定義全局變數 ---

def load_and_define_global_data(data_path: str) -> Tuple[pl.LazyFrame, pl.DataFrame, pl.DataFrame, int]:
    """
    載入原始交易資料、警示帳戶名單及預測帳戶名單，並計算全局最大交易日期。

    使用 Polars 的 scan_csv 進行 Lazy 載入，以優化記憶體使用。

    Args:
        data_path (str): 資料檔案 (acct_transaction.csv, acct_alert.csv, acct_predict.csv) 所在的目錄路徑。

    Returns:
        Tuple[pl.LazyFrame, pl.DataFrame, pl.DataFrame, int]: 
            (scan_txn, acc_alert, acc_predict, MAX_DATE_OVERALL)
            - scan_txn: 交易資料的 LazyFrame。
            - acc_alert: 警示名單 DataFrame。
            - acc_predict: 預測名單 DataFrame。
            - MAX_DATE_OVERALL: 資料集中最大的 txn_date (作為 Label=0 的觀測日)。

    Raises:
        Exception: 無法讀取或計算 MAX_DATE_OVERALL 時拋出。
    """
    # 使用 os.path.join 確保路徑在不同作業系統下皆正確
    acc_tran_path = os.path.join(data_path, "acct_transaction.csv")
    acc_alert_path = os.path.join(data_path, "acct_alert.csv")
    acc_predict_path = os.path.join(data_path, "acct_predict.csv")

    # 1.1) 使用 scan_csv 保持 Lazy
    # 注意：需確保 schema 與資料集一致
    scan_txn = pl.scan_csv(acc_tran_path, has_header=True, schema_overrides={
        "txn_date": pl.Int64, "txn_time": pl.Utf8, "txn_amt": pl.Float64,
        "is_self_txn": pl.Utf8, "currency_type": pl.Utf8,
        "to_acct_type": pl.Utf8, "from_acct_type": pl.Utf8
    })
    
    # 讀取 CSV (警示與預測名單檔案較小，可直接讀入記憶體)
    acc_alert = pl.read_csv(acc_alert_path)
    acc_predict = pl.read_csv(acc_predict_path)

    # 1.2) 取得全資料的「最後一天」，作為 Label=0 帳戶的觀測日
    try:
        MAX_DATE_OVERALL = scan_txn.select(pl.col("txn_date").max()).collect().item(0, 0)
    except Exception as e:
        print(f"[ERROR] 無法從交易資料中讀取 txn_date。請檢查資料路徑或格式。錯誤訊息: {e}")
        # 若失敗則拋出異常，避免後續計算錯誤
        raise e
    
    return scan_txn, acc_alert, acc_predict, MAX_DATE_OVERALL


def create_observation_points(acc_alert: pl.DataFrame, acc_predict: pl.DataFrame, max_date: int, offsets: List[int]) -> pl.DataFrame:
    """
    建立訓練集與測試集的觀測點 (Observation Points) 總表。

    針對警示帳戶 (Label=1)，應用多個時間偏移量 (Multi-PIT) 產生多個觀測點；
    針對正常帳戶 (Label=0)，則統一使用資料集的最大日期作為觀測日。

    Args:
        acc_alert (pl.DataFrame): 警示帳戶名單 (包含 acct 和 event_date)。
        acc_predict (pl.DataFrame): 預測帳戶名單 (包含 acct 和 label)。
        max_date (int): 全資料集的最大交易日期。
        offsets (List[int]): 針對 Label=1 帳戶，往前取觀測點的日數偏移列表。

    Returns:
        pl.DataFrame: 包含 'acct', 'event_date', 'label' 的觀測總表 (df_observation_points)。
    """
    # 2.1) 警示帳戶 (Label=1)
    df_obs_1 = acc_alert.with_columns([
        pl.lit(1).cast(pl.Int64).alias("label"),
        pl.col("event_date").cast(pl.Int64)
    ]).select("acct", "event_date", "label")

    # 2.2) 正常帳戶 (Label=0)
    df_obs_0 = (
        acc_predict
        .filter(pl.col("label") == 0)
        .select("acct", "label")
        .with_columns(
            pl.lit(max_date).cast(pl.Int64).alias("event_date"),
            pl.col("label").cast(pl.Int64)
        )
    )

    # 2.3) 多觀測點訓練（Multi-PIT）
    multi_obs = []
    for offset in offsets:
        temp = (
            df_obs_1.with_columns((pl.col("event_date") - offset).alias("event_date"))
            .filter(pl.col("event_date") > 0)
        )
        multi_obs.append(temp)

    df_obs_1_multi = pl.concat(multi_obs + [df_obs_1]).unique(subset=["acct", "event_date"])

    # 2.4) 合併成一個完整的「觀測總表」
    df_observation_points = pl.concat([
        df_obs_1_multi.select(["acct", "event_date", "label"]),
        df_obs_0.select(["acct", "event_date", "label"])
    ])
    
    return df_observation_points


def preprocess_transaction_data(scan_txn: pl.LazyFrame) -> pl.LazyFrame:
    """
    對原始交易資料進行基礎清洗和衍生二元特徵 (Binary Features)。

    包括時間格式轉換、交易類型標準化、夜間交易標記、是否台幣交易，
    以及跨行轉入/轉出標記。

    Args:
        scan_txn (pl.LazyFrame): 原始交易資料的 LazyFrame。

    Returns:
        pl.LazyFrame: 包含基礎衍生特徵的 LazyFrame (scan_tx_processed)。
    """
    scan_tx_processed = (
        scan_txn
        .with_columns([
            pl.col("is_self_txn").cast(pl.Utf8).str.to_uppercase(),
            pl.col("txn_time").str.slice(0, 2).cast(pl.Int64).alias("txn_hour"),
            pl.col("currency_type").cast(pl.Utf8),
            pl.col("to_acct_type").cast(pl.Utf8),
            pl.col("from_acct_type").cast(pl.Utf8),
        ])
        .with_columns([
            pl.when(pl.col("is_self_txn") == "Y").then(1).otherwise(0).alias("is_self_flag"),
            # 夜間定義：23:00–06:59
            pl.when((pl.col("txn_hour") >= 23) | (pl.col("txn_hour") <= 6))
                .then(1).otherwise(0).alias("is_night"),
            # 是否為台幣交易
            pl.when(pl.col("currency_type") == "TWD")
                .then(1).otherwise(0).alias("is_twd"),
            # 是否跨行轉出 (to_acct_type='02' 且 from_acct_type='01')
            pl.when(
                (pl.col("from_acct_type") == "01") & (pl.col("to_acct_type") == "02")
            ).then(1).otherwise(0).alias("is_cross_bank_out"),
            # 是否跨行轉入 (from_acct_type='02' 且 to_acct_type='01')
            pl.when(
                (pl.col("from_acct_type") == "02") & (pl.col("to_acct_type") == "01")
            ).then(1).otherwise(0).alias("is_cross_bank_in"),
        ])
    )
    return scan_tx_processed

# --- 主執行區塊 (測試用) ---
if __name__ == '__main__':
    # 這部分僅用於單獨測試 data_preprocess.py
    # 預設測試資料路徑為 ./data/
    TEST_DATA_PATH = "./data/"
    
    if os.path.exists(TEST_DATA_PATH) and os.path.exists(os.path.join(TEST_DATA_PATH, "acct_transaction.csv")):
        print(f"[INFO] 檢測到測試資料，開始執行模組測試...")
        try:
            scan_txn, acc_alert, acc_predict, max_date = load_and_define_global_data(TEST_DATA_PATH)
            df_observation_points = create_observation_points(acc_alert, acc_predict, max_date, OBS_OFFSETS)
            scan_tx_processed = preprocess_transaction_data(scan_txn)
            
            print(f"[SUCCESS] 測試成功。")
            print(f"資料集 Max Date: {max_date}")
            print(f"觀測點總數 (Observation Points): {df_observation_points.height}")
        except Exception as e:
            print(f"[ERROR] 測試執行失敗: {e}")
    else:
        print("[INFO] 此模組為資料前處理函式庫。")
        print(f"[INFO] 若要測試，請建立 '{TEST_DATA_PATH}' 資料夾並放入 csv 檔案。")