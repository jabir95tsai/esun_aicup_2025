# main.py
"""
2025 玉山人工智慧公開挑戰賽：初賽資格審查主程式

本模組負責整合資料前處理、特徵工程、模型訓練與預測流程。
它支援透過命令列參數指定資料路徑，確保程式碼的可攜性與可復現性。

執行方式範例:
    python main.py --data_path ./data/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# --- 導入自訂模組 ---
# 確保 Python 能找到 Preprocess 和 Model 資料夾
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from Model.feature_engineering import get_all_features_and_dataframes
except ImportError as e:
    print("[ERROR] 無法匯入模組。請確保 'Model/feature_engineering.py' 存在且路徑正確。")
    print(f"詳細錯誤: {e}")
    sys.exit(1)


def run_model_training_and_prediction(df_final_pd, df_predict_pd, feature_cols, output_path):
    """
    執行 LightGBM 模型訓練、閾值最佳化與最終預測。
    邏輯完全復刻 aicup_2025.py 的 '最終版' 區塊。
    """
    print(f"\n--- 開始模型訓練 (特徵數: {len(feature_cols)}) ---")
    
    # 強制排序以確保跨平台復現性 (Polars 多執行緒可能會打亂順序)
    df_final_pd = df_final_pd.sort_values(by='acct').reset_index(drop=True)
    df_predict_pd = df_predict_pd.sort_values(by='acct').reset_index(drop=True)

    # 準備訓練數據
    X = df_final_pd[feature_cols].copy()
    y = df_final_pd['label'].astype(int).copy()
    X_test = df_predict_pd[feature_cols].copy()

    # 型別安全檢查與補值
    for df_ in (X, X_test):
        for c in df_.columns:
            if df_[c].dtype == object or str(df_[c].dtype) == 'category':
                df_[c] = pd.to_numeric(df_[c], errors='coerce')
        df_.fillna(0, inplace=True)

    # 1) K-Fold 驗證：找最佳閾值與合理的樹數
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    th_grid = np.linspace(0.02, 0.98, 49)
    best_thresholds, best_iters = [], []
    oof_pred = np.zeros(len(X), dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # 類別不平衡權重（PU 假設 alpha=0.9）
        alpha = 0.9
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        scale_pos_weight = alpha * (neg / max(pos, 1))

        # --- LGBMClassifier 實例化 (完全對應原始碼參數) ---
        clf = LGBMClassifier(
            objective='binary', 
            metric='binary_logloss', 
            n_estimators=2000,
            learning_rate=0.03, 
            num_leaves=63, 
            subsample=0.9,          
            colsample_bytree=0.9,   
            reg_alpha=0.0,          
            reg_lambda=0.0,           
            random_state=42, 
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight, 
            verbose=-1
        )

        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='binary_logloss',
                callbacks=[early_stopping(stopping_rounds=150, verbose=False)])

        best_iters.append(clf.best_iteration_)
        val_pred_prob = clf.predict_proba(X_val, num_iteration=clf.best_iteration_)[:, 1]

        # 找此折最佳 F1 閾值
        f1_list = [f1_score(y_val, (val_pred_prob > th).astype(int)) for th in th_grid]
        th_star = float(th_grid[np.argmax(f1_list)])
        best_thresholds.append(th_star)
        oof_pred[val_idx] = val_pred_prob
        print(f"Fold {fold}: Best F1={max(f1_list):.4f} at Th={th_star:.3f}")

    # 計算全域最佳閾值
    f1_grid = [f1_score(y, (oof_pred > th).astype(int)) for th in th_grid]
    th_global = float(th_grid[np.argmax(f1_grid)])
    print(f"\nOOF 全域最佳 F1: {max(f1_grid):.4f} (Threshold: {th_global:.3f})")

    # 2) 以平均最佳迭代數重訓最終模型
    final_n_estimators = max(100, int(np.mean(best_iters)))
    pos_all = (y == 1).sum()
    neg_all = (y == 0).sum()
    scale_pos_weight_all = 0.9 * (neg_all / max(pos_all, 1))

    print(f"正在以 n_estimators={final_n_estimators} 重訓最終模型...")
    final_clf = LGBMClassifier(
        objective='binary', 
        metric='binary_logloss', 
        n_estimators=final_n_estimators,
        learning_rate=0.03, 
        num_leaves=63, 
        subsample=0.9,          
        colsample_bytree=0.9,   
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42, 
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight_all, 
        verbose=-1
    )
    final_clf.fit(X, y)
    test_prob = final_clf.predict_proba(X_test)[:, 1]

    # 3) 選擇 Top-K 策略輸出
    TOPK_PCT = 0.05
    k = max(1, int(len(test_prob) * TOPK_PCT))
    topk_threshold = np.partition(test_prob, -k)[-k]
    final_pred = (test_prob >= topk_threshold).astype(int)

    # 4) 輸出 submission.csv
    df_submission = pd.DataFrame({
        'acct': df_predict_pd['acct'],
        'label': final_pred
    })
    
    # 強制排序輸出，方便比對
    df_submission = df_submission.sort_values(by='acct')
    
    out_file = os.path.join(output_path, "submission.csv")
    df_submission.to_csv(out_file, index=False)

    print(f"\n[INFO] 預測完成！結果已輸出至：{out_file}")
    return df_submission


def main():
    parser = argparse.ArgumentParser(description='2025 E.SUN AI Challenge - Training & Prediction')
    parser.add_argument('--data_path', type=str, default='./data/', 
                        help='Path to the directory containing dataset csv files.')
    args = parser.parse_args()
    DATA_PATH = os.path.join(args.data_path, '')
    
    print("="*60)
    print(f"2025 AI CUP 挑戰賽程式啟動")
    print(f"資料讀取路徑: {DATA_PATH}")
    print("="*60)
    
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] 錯誤: 找不到資料夾 '{DATA_PATH}'")
        sys.exit(1)
        
    required_files = ["acct_transaction.csv", "acct_alert.csv", "acct_predict.csv"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(DATA_PATH, f))]
    
    if missing_files:
        print(f"[ERROR] 錯誤: 在路徑 '{DATA_PATH}' 中找不到以下必要檔案: {missing_files}")
        sys.exit(1)

    print("\n[Step 1/2] 正在執行特徵工程 (Feature Engineering)...")
    try:
        df_final_pd, df_predict_pd, feature_cols = get_all_features_and_dataframes(DATA_PATH)
        print(f"[INFO] 特徵工程完成。訓練集形狀: {df_final_pd.shape}, 預測集形狀: {df_predict_pd.shape}")
    except Exception as e:
        print(f"[ERROR] 特徵工程執行失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[Step 2/2] 正在執行模型訓練與預測 (Model Training)...")
    try:
        run_model_training_and_prediction(df_final_pd, df_predict_pd, feature_cols, DATA_PATH)
    except Exception as e:
        print(f"[ERROR] 模型訓練失敗: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()