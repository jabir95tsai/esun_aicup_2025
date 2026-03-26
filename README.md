# 2025 玉山人工智慧公開挑戰賽 - 異常帳戶偵測

## 專案結構

```text
esun_aicup_2025/
├── data/
│   ├── raw/                    # 原始資料（放置競賽 CSV）
│   ├── interim/                # 中間產物（可選）
│   └── processed/              # 最終輸出（submission.csv）
├── src/
│   └── esun_aicup_2025/
│       ├── preprocess/
│       │   └── data_preprocess.py
│       ├── features/
│       │   └── feature_engineering.py
│       ├── models/
│       │   └── lgbm_trainer.py
│       ├── utils/
│       │   └── config.py
│       └── pipeline.py
├── scripts/
│   └── run_train.py
├── configs/
│   └── default.yaml
├── tests/
│   └── test_preprocess.py
├── requirements.txt
└── README.md
```

## 資料準備

競賽資料放在 `data/raw/`：

- `acct_transaction.csv`
- `acct_alert.csv`
- `acct_predict.csv`

## 安裝

```bash
pip install -r requirements.txt
```

## 執行訓練與輸出

```bash
PYTHONPATH=src python scripts/run_train.py \
  --data-dir data/raw \
  --output-dir data/processed \
  --config configs/default.yaml
```

執行完成後，`submission.csv` 會輸出到 `data/processed/`。

## 測試

```bash
PYTHONPATH=src pytest -q
```

## 模組說明

- `preprocess`：資料載入、欄位清理、觀測點建立。
- `features`：PIT、歷史、穩定性、圖特徵與衍生特徵。
- `models`：LightGBM 訓練、CV、Top-K 推論與 submission 產出。
- `pipeline`：整合流程並執行端到端工作流。
