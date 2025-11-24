# 2025 玉山人工智慧公開挑戰賽 - 異常帳戶偵測

本專案為 2025 E.SUN AI Cup 初賽前標隊伍的提交程式碼。我們提出了一套基於機器學習的異常帳戶偵測模型，利用 Polars 進行高效特徵工程，並結合 LightGBM 與 PU Learning 策略進行精準預測。

## 目錄

- [專案簡介與特色](#專案簡介與特色)
- [環境需求](#環境需求)
- [專案結構](#專案結構)
- [資料準備](#資料準備)
- [安裝與執行方式](#安裝與執行方式)
- [復現性說明](#復現性說明)
- [隊伍資訊](#隊伍資訊)

---

## 專案簡介與特色

本解決方案的方法論聚焦於處理大規模交易資料的效率與模型穩定性，主要特色如下：

1.  **端到端流程 (End-to-End Pipeline)**
    整合資料載入、前處理、特徵工程至模型預測的完整自動化腳本，確保流程的一致性。

2.  **高效資料處理 (Efficient Processing)**
    使用 Polars LazyFrame 技術，大幅降低記憶體消耗並提升計算速度，適合處理數百萬筆交易資料。

3.  **高階特徵工程 (Advanced Feature Engineering)**
    - **Point-in-Time (PIT) 機制**：精確的時間切片統計，嚴格防止資料洩漏。
    - **行為穩定性特徵**：包含交易間隔變異係數 (CV)、赫芬達爾集中度指數 (HHI)、交易爆發度 (Burstiness)。
    - **網絡拓樸特徵**：計算帳戶的出入度 (Degree) 及其互動比例。
    - **相對特徵**：計算個體行為與群體平均的相對比率 (Relative-to-Population)。

4.  **穩健模型架構 (Robust Modeling)**
    採用 LightGBM 搭配 Stratified 5-Fold 交叉驗證，並針對資料不平衡問題引入 PU Learning (Positive-Unlabeled Learning) 的權重調整策略。

---

## 環境需求

本專案使用 Python 進行開發。

* **Python 版本**：Python 3.12.12
* **核心套件**：
    * lightgbm
    * polars
    * pandas
    * scikit-learn
    * numpy
    * pyarrow

詳細的版本依賴關係請參考 `requirements.txt`。

---

## 專案結構

```text
esun_aicup_2025/
├── data/                   # [需自行建立] 存放原始比賽資料集的目錄
│   ├── acct_alert.csv
│   ├── acct_predict.csv
│   └── acct_transaction.csv
├── Model/                  # 模型與特徵工程相關程式碼
│   ├── feature_engineering.py  # 核心特徵提取邏輯
│   └── README.md               # Model 模組說明
├── Preprocess/             # 資料前處理相關程式碼
│   ├── data_preprocess.py      # 資料清洗與 PIT 觀測點建立
│   └── README.md               # Preprocess 模組說明
├── main.py                 # 專案主程式入口
├── requirements.txt        # 套件依賴清單
└── README.md               # 本文件
```

---

## 資料準備

**重要提示**：由於比賽規範與檔案大小限制，原始資料集 (.csv) 並未包含在此儲存庫中。您必須手動配置資料。

請依照以下步驟準備：

1.  在專案**根目錄**下建立一個名為 `data` 的資料夾。
2.  將比賽提供的三個原始檔案放入該資料夾：
    * `acct_alert.csv`
    * `acct_predict.csv`
    * `acct_transaction.csv`

---

## 安裝與執行方式

本專案支援透過命令列介面 (CLI) 彈性執行。

### 1. 安裝相依套件

建議使用虛擬環境進行安裝以避免版本衝突。

```bash
 
pip install -r requirements.txt
 
```

### 2. 執行程式

**預設模式**
若資料已放置於預設的 `./data/` 資料夾中，請直接執行主程式：

```bash
 
python main.py
 
```

**指定資料路徑模式**
若您的資料位於其他路徑（例如 Google Drive 或外部硬碟），請使用 `--data_path` 參數指定：

```bash
 
python main.py --data_path "/path/to/your/dataset/"
 
```

### 3. 輸出結果

程式執行完畢後，將會在指定的資料路徑下生成預測結果檔案：

* **`submission.csv`**：包含 `acct` (帳戶ID) 與 `label` (預測標籤) 的最終提交檔案。

---

## 復現性說明

為確保評審委員能夠順利復現結果，請留意以下關於跨平台執行的數值差異說明：

* **主要實驗環境**：本專案之最佳成績是在 Google Colab (Linux / Intel Xeon CPU) 環境下產出。
* **差異原因**：由於 LightGBM 演算法與底層浮點數運算在不同作業系統（Linux vs. Windows）及 CPU 架構下存在微小的實作差異，預測機率可能會有些微浮動。
* **驗證結果**：經內部測試，本地端 (Windows) 執行結果與 Colab 原始結果的標籤一致性高達 **99.2% 以上** (僅約 0.8% 的邊緣案例發生翻轉）。F1-Score 表現保持穩定，證明了模型邏輯的正確性與可移植性。

---

## 隊伍資訊

* **隊伍編號**: TEAM_9597
* **隊員姓名**: 蔡智博
