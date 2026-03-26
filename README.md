# AI CUP 2025 玉山人工智慧公開挑戰賽－AI偵探出任務，精準揪出警示帳戶！



#### 一個參加 AI CUP 2025 玉山人工智慧公開挑戰賽－AI偵探出任務，精準揪出警示帳戶！ 獲得 34 / 790 名之成績的 Python 機器學習專案。



---

## 目錄

- [專案簡介](#專案簡介)
- [專案特色](#專案特色)
- [專案結構](#專案結構)
- [環境建置](#環境建置)
  - [1) 建立並啟用虛擬環境](#1-建立並啟用虛擬環境)
  - [2) 安裝相依套件](#2-安裝相依套件)
  - [3) Windows PowerShell 匯入路徑設定](#3-windows-powershell-匯入路徑設定)
- [使用方式](#使用方式)
  - [資料放置位置](#資料放置位置)
  - [訓練 / 產生預測](#訓練--產生預測)
- [設定檔](#設定檔)
- [相依套件](#相依套件)
- [測試](#測試)
- [可重現性](#可重現性)
- [未來擴充方向](#未來擴充方向)
- [授權](#授權)

---

## 專案簡介

本專案旨在支援競賽情境下的 **表格型異常偵測（tabular anomaly detection）** 完整機器學習流程，包含：

- 資料載入與前處理
- 特徵工程
- 模型訓練與預測
- 可配置化的實驗執行
- 以可重現性為導向的本地開發結構

整體程式架構採模組化設計，使前處理、特徵、模型與流程控制可以獨立演進，方便後續維護與擴充。

---

## 專案特色

- **模組化流程設計**：將前處理、特徵生成與模型訓練拆分管理
- **`src` 專案結構**：提升可維護性與匯入一致性
- **YAML 設定驅動**：方便管理不同實驗與執行參數
- **單一訓練入口**：統一執行流程，降低操作混亂
- **輕量測試支援**：可快速檢查核心工具與流程是否回歸異常
- **競賽導向組織方式**：適合反覆實驗、版本管理與團隊合作

---

## 專案結構

```text
esun_aicup_2025/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/            # 原始輸入資料
│   ├── interim/        # 可選的中間產物
│   └── processed/      # 輸出結果（如預測檔、submission）
├── scripts/
│   └── run_train.py    # 訓練入口
├── src/
│   └── esun_aicup_2025/
│       ├── preprocess/
│       ├── features/
│       ├── models/
│       ├── utils/
│       └── pipeline.py
├── tests/
├── requirements.txt
└── README.md
```

---

## 環境建置

### 1) 建立並啟用虛擬環境

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) 安裝相依套件

```bash
pip install -r requirements.txt
```

### 3) Windows PowerShell 匯入路徑設定

若你的 shell 或編輯器執行設定不會自動包含 `src`，請先執行：

```powershell
$env:PYTHONPATH="src"
```

這樣 Python 才能正確找到 `src/esun_aicup_2025` 下的套件。

---

## 使用方式

### 資料放置位置

請將競賽提供的輸入資料放在：

```text
data/raw/
```

實際檔名需依官方資料包內容為準。

### 訓練 / 產生預測

```bash
python scripts/run_train.py --data-dir data/raw --output-dir data/processed --config configs/default.yaml
```

常見輸出結果會寫入：

```text
data/processed/
```

在 Windows PowerShell 中，常見執行方式如下：

```powershell
$env:PYTHONPATH="src"
python scripts/run_train.py --data-dir data/raw --output-dir data/processed --config configs/default.yaml
```

---

## 設定檔

本專案透過 YAML 檔案管理設定，預設使用：

```text
configs/default.yaml
```

常見可由設定檔管理的內容包括：

- 模型超參數
- 交叉驗證或訓練設定
- 執行時參數
- 閾值或推論相關行為

建議做法：

- 將 `default.yaml` 作為穩定基準設定
- 為不同實驗建立獨立的 config 版本
- 比較結果時，同步記錄 config 與程式版本

---

## 相依套件

本專案目前主要使用以下套件：

- `numpy==2.0.2`
- `pandas==2.2.2`
- `scikit-learn==1.6.1`
- `lightgbm==4.6.0`
- `polars==1.31.0`
- `pyarrow==18.1.0`
- `PyYAML==6.0.2`
- `pytest==8.3.5`

安裝方式：

```bash
pip install -r requirements.txt
```

---

## 測試

執行測試：

```bash
python -m pytest -q
```

目前測試設計偏輕量，主要用於快速檢查核心流程與工具模組是否產生回歸問題。

---

## 可重現性

本專案遵循競賽型機器學習常見的可重現性原則：

- 在 `requirements.txt` 中固定套件版本
- 採用模組化 `src` 結構
- 以設定檔驅動執行流程
- 提供單一訓練入口指令

但仍需注意，對於梯度提升類模型流程，不同作業系統、Python 版本或硬體環境之間，可能出現些微數值差異，因此分數或預測機率不一定能完全逐位一致。

為了提升一致性，建議：

- 團隊成員統一 Python 版本
- 固定相依套件版本
- 將設定檔與程式版本一併管理
- 在實驗開始後避免任意修改原始資料

---

## 未來擴充方向

可進一步加強本專案的方向包括：

- 加入實驗追蹤機制（metrics、parameters、artifacts）
- 擴充特徵與模型模組的測試覆蓋率
- 加入 lint、型別檢查與 CI 流程
- 增加模型解釋性或報表產生工具
- 提供多組 config 預設以支援 ablation study

---

## 授權


在尚未加入授權檔前，原則上可視為保留所有權利。
```


