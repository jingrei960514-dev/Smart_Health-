# 🏋️‍♂️ Smart Health — 個人化健身與營養決策支援系統

基於機器學習的個人化健身分析系統。輸入使用者的生活型態資料（步數、睡眠、心率、運動習慣），透過 KMeans 分群與 Random Forest 分類器分析活動等級，並推薦個人化課表、飲食建議，以及 Gemini LLM 生成的專屬菜單。

---

## 📁 專案結構

```
SMART_HEALTH/
├── data/                              # ⚠️ 不在 repo 內，請見下方下載說明
├── fitness_pipeline/                  # 主套件
│   ├── pipeline.py                    # 終端機執行入口
│   ├── models/
│   │   ├── schemas.py                 # 所有 dataclass 定義
│   │   └── fitness_models.py          # KMeans、RF 分類器、熱量迴歸
│   ├── recommenders/
│   │   ├── goal_engine.py             # 目標天數估算、what-if 分析
│   │   ├── food_recommender.py        # Cosine Similarity 食物推薦
│   │   ├── program_recommender.py     # 課表評分與推薦
│   │   └── llm_meal_plan.py           # Gemini LLM 菜單生成
│   ├── report/
│   │   └── renderer.py                # 終端機報告輸出
│   ├── utils/
│   │   └── helpers.py                 # 顯示工具、中文本地化
│   └── tests/
│       └── test_fixes.py              # 單元測試
├── app.py                             # Streamlit 網頁介面
├── .env                               # API Key 設定（不在 repo 內）
├── 機器學習與實驗說明.md                # 機器學習調整過程與模型效能評估
├── 主程式pipeline說明.md               # 使用者流程與模型間互動關係(以尚未模組化程式碼版本為例)
├── dataset應用說明.md                  # 資料集基本資訊與清理
──  fitness_personalized_pipeline_v2.py # 尚未模組化之程式碼
├── .gitignore
├── requirements.txt
└── README.md

```

---

## ⚡ 快速開始

### 1. Clone 此 repo

```bash
git clone https://github.com/jingrei960514-dev/Smart_Health-.git
cd Smart_Health-
```

### 2. 建立虛擬環境

**Windows（PowerShell）**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows（Git Bash）**
```bash
python -m venv venv
source venv/Scripts/activate
```

**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. 安裝依賴套件

```bash
pip install -r requirements.txt
```

若要啟用 LLM 菜單功能，另外安裝：

```bash
pip install google-genai
```

### 4. 下載資料集

資料集因檔案過大未上傳至 GitHub，請從以下連結下載：

👉 👉 **[點此下載 data 資料夾](https://drive.google.com/drive/folders/14rte5yBcbsJwVmXuCeR0EV-sdYZ0fzn1?usp=sharing)**

方式 A：下載到的是 data.zip（推薦）
直接解壓縮，將 data/ 資料夾放到專案根目錄即可。
SMART_HEALTH/
└── data/
    ├── cleaned_fitness_data_v2.csv
    ├── cleaned_nutrients_analysis.csv
    ├── fitness_and_workout_dataset.csv
    ├── program_profiles_zh.csv
    ├── program_profiles.csv
    ├── program_summary.csv
    └── programs_detailed_boostcamp_kaggle.csv

方式 B：下載到的是 .xlsx 檔案
Google Drive 有時會自動將 CSV 轉為 xlsx 格式。請手動轉換成CSV檔，再將其放置於專案根目錄。
```
SMART_HEALTH/
├── data/
│   ├── cleaned_fitness_data_v2.csv
│   ├── cleaned_nutrients_analysis.csv
│   ├── fitness_and_workout_dataset.csv
│   ├── program_profiles_zh.csv
│   ├── program_profiles.csv
│   ├── program_summary.csv
│   └── programs_detailed_boostcamp_kaggle.csv
```

### 5. 設定環境變數

在專案根目錄建立 `.env` 檔案：

```
GEMINI_API_KEY=你的_Gemini_API_Key
```

> 若未設定，LLM 菜單功能會自動略過，其他功能不受影響。

---

## 🚀 執行方式

### 方式一：Streamlit 網頁介面

```bash
streamlit run app.py
```

瀏覽器會自動開啟 `http://localhost:8501`，在側邊欄填入個人資料後按下「產生個人化分析報告」即可。

### 方式二：終端機執行（使用預設參數）

```bash
python -m fitness_pipeline.pipeline
```

### 方式二：終端機執行（自訂參數）

```bash
python -m fitness_pipeline.pipeline \
  --age 28 \
  --gender Male \
  --height_cm 175 \
  --weight_kg 72 \
  --steps 7000 \
  --sleep_hours 7.0 \
  --heart_rate_avg 76 \
  --workout_type Walking \
  --workout_duration_minutes 30 \
  --goal_type 減重 \
  --target_change_kg 5.0 \
  --daily_diet_adjust_kcal 300
```

### 可用參數列表

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--age` | 24 | 年齡 |
| `--gender` | Male | 性別（Male / Female） |
| `--height_cm` | 170.0 | 身高（公分） |
| `--weight_kg` | 68.0 | 體重（公斤） |
| `--steps` | 6500 | 每日步數 |
| `--sleep_hours` | 6.5 | 每日睡眠時數 |
| `--heart_rate_avg` | 78 | 平均靜止心率 |
| `--workout_type` | Walking | 運動類型 |
| `--workout_duration_minutes` | 25 | 每次運動時長（分鐘） |
| `--goal_type` | 減重 | 目標類型（減重 / 增肌） |
| `--target_change_kg` | 3.0 | 目標體重變化（公斤） |
| `--daily_diet_adjust_kcal` | 250.0 | 每日飲食熱量調整（大卡） |
| `--k` | 4 | KMeans 分群數 |

---

## 🧪 執行測試

```bash
python -m pytest fitness_pipeline/tests/test_fixes.py -v
```

測試涵蓋以下 7 個修正項目：

| # | 測試項目 |
|---|---------|
| 1 | `estimate_goal_days()` dead code 移除（增肌分支） |
| 2 | `run_pipeline()` 與 `renderer` 職責分離 |
| 3 | `print_decision_report()` UI 與邏輯分離 |
| 4 | 課表評分權重加總驗證（assert sum == 1.0） |
| 5 | LLM 呼叫 timeout + retry 機制 |
| 6 | Cosine Similarity 除以零防護 |
| 7 | `load_program_candidates()` 空欄位 DataFrame 防護 |

---

## 📦 依賴套件

| 套件 | 版本建議 | 用途 |
|------|----------|------|
| `pandas` | ≥ 1.5 | 資料處理 |
| `numpy` | ≥ 1.23 | 數值計算 |
| `scikit-learn` | ≥ 1.2 | KMeans、Random Forest、Pipeline |
| `streamlit` | ≥ 1.30 | 網頁介面 |
| `python-dotenv` | ≥ 1.0 | 讀取 `.env` 的 API Key |
| `google-genai` | ≥ 0.5 | Gemini LLM 菜單生成（選用） |
| `pytest` | ≥ 7.0 | 單元測試 |

---

## 🔄 推送更動到 GitHub

```bash
# 查看目前變更
git status

# 加入所有變更
git add .

# 或只加入特定檔案
git add fitness_pipeline/recommenders/goal_engine.py

# commit
git commit -m "fix: 說明你改了什麼"

# 推送
git push
```

### commit 訊息建議格式

| 前綴 | 使用時機 |
|------|---------|
| `feat:` | 新增功能 |
| `fix:` | 修正 bug |
| `refactor:` | 重構，不影響功能 |
| `chore:` | 雜項（更新套件、調整設定） |
| `docs:` | 更新文件或 README |

> ⚠️ 注意：`data/` 資料夾與 `.env` 已被 `.gitignore` 排除，不會被推上去。

---

## 📋 輸出報告說明

| 章節 | 內容 |
|------|------|
| 一 | 使用者輸入摘要（BMI、步數、睡眠） |
| 二 | 模型摘要（Silhouette Score、Macro F1） |
| 三 | 生活型態分析（分群結果與活動等級） |
| 四 | 目標達成估算（預估天數與每日熱量缺口） |
| 五 | ML 熱量預測參考（公式法 vs 機器學習） |
| 六 | 邊際效益分析（步數、運動、睡眠 what-if） |
| 七 | 行動建議 |
| 八 | Top 3 課表推薦（含推薦理由與限制） |
| 九 | Top 5 食物推薦（Cosine Similarity 排名） |
| 十 | 飲食推薦解釋 |
| 十一 | LLM 菜單建議（需 Gemini API Key） |

---

## ⚠️ 注意事項

- 所有建議為輔助參考，**非醫療建議**
- LLM 菜單生成設有 20 秒 timeout，網路異常時自動略過
- 課表評分權重若被修改導致加總不等於 1.0，程式啟動時會立即報錯