# 模組化架構說明

這份文件說明 `fitness_pipeline` 套件的設計邏輯與分工方式，方便組員理解每個檔案的職責與彼此的關係。

---

## 核心設計原則

**每個檔案只做一件事。**

原本的程式碼把所有邏輯（資料讀取、模型訓練、推薦、格式化輸出）全部塞在一個檔案裡，導致：

- 任何一處改動都可能影響其他功能
- 無法針對單一功能寫測試
- 多人協作時容易衝突

模組化之後，每個檔案的職責清楚分開，改 A 不會動到 B。

---

## 套件結構與職責

```
fitness_pipeline/
├── pipeline.py              ← 入口：負責「組裝」，本身不做任何計算
├── models/
│   ├── schemas.py           ← 資料結構定義
│   └── fitness_models.py    ← 機器學習模型
├── recommenders/
│   ├── goal_engine.py       ← 目標估算邏輯
│   ├── food_recommender.py  ← 食物推薦邏輯
│   ├── program_recommender.py ← 課表推薦邏輯
│   └── llm_meal_plan.py     ← LLM 菜單生成
├── report/
│   └── renderer.py          ← 輸出格式化（只管印出，不管計算）
├── utils/
│   └── helpers.py           ← 共用工具（本地化、顯示）
└── tests/
    └── test_fixes.py        ← 單元測試
```

---

## 各層說明

### models/schemas.py — 資料結構層

定義所有在模組之間傳遞的資料結構（dataclass）。

```
UserInput        ← 使用者輸入（年齡、體重、步數…）
GoalInput        ← 目標設定（減重 / 增肌、目標公斤數）
GoalEstimate     ← 目標估算結果（預估天數、建議熱量）
WhatIfScenario   ← 情境分析結果
ProgramRec       ← 單筆課表推薦結果
PipelineResult   ← 整個 pipeline 的最終輸出
```

**為什麼用 dataclass 而不用 dict？**

舊版用 `dict` 傳資料，取值要寫 `goal_est.get('estimated_days')`，
打錯 key 在執行期才會報錯。改成 dataclass 之後寫 `goal_est.estimated_days`，
打錯會在寫程式時就被 IDE 抓到，也更容易看懂每個欄位代表什麼。

---

### models/fitness_models.py — 模型層

負責所有機器學習相關的邏輯，包含：

- `load_and_prepare_fitness()` — 讀取 CSV、計算 BMI 與 activity_score
- `build_cluster_model()` — 訓練 KMeans 分群模型
- `build_activity_classifier()` — 訓練 Random Forest 活動等級分類器
- `estimate_daily_exercise_kcal()` — 公式法估算運動熱量
- `predict_calories_ml()` — ML 法預測運動熱量（輔助參考）

這個檔案**只做計算，不印任何東西**，方便單獨測試。

---

### recommenders/ — 推薦層

四個推薦模組各自獨立，互不依賴：

#### goal_engine.py
- `estimate_goal_days()` — 根據 BMR / TDEE 估算達標天數與建議熱量
- `what_if_analysis()` — 模擬「多走 2000 步 / 多睡 0.5 小時」等情境的效益
- `generate_action_suggestions()` — 根據活動等級給出行動建議

#### food_recommender.py
- `recommend_foods()` — 用 Cosine Similarity 比對食物營養向量
- 修正了原版的**除以零 bug**：當食物的所有營養值都是 0 時，舊版會產生 NaN，新版強制設為相似度 0

#### program_recommender.py
- `recommend_programs()` — 用加權評分推薦課表
- 修正了原版的**權重問題**：加入 `assert` 確保五個權重加總永遠等於 1.0，改錯會立即報錯
- 修正了原版的**空資料問題**：fallback CSV 沒有可用欄位時直接跳過，不會產生空白 DataFrame

#### llm_meal_plan.py
- `generate_llm_meal_plan()` — 呼叫 Gemini API 生成菜單
- 修正了原版**沒有 timeout** 的問題：設定 20 秒上限，超時自動重試最多 2 次，不會讓整個程式卡住

---

### report/renderer.py — 輸出層

只負責把 `PipelineResult` 印出來，**完全不做任何計算**。

舊版的 `print_decision_report()` 裡面混雜了格式判斷與業務邏輯，
現在所有計算都在 `pipeline.py` 完成後才傳進來，renderer 只管排版。

好處：想改輸出格式（例如改成 JSON 或 HTML）只需要動這個檔案，不會影響任何計算邏輯。

---

### utils/helpers.py — 工具層

所有模組共用的小工具：

- 中文顯示對齊（CJK 字元寬度計算）
- 本地化 mapping（英文課表名稱 → 中文、食物類別 → 中文）
- `safe_get()` — 安全取值，避免 KeyError
- `parse_list_like()` — 解析 CSV 裡存成字串的 list

這個檔案**沒有任何業務邏輯**，只是工具函式的集合，任何模組都可以 import。

---

### pipeline.py — 入口層

整個系統的唯一執行入口，負責：

1. 讀取 CLI 參數
2. 依序呼叫各模組
3. 把結果組裝成 `PipelineResult`
4. 傳給 renderer 輸出

`pipeline.py` **本身不做任何計算**，只是把各模組串起來。
這樣的好處是：測試時可以單獨呼叫 `run_pipeline()` 取得結果，不會有任何輸出干擾。

---

## 資料流向

```
使用者輸入（CLI 參數 / Streamlit 表單）
        │
        ▼
   pipeline.py          ← 組裝所有模組
   ├── fitness_models   ← 訓練模型、預測分群與活動等級
   ├── goal_engine      ← 估算目標天數與建議熱量
   ├── food_recommender ← 推薦食物
   ├── program_recommender ← 推薦課表
   └── llm_meal_plan    ← 生成 LLM 菜單
        │
        ▼
   PipelineResult       ← 所有結果打包成一個 dataclass
        │
        ▼
   renderer.py          ← 格式化輸出（終端機）
   app.py               ← 格式化輸出（Streamlit 網頁）
```

---

## 新增功能的方式

### 新增一種推薦邏輯
在 `recommenders/` 新增一個檔案，在 `pipeline.py` import 並呼叫，
最後把結果加進 `PipelineResult`，在 `renderer.py` / `app.py` 顯示。

### 新增一個輸出欄位
在 `schemas.py` 的對應 dataclass 加上欄位，
在產生該資料的模組填入值，在 renderer 取用。

### 修改推薦權重
只需要動 `program_recommender.py` 裡的 `_W_GOAL`、`_W_LEVEL` 等常數，
存檔後 `assert` 會自動幫你檢查加總是否還是 1.0。

---

## 測試方式

```bash
python -m pytest fitness_pipeline/tests/test_fixes.py -v
```

每個測試對應一個 bug 修正，確保改動不會讓已修好的問題復發。
測試**不需要任何 CSV 資料**，完全用假資料跑，任何人 clone 後都能直接執行。