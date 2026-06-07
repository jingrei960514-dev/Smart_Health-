# Dataset 應用說明（以 `fitness_personalized_pipeline_v2.py` 為主）

本文件整理目前專案中各資料集的實際用途、清洗與欄位轉換方式，以及它們被用在哪些模組。

---

## 1) `cleaned_fitness_data_v2.csv`

### 角色
- **主模型核心資料集**（健康/活動紀錄主資料）
- 用於：
  - KMeans 使用者生活型態分群
  - 活動等級分類（RandomForest）
  - 公式達標估算（搭配使用者輸入）
  - ML 熱量預測參考（補充顯示）

### 主要原始欄位
- `user_id`
- `date`
- `age`
- `gender`
- `height_cm`
- `weight_kg`
- `steps`
- `sleep_hours`
- `heart_rate_avg`
- `workout_type`
- `workout_duration_minutes`
- `calories_burned`
- `calories_burned_per_minute`

### 做過的清洗 / 欄位轉換
- 欄位存在性補齊（`ensure_columns`）：若缺欄位補預設值（防 crash）
- `workout_type` 清洗：空值/`nan` 字串 -> `No Workout`
- 衍生欄位：
  - `bmi = weight_kg / (height_m^2)`
  - `activity_score`（步數/運動時長/睡眠/心率組合）
  - `activity_level`（依 `activity_score` 分位數切 Low/Medium/High）
- 分群前數值標準化：`StandardScaler`
- 分類前前處理：
  - 數值：`median imputation + StandardScaler`
  - 類別：`most_frequent imputation + OneHotEncoder`

### 最後用在哪個模組
- 模組 A：KMeans 分群與分群解釋
- 模組 B：活動等級分類（RandomForestClassifier）
- 模組 C（補充）：ML 熱量預測參考（RandomForestRegressor）
- 模組 D：達標天數公式估算 / What-if 分析（以輸入特徵為主）

---

## 2) `cleaned_nutrients_analysis.csv`

### 角色
- **飲食推薦資料集**
- 用於 Top 5 營養輔助推薦（Cosine Similarity）

### 主要原始欄位
- `Food`
- `Grams`
- `Calories`
- `Protein`
- `Fat`
- `Carbs`
- `Category`

### 做過的清洗 / 欄位轉換
- 欄位存在性補齊（`ensure_columns`，避免缺欄位失敗）
- 使用欄位：`Calories/Protein/Fat/Carbs`
- 標準化：`StandardScaler`
- 以目標營養向量（fat_loss / balanced / muscle_gain）計算 cosine similarity
  - 飲食模式由 `goal_type` 決定：`減重 -> fat_loss`、`增肌 -> muscle_gain`、其他 -> `balanced`
- 輸出前文字在地化（食物名與類別部分中文化）

### 最後用在哪個模組
- 模組 E：Top 5 飲食推薦 + 推薦解釋

---

## 3) `program_profiles_zh.csv`

### 角色
- **v2 課表推薦優先資料源（Program-level）**
- 目前 v2 的 Top 3 個人化課表推薦優先讀取這份

### 主要欄位（含中文化擴充）
- 原始 profile 欄位：
  - `title`
  - `program_length_weeks`
  - `training_days_per_week`
  - `estimated_difficulty`
  - `primary_goal`
  - `training_split_type`
  - `major_muscle_groups`
  - `representative_exercises`
  - `equipment_type`
  - `exercise_count`
- 中文擴充欄位：
  - `display_title_zh`
  - `primary_goal_zh`
  - `training_split_type_zh`
  - `major_muscle_groups_zh`
  - `representative_exercises_zh`
  - `equipment_type_zh`
  - `estimated_difficulty_zh`
  - `source_dataset`

### 做過的清洗 / 欄位轉換
- 由 `program_profiles.csv` 轉中文欄位映射而來
- `display_title_zh` 以規則由目標/難度/器材/分化型態組合
- 保留原始 `title` 供追溯

### 最後用在哪個模組
- 模組 F：Top 3 個人化課表推薦（Weighted Scoring + 中文友善輸出）

---

## 4) `program_profiles.csv`

### 角色
- **課表推薦次優先 fallback（Program-level）**
- 當 `program_profiles_zh.csv` 不存在時，v2 讀取這份

### 主要欄位
- `title`
- `program_length_weeks`
- `training_days_per_week`
- `estimated_difficulty`
- `primary_goal`
- `training_split_type`
- `major_muscle_groups`
- `representative_exercises`
- `equipment_type`
- `exercise_count`

### 做過的清洗 / 欄位轉換
- 由 `program_profile_builder.py` 從 exercise-level 彙總而來（每課表 1 筆）
- `difficulty/split/major_muscles` 為規則推估欄位

### 最後用在哪個模組
- 模組 F fallback：課表推薦

---

## 5) `programs_detailed_boostcamp_kaggle.csv`

### 角色
- **exercise-level 細節資料庫（不直接進主推薦模型）**
- 在 v2 中僅用於 detail lookup（例如估計可查詢動作明細筆數）

### 主要原始欄位（常用）
- `title`
- `week`
- `day`
- `exercise_name`
- `sets`
- `reps`
- `intensity`
- 另含 `goal/level/equipment/program_length/time_per_workout` 等

### 做過的清洗 / 欄位轉換
- 不直接在 v2 主流程逐筆建模
- 由 `program_profile_builder.py` 彙整為 `program_profiles.csv`

### 最後用在哪個模組
- 模組 F（輔助）：課表推薦輸出中的「可查詢動作明細筆數」

---

## 6) `program_summary.csv` 與 `fitness_and_workout_dataset.csv`

### 角色
- **課表推薦最末層 fallback 候選庫**
- 當 `program_profiles_zh.csv`、`program_profiles.csv` 都不存在時才直接使用

### 主要原始欄位
- `title`
- `description`
- `level`
- `goal`
- `equipment`
- `program_length`
- `time_per_workout`
- `total_exercises`

### 做過的清洗 / 欄位轉換
- 去重（依 `title`）
- list-like 欄位（`goal/level`）以規則解析
- 套用同一套 Weighted Scoring

### 最後用在哪個模組
- 模組 F（最末 fallback）：課表推薦

---

## 7) `gym_members_exercise_tracking.csv`

### 角色
- **實驗資料集（不在 v2 主流程）**
- 用於 `experiments/dataset_expansion` 的 Calories Prediction Comparison

### 主要原始欄位
- `Age`, `Gender`, `Weight (kg)`, `Height (m)`, `Avg_BPM`, `Session_Duration (hours)`, `Calories_Burned`, `Workout_Type`, `BMI`, ...

### 做過的清洗 / 欄位轉換（在實驗腳本中）
- `Height (m)` -> `height_cm`
- `Session_Duration (hours)` -> `duration_minutes`
- `Avg_BPM` -> `heart_rate_avg`
- `Calories_Burned` -> `calories_burned`
- 與 baseline 資料做 schema 對齊後 `concat` 比較

### 最後用在哪個模組
- 不進 v2 主流程；只在擴充實驗中使用

---

## 8) 哪些資料集「只是補充資料庫，不進主模型」

### 不進主模型（v2）
- `programs_detailed_boostcamp_kaggle.csv`：只當 exercise detail lookup / profile 來源
- `program_summary.csv`：fallback 候選庫
- `fitness_and_workout_dataset.csv`：fallback 候選庫
- `gym_members_exercise_tracking.csv`：實驗比較資料

### 進主流程核心
- `cleaned_fitness_data_v2.csv`：分群 + 分類 + 公式估算 + ML 熱量參考
- `cleaned_nutrients_analysis.csv`：營養推薦
- `program_profiles_zh.csv`：課表推薦主來源

---

## 9) v2 的資料讀取優先順序（課表推薦）

`fitness_personalized_pipeline_v2.py` 目前順序：
1. `program_profiles_zh.csv`
2. `program_profiles.csv`
3. `program_summary.csv + fitness_and_workout_dataset.csv`
4. 若都不可用：顯示「課表推薦資料不足」

這樣可兼顧：
- 執行效率（避免直接掃 60 萬筆細節做即時推薦）
- 可解釋性（profile 欄位可直接輸出）
- 穩定性（多層 fallback 不崩潰）
