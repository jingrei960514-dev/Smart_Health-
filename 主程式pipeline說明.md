# 主程式 Pipeline 說明（以 `fitness_personalized_pipeline_v2.py` 為主）

本文件用來快速理解主程式從「使用者輸入」到「最終決策報告」的完整流程。

---

## 1) 主程式 Pipeline（總覽）

`fitness_personalized_pipeline_v2.py` 的核心流程：

1. 讀取主資料集與營養資料集
2. 建立使用者分群模型（KMeans）
3. 建立活動等級分類模型（RandomForestClassifier）
4. 接收使用者輸入（目標設定 + 當日活動與健康資料）
5. 對該使用者做：
   - 分群判定與分群解釋
   - 活動等級預測與解釋
6. 用公式估算目標達成時間（減重/增肌）
7. 進行 What-if 邊際效益分析（步數、運動時長、睡眠）
8. 產生行動建議
9. 產生 Top 3 個人化課表推薦（v2 新增）
10. 依 `goal_type` 決定飲食模式（fat_loss / muscle_gain / balanced），產生 Top 5 飲食推薦（Cosine Similarity）
11. 組裝成最終中文決策報告輸出到終端機

---

## 資料粒度（Data Granularity）

本系統主資料集 `cleaned_fitness_data_v2.csv`
的資料粒度（granularity）為：

`(user_id, date)`

也就是：「一位使用者在某一天的健康與活動紀錄」。

因此：
- 一筆資料不代表一個人一生的固定狀態
- 也不是長期平均習慣
- 而是某位使用者在特定日期的 Daily Activity Snapshot（每日活動快照）

- steps：當日步數
- sleep_hours：當日睡眠時數
- workout_duration_minutes：當日總運動時長
- workout_type：當日主要運動類型
- heart_rate_avg：當日平均心率

因此 KMeans 分群與 Random Forest 活動等級預測，較適合解讀為：「當日活動狀態分析」

---

## 2) 使用者輸入什麼

主程式透過 CLI 參數接收以下輸入，分成兩類：

## (1) 目標設定

- `--goal_type`：目標（`減重` / `增肌`）
- `--target_change_kg`：目標變化公斤數

## (2) 當日活動與健康資料

- `--age`：年齡
- `--gender`：性別
- `--height_cm`：身高（cm）
- `--weight_kg`：體重（kg）
- `--bmi`：`v2` 不直接由 CLI 輸入，於程式內由 `height_cm` 與 `weight_kg` 計算
- `--steps`：每日步數
- `--sleep_hours`：睡眠時數
- `--heart_rate_avg`：平均心率
- `--workout_type`：運動類型
- `--workout_duration_minutes`：運動時長（分鐘）
- `--daily_diet_adjust_kcal`：每日飲食熱量調整  **這個是什麼意思?**
- `--k`：KMeans 分群數（預設 4）

## goal_type 影響範圍

### 會影響
- 達標時間估算（`estimate_goal_days`）
- 邊際效益分析（`what_if_analysis` 會重算達標天數）
- 課表推薦（`recommend_programs` 的加權與目標匹配）
- 飲食推薦模式（`減重 -> fat_loss`、`增肌 -> muscle_gain`、其他 -> `balanced`）

### 不會影響
- KMeans 分群
- Random Forest 活動等級預測

---

## 3) 每一步模型在做什麼

## Step A. 資料載入與前處理
- 函式：`load_and_prepare_fitness()`
- 作用：
  - 讀取 `cleaned_fitness_data_v2.csv`
  - 修正 `workout_type` 空值為 `No Workout`
  - 建立 `bmi`、`activity_score`
  - 依分位數切出 `activity_level`（Low/Medium/High）

## Step B. 分群模型（KMeans）
- 函式：`build_cluster_model()`
- 特徵：`age, bmi, steps, sleep_hours, heart_rate_avg, workout_duration_minutes`
- 流程：
  - `StandardScaler` 標準化
  - `KMeans(n_clusters=k)` 分群
  - 計算 `silhouette_score`
- 目的：把使用者放到健康行為族群，作為解釋與策略定位

## Step C. 活動等級分類模型（RandomForestClassifier）
- 函式：`build_activity_classifier()`
- 輸入特徵：
  - 數值：`age, height_cm, weight_kg, steps, sleep_hours, heart_rate_avg, workout_duration_minutes, bmi`
  - 類別：`gender, workout_type`
- 流程：
  - 數值：imputer + scaler
  - 類別：imputer + one-hot
  - RandomForest 分類 `activity_level`
- 目的：輸出使用者活動狀態（低/中/高）

## Step D. 公式估算（非 ML 主依據）
- 函式：`estimate_goal_days()`
- 目的：依目標（減重/增肌）計算預估達標天數
- 減重邏輯：`target_change_kg * 7700 / daily_gap`      
> **這個公式是怎麼推導出來的?**
- 增肌邏輯：保守每週增肌速率推估

## Step E. What-if 邊際效益分析
- 函式：`what_if_analysis()`
- 三種情境：
  1. 步數 +2000
  2. 運動時長 +15 分鐘
  3. 睡眠 +0.5 小時
- 目的：比較哪個改變最能縮短達標時間

## Step F. ML 熱量預測（輔助參考）
- 函式：`predict_calories_ml()`
- 作用：用基礎回歸流程預測一次運動熱量
- 重要：**只做參考，不取代公式達標估算**     
> **所以主要的熱量消耗其實是由公式算而不是推估出來的嗎?**

## Step G. 課表推薦（v2 版本重點）
- 函式：`recommend_programs()`
- 資料來源優先順序：
  1. `program_profiles_zh.csv`
  2. `program_profiles.csv`
  3. `program_summary.csv + fitness_and_workout_dataset.csv`（fallback）
- 方法：Weighted Scoring（goal/level/equipment/time/program_length）
- 輔助：可讀取 detailed dataset 做動作明細筆數 lookup

## Step H. 飲食推薦
- 函式：`recommend_foods()`
- 模式決策：依 `goal_type` 決定 `food_mode`
  - `減重 -> fat_loss`
  - `增肌 -> muscle_gain`
  - 其他 -> `balanced`
- 方法：以營養向量做 `Cosine Similarity`
- 目的：輸出 Top 5 營養輔助建議

---

## 4) 每一步輸出什麼

## 分群輸出
- `cluster_id`
- 族群中文標籤（如：中等平衡型生活）
- 分群解釋（步數/睡眠/心率/時長/BMI vs 群均值）

## 活動等級輸出
- 預測等級：低/中/高
- 前三重要特徵
- 解釋文字（為何判定此等級）

## 目標估算輸出
- 目標類型與公斤數
- 預估每日熱量差
- 預估達標天數
- 估算注意事項

## 邊際效益輸出
- 三情境的新達標天數
- 改善天數
- 最佳優先策略

## 課表推薦輸出（v2）
- Top 3 推薦課表
- 中文顯示名稱 + 原始 title
- 適合對象、課表長度、每週天數、分化類型、主要肌群、代表動作
- 推薦理由與限制
- 資料來源與可查詢動作明細數量

## 飲食推薦輸出
- Top 5 食物
- 熱量/蛋白質/脂肪/碳水/相似度
- 推薦原因（營養輔助，非醫療）

---

## 5) 最後報告怎麼生成

## Pipeline 流程圖（v2）

```text
使用者輸入
  ├─ 目標設定：goal_type, target_change_kg
  └─ 當日資料：age, gender, height/weight, steps, sleep, hr, workout

主資料處理 -> KMeans 分群 -> RF 活動等級
                 │               │
                 │               └─ 活動等級解釋
                 └─ 分群解釋

goal_type
  ├─ 達標估算（公式）
  ├─ 邊際效益分析（what-if）
  ├─ 課表推薦加權
  └─ 飲食模式選擇（fat_loss / muscle_gain / balanced）

課表推薦 + 飲食推薦 + 行動建議 -> print_decision_report() 生成最終報告
```

最終報告由 `print_decision_report()` 統一輸出成章節式文字，順序如下：

1. 使用者輸入摘要
2. 模型與方法摘要
3. 生活型態分析（分群 + 活動等級）
4. 目標達成估算
5. ML 熱量預測參考
6. 邊際效益分析
7. 行動建議
8. Top 3 個人化課表推薦（v2）
9. Top 5 營養輔助推薦
10. 飲食推薦解釋

也就是說，主程式不是單一模型輸出，而是把多個模組結果整合成「可解釋決策支援報告」。

---

## 6) 一句話總結（答辯可用）

`fitness_personalized_pipeline_v2.py` 是一個多模組決策支援 pipeline：
用分群定位使用者型態、用分類判斷活動等級、用公式穩定估算達標時間、用 what-if 提供策略優先順序，再整合課表與飲食推薦，輸出可讀性高且可追溯的個人化建議報告。
