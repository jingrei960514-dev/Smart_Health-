"""
report/renderer.py — terminal report renderer.

Separates all formatting/print logic from business logic.
Receives a PipelineResult dataclass and produces human-readable output.
"""
from __future__ import annotations

import numpy as np

from fitness_pipeline.models.schemas import PipelineResult
from fitness_pipeline.utils.helpers import (
    map_feature_name_to_zh,
    print_aligned_table,
    safe_get,
)

_LEVEL_ZH = {"Low": "低", "Medium": "中", "High": "高"}


def _section(title: str) -> None:
    print(f"\n{title}")
    print("-" * 60)


def render(result: PipelineResult) -> None:  # noqa: C901
    r = result
    u = r.user
    g = r.goal
    ge = r.goal_est

    print("=" * 60)
    print("可解釋個人化運動與營養決策支援系統")
    print("=" * 60)

    # ── Section 1: User summary ──────────────────────────────────────────
    _section("一、使用者輸入摘要")
    bmi = u.weight_kg / ((u.height_cm / 100.0) ** 2)
    print(f"年齡：{u.age} 歲")
    print(f"性別：{u.gender}")
    print(f"BMI：{bmi:.2f}")
    print(f"每日步數：{u.steps}")
    print(f"睡眠時數：{u.sleep_hours:.1f} 小時")
    print(f"平均心率：{u.heart_rate_avg}")
    print(f"運動類型：{u.workout_type}")
    print(f"運動時長：{u.workout_duration_minutes} 分鐘")
    print(f"目標：{ge.goal_type}")
    print(f"目標變化：{ge.target_change_kg:.1f} kg")

    # ── Section 2: Model summary ─────────────────────────────────────────
    _section("二、模型與方法摘要")
    macro_f1 = r.activity_report.get("macro avg", {}).get("f1-score", "資料不足")
    macro_f1_str = macro_f1 if isinstance(macro_f1, str) else f"{macro_f1:.3f}"
    print(f"分群模型：KMeans，分群數 = {r.k}，Silhouette = {r.sil:.3f}")
    print(f"活動等級模型：Random Forest，Macro F1 = {macro_f1_str}")
    print("飲食推薦方法：Cosine Similarity")
    print("系統用途：根據生活型態、活動資料與營養資料，提供決策輔助建議。")

    # ── Section 3: Lifestyle analysis ────────────────────────────────────
    _section("三、生活型態分析")
    print(f"分群結果：{r.profile_name}（Cluster {r.cluster_id}）")
    print(f"活動等級：{_LEVEL_ZH.get(r.predicted_level, r.predicted_level)}")
    print("\n分群解釋：")
    for line in r.cluster_reasons:
        if line.startswith("因此系統判定"):
            continue
        print(f"- {line}")

    print("\n活動等級解釋：")
    print("模型重要特徵前三名：")
    if r.top_features:
        for i, (name, score) in enumerate(r.top_features, 1):
            print(f"{i}. {map_feature_name_to_zh(name)}（重要度 {score:.3f}）")
    else:
        print("1. 資料不足")

    print("\n系統判斷：")
    if r.predicted_level == "Low":
        print("每日步數與運動時長是影響活動等級的主要因素，因此目前被判定為低活動等級。")
    elif r.predicted_level == "Medium":
        print("步數、睡眠與運動時長共同影響活動等級，目前判定為中活動等級。")
    else:
        print("步數與運動時長表現較佳，且恢復狀態可接受，因此判定為高活動等級。")

    # ── Section 4: Goal estimation ───────────────────────────────────────
    _section("四、目標達成估算")
    print(f"目標：{ge.goal_type} {ge.target_change_kg:.1f} kg")
    if ge.goal_type == "減重":
        print(f"估計總熱量差：約 {ge.target_calorie_gap:.0f} kcal")
        print(f"預估每日熱量差：約 {ge.daily_calorie_gap:.0f} kcal")
    print(f"預估達標時間：約 {ge.estimated_days:.0f} 天")
    print(f"\n提醒：\n{ge.note}")

    # ── Section 5: ML calorie cross-check ────────────────────────────────
    _section("五、補充：ML 熱量預測參考")
    print(f"公式估算本次運動熱量：約 {r.formula_calories:.0f} kcal")
    if r.ml_calories is None:
        print("ML 模型預測本次運動熱量：資料不足，暫未啟用")
        print("兩者差異：資料不足")
    else:
        print(f"ML 模型預測本次運動熱量：約 {r.ml_calories:.0f} kcal")
        print(f"兩者差異：約 {abs(r.ml_calories - r.formula_calories):.0f} kcal")
    print("\n說明：")
    print("ML 預測值來自 calories prediction regression model，僅作為輔助參考。")
    print("目前達標天數仍以公式估算為主，避免模型不確定性影響主要建議。")
    print("此 ML 預測為輔助參考，非醫療或保證結果。")

    # ── Section 6: Marginal benefit / what-if ────────────────────────────
    _section("六、邊際效益分析")
    print(f"目前方案預估達標時間：{ge.estimated_days:.0f} 天")
    print("\n情境比較：")
    for i, s in enumerate(r.whatif, 1):
        short = s.name.replace("情境 A：", "").replace("情境 B：", "").replace("情境 C：", "")
        print(f"{i}. {short}")
        print(f"   新預估達標時間：{s.new_days:.0f} 天")
        print(f"   改善：{s.improve_days:.0f} 天\n")
    best_short = r.best_case.replace("情境 A：", "").replace("情境 B：", "").replace("情境 C：", "")
    print(f"系統建議：\n對此使用者而言，「{best_short}」的邊際效益最大，可優先嘗試。")

    # ── Section 7: Actions ───────────────────────────────────────────────
    _section("七、行動建議")
    for i, a in enumerate(r.actions, 1):
        print(f"{i}. {a}")

    # ── Section 8: Program recommendations ──────────────────────────────
    _section("八、Top 3 個人化課表推薦")
    if not r.program_recs:
        print("課表推薦資料不足")
    else:
        for i, rec in enumerate(r.program_recs, 1):
            print(f"推薦課表 {i}")
            print(f"課表名稱：{rec.display_title_zh}")
            print(f"原始課表名稱：{rec.title}\n")
            print("適合對象：")
            print(f"- {rec.level}")
            print(f"- {rec.primary_goal_zh}")
            print(f"- {rec.equipment}\n")
            print("課表資訊：")
            print(f"- 課表長度：{rec.program_length}")
            print(f"- 每週訓練天數：{rec.training_days_per_week}")
            print(f"- 每次訓練：約 {rec.workout_duration}")
            print(f"- 訓練方式：{rec.training_split_type_zh}")
            print(f"- 主要訓練肌群：{rec.major_muscle_groups_zh}")
            print(f"- 代表動作：{rec.representative_exercises_zh}\n")
            print(f"推薦分數：{rec.score:.3f}\n")
            print("推薦理由：")
            for line in rec.reason_lines:
                print(f"- {line}")
            print("\n限制：")
            for line in rec.limit_lines:
                print(f"- {line}")
            print(f"\n資料來源：{rec.source_dataset}")
            print(f"可查詢動作明細：{rec.exercise_detail_count} 筆\n")

    # ── Section 9: Food table ────────────────────────────────────────────
    _section("九、Top 5 營養輔助推薦")
    table_df = r.food_display_df.copy().reset_index(drop=True)
    table_df.insert(0, "排名", np.arange(1, len(table_df) + 1))
    table_df = table_df.rename(columns={"相似度分數": "相似度"})
    print_aligned_table(table_df.round(3))

    # ── Section 10: Food explanation ─────────────────────────────────────
    _section("十、飲食推薦解釋")
    print("系統根據使用者目標營養向量與食物營養成分進行 Cosine Similarity 比對。")
    print("相似度越高，代表該食物的熱量、蛋白質、脂肪與碳水比例越接近使用者目標。")
    print(f"本次飲食推薦模式依使用者目標設定為：{r.goal_mode}。")
    print("\n推薦說明：")
    for line in (r.food_reasons[2:5] if len(r.food_reasons) >= 5 else r.food_reasons):
        print(f"- {line}")
    print("\n注意：\n以上為營養輔助建議，非醫療建議。")

    # ── Section 11: LLM meal plan ────────────────────────────────────────
    _section("十一、LLM 營養師專屬菜單建議")
    print(r.llm_meal_plan)
    print("\n" + "=" * 60)