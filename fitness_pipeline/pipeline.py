"""
pipeline.py — orchestrates the full analysis pipeline.

run_pipeline() is the single entry point.  It:
  1. Loads & prepares data.
  2. Fits models.
  3. Runs all recommenders / estimators.
  4. Bundles results into a PipelineResult dataclass.
  5. Calls the renderer.

This replaces the original monolithic run_demo() with a function that
is easy to unit-test (return PipelineResult without printing) and extend.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path  # <-- 引入 pathlib

import pandas as pd

from fitness_pipeline.models.fitness_models import (
    build_activity_classifier,
    build_cluster_model,
    estimate_daily_exercise_kcal,
    explain_activity_level,
    get_top_feature_importance,
    load_and_prepare_fitness,
    predict_calories_ml,
    summarize_clusters,
    cluster_label,
    explain_cluster_assignment,
    NUM_FEATURES,
    CAT_FEATURES,
    CLUSTER_FEATURES,
)
from fitness_pipeline.models.schemas import (
    GoalInput,
    PipelineResult,
    UserInput,
)
from fitness_pipeline.recommenders.food_recommender import (
    build_food_display,
    explain_food_recommendation,
    recommend_foods,
)
from fitness_pipeline.recommenders.goal_engine import (
    estimate_goal_days,
    generate_action_suggestions,
    what_if_analysis,
)
from fitness_pipeline.recommenders.llm_meal_plan import generate_llm_meal_plan
from fitness_pipeline.recommenders.program_recommender import recommend_programs
from fitness_pipeline.report.renderer import render

# =====================================================================
# 📌 修正後的路徑邏輯 (扁平化結構專用)
# =====================================================================
# __file__ 是 pipeline.py，它的 parent 是 fitness_pipeline/
# 再一層 parent 就是專案頂層根目錄 (Smart_Health_new/)
BASE_DIR = Path(__file__).resolve().parent.parent

# 乾淨地指向最外層的 data/ 資料夾
FITNESS_PATH = BASE_DIR / "data" / "cleaned_fitness_data_v2.csv"
FOOD_PATH    = BASE_DIR / "data" / "cleaned_nutrients_analysis.csv"
# =====================================================================


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    """
    Execute the full pipeline and return a PipelineResult.
    Printing is handled separately by the renderer so this function
    is independently testable.
    """
    # ── 1. Load data ─────────────────────────────────────────────────────
    fitness = load_and_prepare_fitness(FITNESS_PATH)
    food_df = pd.read_csv(FOOD_PATH)

    # ── 2. Fit models ─────────────────────────────────────────────────────
    clustered, kmeans, scaler, _cluster_feats, sil = build_cluster_model(fitness, k=args.k)
    cluster_profile = summarize_clusters(clustered)
    clf, _num_cols, _cat_cols, activity_report = build_activity_classifier(clustered)

    # ── 3. Build user row ─────────────────────────────────────────────────
    user = UserInput(
        age=args.age,
        gender=args.gender,
        height_cm=args.height_cm,
        weight_kg=args.weight_kg,
        steps=args.steps,
        sleep_hours=args.sleep_hours,
        heart_rate_avg=args.heart_rate_avg,
        workout_type=args.workout_type,
        workout_duration_minutes=args.workout_duration_minutes,
    )
    goal = GoalInput(
        goal_type=args.goal_type,
        target_change_kg=args.target_change_kg,
        daily_diet_adjust_kcal=args.daily_diet_adjust_kcal,
    )

    user_df = pd.DataFrame([user.__dict__])
    user_df["bmi"] = user_df["weight_kg"] / ((user_df["height_cm"] / 100.0) ** 2)

    # ── 4. Cluster & classify ─────────────────────────────────────────────
    cXs = scaler.transform(user_df[CLUSTER_FEATURES])
    cluster_id = int(kmeans.predict(cXs)[0])
    predicted_level = str(clf.predict(user_df[NUM_FEATURES + CAT_FEATURES])[0])

    profile_row  = cluster_profile[cluster_profile["cluster_id"] == cluster_id].iloc[0]
    profile_name = cluster_label(profile_row)
    cluster_reasons = explain_cluster_assignment(user_df, profile_row, profile_name)
    top_features    = get_top_feature_importance(clf, topn=3)
    level_reasons   = explain_activity_level(user, predicted_level, top_features)

    # ── 5. Goal & calorie estimation ──────────────────────────────────────
    goal_est         = estimate_goal_days(user, goal)
    formula_calories = estimate_daily_exercise_kcal(user)
    
    # 💡 這裡原本也是傳入舊路徑，使用修正後的 FITNESS_PATH (支援 pathlib 物件或轉字串)
    ml_calories      = predict_calories_ml(user, str(FITNESS_PATH)) 
    
    whatif, best_case = what_if_analysis(user, goal, goal_est.estimated_days)
    actions          = generate_action_suggestions(user, predicted_level)

    # ── 6. Recommendations ────────────────────────────────────────────────
    food_mode_map = {"減重": "fat_loss", "增肌": "muscle_gain"}
    goal_mode = food_mode_map.get(goal.goal_type, "balanced")

    food_rec      = recommend_foods(food_df, mode=goal_mode, topn=5)
    food_display  = build_food_display(food_rec)
    food_reasons  = explain_food_recommendation(food_rec)
    program_recs  = recommend_programs(user, goal, predicted_level, profile_name, topn=3)
    llm_meal_plan = generate_llm_meal_plan(goal_est, food_rec)

    return PipelineResult(
        user=user,
        goal=goal,
        sil=sil,
        activity_report=activity_report,
        cluster_id=cluster_id,
        profile_name=profile_name,
        cluster_reasons=cluster_reasons,
        predicted_level=predicted_level,
        top_features=top_features,
        level_reasons=level_reasons,
        goal_est=goal_est,
        goal_mode=goal_mode,
        formula_calories=formula_calories,
        ml_calories=ml_calories,
        whatif=whatif,
        best_case=best_case,
        actions=actions,
        program_recs=program_recs,
        food_display_df=food_display,
        food_reasons=food_reasons,
        llm_meal_plan=llm_meal_plan,
        k=args.k,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="健身個人化建議 pipeline 展示")
    p.add_argument("--k",                        type=int,   default=4)
    p.add_argument("--age",                      type=int,   default=24)
    p.add_argument("--gender",                   type=str,   default="Male")
    p.add_argument("--height_cm",                type=float, default=170.0)
    p.add_argument("--weight_kg",                type=float, default=68.0)
    p.add_argument("--steps",                    type=int,   default=6500)
    p.add_argument("--sleep_hours",              type=float, default=6.5)
    p.add_argument("--heart_rate_avg",           type=int,   default=78)
    p.add_argument("--workout_type",             type=str,   default="Walking")
    p.add_argument("--workout_duration_minutes", type=int,   default=25)
    p.add_argument("--goal_type",                type=str,   default="減重")
    p.add_argument("--target_change_kg",         type=float, default=3.0)
    p.add_argument("--daily_diet_adjust_kcal",   type=float, default=250.0)
    args = p.parse_args()

    result = run_pipeline(args)
    render(result)


if __name__ == "__main__":
    main()