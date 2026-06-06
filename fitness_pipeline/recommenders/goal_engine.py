"""
recommenders/goal_engine.py — goal-day estimation, what-if analysis,
action suggestions.  No I/O.
"""
from __future__ import annotations

from typing import List, Tuple

from fitness_pipeline.models.schemas import GoalEstimate, GoalInput, UserInput, WhatIfScenario
from fitness_pipeline.models.fitness_models import estimate_daily_exercise_kcal


# ---------------------------------------------------------------------------
# BMR / TDEE helpers
# ---------------------------------------------------------------------------

def _bmr(user: UserInput) -> float:
    """Mifflin-St Jeor BMR."""
    base = 10 * user.weight_kg + 6.25 * user.height_cm - 5 * user.age
    return base + 5 if user.gender.lower() in ("male", "男") else base - 161


def _tdee(user: UserInput) -> float:
    return _bmr(user) * 1.2 + estimate_daily_exercise_kcal(user)


# ---------------------------------------------------------------------------
# Goal estimation
# ---------------------------------------------------------------------------

def estimate_goal_days(user: UserInput, goal: GoalInput) -> GoalEstimate:
    """
    Returns a fully-populated GoalEstimate.

    Fixes from original:
    - Dead-code block after `return` in the muscle-gain branch is removed.
    - Both branches now return a GoalEstimate dataclass (not a raw dict).
    - Safety floor on target_intake is enforced for both goals.
    """
    goal_type = str(goal.goal_type).strip()
    if goal_type not in ("減重", "增肌"):
        goal_type = "減重"

    exercise_kcal = estimate_daily_exercise_kcal(user)
    tdee = _tdee(user)
    min_intake = 1500 if user.gender.lower() in ("male", "男") else 1200

    if goal_type == "減重":
        total_gap = max(0.1, goal.target_change_kg) * 7700
        daily_gap = max(1.0, exercise_kcal + max(0.0, goal.daily_diet_adjust_kcal))
        days = total_gap / daily_gap
        target_intake = max(min_intake, tdee - daily_gap)
        return GoalEstimate(
            goal_type=goal_type,
            target_change_kg=goal.target_change_kg,
            target_calorie_gap=total_gap,
            daily_calorie_gap=daily_gap,
            target_intake=target_intake,
            estimated_days=days,
            note="此為簡化估算，實際結果會受代謝、飲食執行度與身體狀況影響。",
        )

    # 增肌 branch — previously had unreachable duplicate code below its return
    weekly_rate = 0.25
    days = (max(0.1, goal.target_change_kg) / weekly_rate) * 7
    target_intake = max(min_intake, tdee + 300)
    return GoalEstimate(
        goal_type=goal_type,
        target_change_kg=goal.target_change_kg,
        target_calorie_gap=0.0,
        daily_calorie_gap=0.0,
        target_intake=target_intake,
        estimated_days=days,
        note="增肌天數以每週約 0.25 kg 的保守假設估算，實際仍需依訓練與營養調整。",
    )


# ---------------------------------------------------------------------------
# What-if analysis
# ---------------------------------------------------------------------------

def what_if_analysis(
    user: UserInput,
    goal: GoalInput,
    base_days: float,
) -> Tuple[List[WhatIfScenario], str]:
    variants = [
        ("情境 A：每日步數 +2000",         {"steps": user.steps + 2000}),
        ("情境 B：運動時間 +15 分鐘",       {"workout_duration_minutes": user.workout_duration_minutes + 15}),
        ("情境 C：睡眠時間 +0.5 小時",      {"sleep_hours": user.sleep_hours + 0.5}),
    ]

    scenarios: List[WhatIfScenario] = []
    for name, delta in variants:
        u2 = UserInput(**{**user.__dict__, **delta})
        est = estimate_goal_days(u2, goal)
        new_days = est.estimated_days
        scenarios.append(WhatIfScenario(
            name=name,
            new_days=new_days,
            improve_days=max(0.0, base_days - new_days),
        ))

    best = max(scenarios, key=lambda s: s.improve_days)
    return scenarios, best.name


# ---------------------------------------------------------------------------
# Action suggestions
# ---------------------------------------------------------------------------

def generate_action_suggestions(user: UserInput, level: str) -> List[str]:
    actions: List[str] = []
    if level == "Low":
        actions.append("每日步數目標：在目前基礎上 +2000（循序增加）")
        actions.append("每週 4 天加入 20-30 分鐘步行或單車")
    elif level == "Medium":
        actions.append("每日步數目標：+1000，並加入每週 1 天間歇訓練")
        actions.append("維持每週 4-5 天、每次 30-40 分鐘有氧+肌力混合訓練")
    else:
        actions.append("維持目前訓練量，新增每週 1 天恢復日並監控心率")
        actions.append("採用週期化訓練，避免停滯或過度訓練")

    if user.sleep_hours < 7:
        actions.append("睡眠目標：提升至至少 7 小時，改善身體適應與恢復")
    if user.heart_rate_avg > 85:
        actions.append("增加低強度日；因平均心率偏高需特別注意恢復")
    return actions