"""
schemas.py — shared dataclasses and typed result structures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

@dataclass
class UserInput:
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    steps: int
    sleep_hours: float
    heart_rate_avg: int
    workout_type: str
    workout_duration_minutes: int


@dataclass
class GoalInput:
    goal_type: str          # "減重" | "增肌"
    target_change_kg: float
    daily_diet_adjust_kcal: float


# ---------------------------------------------------------------------------
# Result schemas (replaces raw Dict[str, float] scattershot)
# ---------------------------------------------------------------------------

@dataclass
class GoalEstimate:
    goal_type: str
    target_change_kg: float
    target_calorie_gap: float
    daily_calorie_gap: float
    target_intake: float
    estimated_days: float
    note: str


@dataclass
class WhatIfScenario:
    name: str
    new_days: float
    improve_days: float


@dataclass
class ProgramRec:
    display_title_zh: str
    title: str
    goal: str
    level: str
    equipment: str
    program_length: str
    workout_duration: str
    training_days_per_week: str
    training_split_type_zh: str
    major_muscle_groups_zh: str
    representative_exercises_zh: str
    primary_goal_zh: str
    estimated_difficulty_zh: str
    score: float
    reason_lines: List[str] = field(default_factory=list)
    limit_lines: List[str] = field(default_factory=list)
    exercise_detail_count: int = 0
    source_dataset: str = "資料不足"


@dataclass
class PipelineResult:
    """Aggregated output of the full pipeline — passed to the report renderer."""
    user: UserInput
    goal: GoalInput
    sil: float
    activity_report: Dict
    cluster_id: int
    profile_name: str
    cluster_reasons: List[str]
    predicted_level: str
    top_features: List[Tuple[str, float]]
    level_reasons: List[str]
    goal_est: GoalEstimate
    goal_mode: str
    formula_calories: float
    ml_calories: Optional[float]
    whatif: List[WhatIfScenario]
    best_case: str
    actions: List[str]
    program_recs: List[ProgramRec]
    food_display_df: object          # pd.DataFrame — avoids circular import
    food_reasons: List[str]
    llm_meal_plan: str
    k: int                           # KMeans k used