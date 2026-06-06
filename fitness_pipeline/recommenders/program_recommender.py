"""
recommenders/program_recommender.py — fitness program scoring & recommendation.

Fixes applied:
1. WEIGHT_SUM assertion: weights are validated at module load; any accidental
   edit that breaks the sum triggers an immediate AssertionError.
2. load_program_candidates() now skips a fallback frame when `keep` is empty
   (previously it appended an empty-column DataFrame that caused downstream
   KeyErrors).
3. setdefault() replaced with proper DataFrame column check.
4. Paths updated to resolve relative to this file's location.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from fitness_pipeline.models.schemas import GoalInput, ProgramRec, UserInput
from fitness_pipeline.utils.helpers import (
    localize_equipment_text,
    localize_exercise_text,
    localize_goal_text,
    localize_level_text,
    parse_list_like,
)

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so they work from any cwd
# ---------------------------------------------------------------------------

_BASE                    = os.path.join(os.path.dirname(__file__), "..", "..", "handoff", "data")
PROGRAM_SUMMARY_PATH     = os.path.join(_BASE, "program_summary.csv")
PROGRAM_FITNESS_PATH     = os.path.join(_BASE, "fitness_and_workout_dataset.csv")
PROGRAM_DETAIL_PATH      = os.path.join(_BASE, "programs_detailed_boostcamp_kaggle.csv")
PROGRAM_PROFILE_ZH_PATH  = os.path.join(_BASE, "program_profiles_zh.csv")
PROGRAM_PROFILE_PATH     = os.path.join(_BASE, "program_profiles.csv")

# ---------------------------------------------------------------------------
# Scoring weights — must sum to 1.0
# ---------------------------------------------------------------------------

_W_GOAL      = 0.35
_W_LEVEL     = 0.22
_W_EQUIPMENT = 0.15
_W_TIME      = 0.18
_W_LENGTH    = 0.10
_WEIGHT_SUM  = _W_GOAL + _W_LEVEL + _W_EQUIPMENT + _W_TIME + _W_LENGTH

assert abs(_WEIGHT_SUM - 1.0) < 1e-9, (
    f"Program scoring weights must sum to 1.0, got {_WEIGHT_SUM:.6f}. "
    "Please fix the weight constants in program_recommender.py."
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_program_candidates() -> pd.DataFrame:
    if os.path.exists(PROGRAM_PROFILE_ZH_PATH):
        df = pd.read_csv(PROGRAM_PROFILE_ZH_PATH)
        if "source_dataset" not in df.columns:
            df["source_dataset"] = "program_profiles_zh.csv"
        return df

    if os.path.exists(PROGRAM_PROFILE_PATH):
        df = pd.read_csv(PROGRAM_PROFILE_PATH)
        if "source_dataset" not in df.columns:
            df["source_dataset"] = "program_profiles.csv"
        return df

    # Fallback: aggregate summary files
    useful_cols = {
        "title", "goal", "level", "equipment",
        "program_length", "time_per_workout",
        "total_exercises", "description",
    }
    frames = []
    for src, path in [
        ("program_summary", PROGRAM_SUMMARY_PATH),
        ("fitness_and_workout", PROGRAM_FITNESS_PATH),
    ]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        keep = [c for c in useful_cols if c in df.columns]
        if not keep:
            continue
        part = df[keep].copy()
        part["source"] = src
        part["source_dataset"] = src
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=list(useful_cols) + ["source", "source_dataset"])

    pool = pd.concat(frames, ignore_index=True)
    return pool.drop_duplicates(subset=["title"], keep="first")


def load_exercise_lookup() -> pd.DataFrame:
    for path in (PROGRAM_DETAIL_PATH, os.path.join(_BASE, "programs_detailed_bootcamp_kaggle.csv")):
        if not os.path.exists(path):
            continue
        header = pd.read_csv(path, nrows=0).columns.tolist()
        cols = [c for c in ["title", "exercise_name"] if c in header]
        if cols:
            return pd.read_csv(path, usecols=cols)
    return pd.DataFrame(columns=["title", "exercise_name"])


# ---------------------------------------------------------------------------
# Scoring & recommendation
# ---------------------------------------------------------------------------

def _level_preference(predicted_level: str, profile_name: str) -> str:
    pref = {"Low": "Beginner", "Medium": "Intermediate", "High": "Advanced"}.get(
        predicted_level, "Intermediate"
    )
    if "低活動量" in profile_name:
        pref = "Beginner"
    elif "高活動量" in profile_name and predicted_level != "Low":
        pref = "Intermediate"
    return pref


def recommend_programs(
    user: UserInput,
    goal: GoalInput,
    predicted_level: str,
    profile_name: str,
    topn: int = 3,
) -> List[ProgramRec]:
    pool = load_program_candidates()
    if pool.empty:
        return []

    detail = load_exercise_lookup()
    ex_count: Dict[str, int] = {}
    if not detail.empty and "title" in detail.columns:
        ex_count = detail.groupby("title").size().to_dict()

    goal_token  = "fat" if goal.goal_type == "減重" else "muscle"
    level_pref  = _level_preference(predicted_level, profile_name)
    target_time = {"Low": 45, "Medium": 60, "High": 75}.get(predicted_level, 60)
    target_len  = 8 if goal.goal_type == "減重" else 12

    scored: List[ProgramRec] = []
    for _, r in pool.iterrows():
        goals      = parse_list_like(r.get("goal", r.get("primary_goal", "")))
        levels     = parse_list_like(r.get("level", r.get("estimated_difficulty", "")))
        equipment  = str(r.get("equipment", r.get("equipment_type", "")))
        time_pw    = pd.to_numeric(r.get("time_per_workout", r.get("duration_minutes", np.nan)), errors="coerce")
        prog_len   = pd.to_numeric(r.get("program_length", r.get("program_length_weeks", np.nan)), errors="coerce")
        train_days = pd.to_numeric(r.get("training_days_per_week", np.nan), errors="coerce")

        split_zh   = str(r.get("training_split_type_zh",   r.get("training_split_type",   "資料不足")))
        muscles_zh = str(r.get("major_muscle_groups_zh",   r.get("major_muscle_groups",   "資料不足")))
        reps_ex_zh = localize_exercise_text(str(r.get("representative_exercises_zh", r.get("representative_exercises", "資料不足"))))
        goal_zh    = str(r.get("primary_goal_zh", localize_goal_text(str(r.get("primary_goal", r.get("goal", "資料不足"))))))
        diff_zh    = str(r.get("estimated_difficulty_zh", localize_level_text(str(r.get("estimated_difficulty", "資料不足")))))
        display_zh = str(r.get("display_title_zh", "個人化訓練計畫"))
        if not display_zh or display_zh == "nan":
            display_zh = "個人化訓練計畫"

        goal_score  = 1.0 if any(goal_token in g.lower() for g in goals) else 0.25
        level_score = 1.0 if any(level_pref.lower() in lv.lower() for lv in levels) else 0.35
        equip_score = 1.0 if "full gym" not in equipment.lower() else 0.6
        time_score  = 0.5 if pd.isna(time_pw)  else max(0.0, 1.0 - abs(float(time_pw) - target_time) / 90.0)
        len_score   = 0.5 if pd.isna(prog_len)  else max(0.0, 1.0 - abs(float(prog_len) - target_len)  / 16.0)

        w_length = 0.10 if goal.goal_type == "減重" else 0.15
        w_goal   = _W_GOAL + (_W_LENGTH - w_length)

        score = (
            w_goal         * goal_score
            + _W_LEVEL     * level_score
            + _W_EQUIPMENT * equip_score
            + _W_TIME      * time_score
            + w_length     * len_score
        )

        prog_len_str = "資料不足" if pd.isna(prog_len)   else f"{float(prog_len):.0f} 週"
        time_pw_str  = "資料不足" if pd.isna(time_pw)    else f"{float(time_pw):.0f} 分鐘"
        days_str     = "資料不足" if pd.isna(train_days) else f"{int(train_days)} 天"

        reason_lines = [
            f"符合使用者目標：{goal.goal_type}（課表主目標：{goal_zh}）",
            f"符合目前活動等級推估程度：{level_pref} 對應課表難度 {diff_zh}",
            (
                "此課表缺少單次訓練時間資料，因此時間適配性僅作輔助參考。"
                if pd.isna(time_pw)
                else f"每次訓練時間接近使用者可用時間（目標 {target_time} 分鐘）"
            ),
            f"課表週期適合作為中短期訓練計畫（{prog_len_str}）",
            f"訓練結構：{split_zh}，每週約 {days_str} 天",
            f"主要肌群：{muscles_zh}",
            f"代表動作：{reps_ex_zh}",
        ]
        limit_lines = [
            "若目前使用者活動等級偏低，建議循序增加訓練量",
            "若器材不足，需替換部分器材動作",
        ]

        scored.append(ProgramRec(
            display_title_zh=display_zh,
            title=str(r.get("title", "")),
            goal=localize_goal_text(str(r.get("goal", "資料不足"))),
            level=localize_level_text(str(r.get("level", diff_zh or "資料不足"))),
            equipment=localize_equipment_text(equipment or "資料不足"),
            program_length=prog_len_str,
            workout_duration=time_pw_str,
            training_days_per_week=days_str,
            training_split_type_zh=split_zh,
            major_muscle_groups_zh=muscles_zh,
            representative_exercises_zh=reps_ex_zh,
            primary_goal_zh=goal_zh,
            estimated_difficulty_zh=diff_zh,
            score=float(score),
            reason_lines=reason_lines,
            limit_lines=limit_lines,
            exercise_detail_count=int(ex_count.get(str(r.get("title", "")), 0)),
            source_dataset=str(r.get("source_dataset", r.get("source", "資料不足"))),
        ))

    return sorted(scored, key=lambda x: x.score, reverse=True)[:topn]