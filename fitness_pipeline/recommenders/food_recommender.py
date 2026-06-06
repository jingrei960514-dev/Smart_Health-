"""
recommenders/food_recommender.py — cosine-similarity food recommendation.

Fix applied:
- Division-by-zero guard: rows where ||x|| == 0 receive similarity = 0
  instead of NaN / inf.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fitness_pipeline.models.fitness_models import ensure_columns
from fitness_pipeline.utils.helpers import localize_food_text


_TARGETS = {
    "balanced":     np.array([450.0, 30.0, 15.0, 45.0]),
    "fat_loss":     np.array([380.0, 35.0, 10.0, 30.0]),
    "muscle_gain":  np.array([550.0, 40.0, 18.0, 50.0]),
}

_NUTRITION_COLS = ["Calories", "Protein", "Fat", "Carbs"]


def recommend_foods(
    food_df: pd.DataFrame,
    mode: str = "balanced",
    topn: int = 5,
) -> pd.DataFrame:
    """
    Return the top-*topn* foods whose nutrition profile is most similar
    to the target vector for *mode*.

    Division-by-zero is handled by masking zero-norm rows (similarity → 0).
    """
    if mode not in _TARGETS:
        mode = "balanced"
    target = _TARGETS[mode]

    food_df = ensure_columns(
        food_df,
        {"Food": "Unknown Food", "Category": "Unknown Category",
         "Calories": 300.0, "Protein": 15.0, "Fat": 10.0, "Carbs": 35.0},
    )

    X = food_df[_NUTRITION_COLS].copy().astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    ts = scaler.transform(pd.DataFrame([target], columns=_NUTRITION_COLS))

    row_norms = np.linalg.norm(Xs, axis=1)          # shape (n,)
    target_norm = float(np.linalg.norm(ts))

    # Guard: avoid division by zero for degenerate rows
    safe_norms = np.where(row_norms == 0, 1.0, row_norms)
    if target_norm == 0:
        sim = np.zeros(len(Xs))
    else:
        dot = (Xs @ ts.T).ravel()
        sim = dot / (safe_norms * target_norm)
        sim = np.where(row_norms == 0, 0.0, sim)

    out = food_df.copy()
    out["score"] = sim
    return out.sort_values("score", ascending=False).head(topn)[
        ["Food", "Category", "Calories", "Protein", "Fat", "Carbs", "score"]
    ]


def explain_food_recommendation(food_rec: pd.DataFrame) -> List[str]:
    lines = [
        "系統根據使用者目標營養向量與食物營養成分進行 cosine similarity 比對。",
        "推薦排名越高，代表熱量、蛋白質、脂肪、碳水比例越接近目標。",
    ]
    if len(food_rec) > 0:
        lines.append(f"{food_rec.iloc[0]['Food']} 在本次排序中最接近目標配比。")
    if len(food_rec) > 2:
        lines.append(f"{food_rec.iloc[2]['Food']} 兼具蛋白質與碳水，可作為運動後補充選項。")
    lines.append("含較高碳水的主食類可提供能量，若目標為減重建議控制份量。")
    lines.append("以上為營養輔助建議，非醫療建議。")
    return lines


def build_food_display(food_rec: pd.DataFrame) -> pd.DataFrame:
    """Localize and rename columns for terminal display."""
    return (
        localize_food_text(food_rec)
        .rename(columns={
            "Food": "食物", "Category": "類別",
            "Calories": "熱量", "Protein": "蛋白質",
            "Fat": "脂肪", "Carbs": "碳水",
            "score": "相似度分數",
        })
    )