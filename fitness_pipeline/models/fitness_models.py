"""
models/fitness_models.py — KMeans clustering + Random Forest activity classifier
+ calorie regression.  All functions are pure (no I/O side effects).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fitness_pipeline.models.schemas import UserInput
from fitness_pipeline.utils.helpers import map_feature_name_to_zh


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def ensure_columns(df: pd.DataFrame, defaults: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val
    return out


def load_and_prepare_fitness(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_columns(
        df,
        {
            "age": 30, "gender": "Unknown", "height_cm": 170.0,
            "weight_kg": 65.0, "steps": 6000, "sleep_hours": 7.0,
            "heart_rate_avg": 75, "workout_type": "No Workout",
            "workout_duration_minutes": 0,
        },
    )
    df["workout_type"] = (
        df["workout_type"].astype(str).str.strip()
        .replace({"nan": "No Workout", "": "No Workout"})
    )
    df.loc[df["workout_type"].isna(), "workout_type"] = "No Workout"

    height_m = df["height_cm"] / 100.0
    df["bmi"] = df["weight_kg"] / (height_m ** 2)
    df["activity_score"] = (
        0.5 * (df["steps"] / 1000)
        + 2.0 * (df["workout_duration_minutes"] / 30)
        + 1.2 * df["sleep_hours"]
        - 0.03 * (df["heart_rate_avg"] - 70).clip(lower=0)
    )

    q1, q2 = df["activity_score"].quantile([0.33, 0.66])
    df["activity_level"] = pd.cut(
        df["activity_score"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Low", "Medium", "High"],
    )
    return df


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

CLUSTER_FEATURES: List[str] = [
    "age", "bmi", "steps", "sleep_hours",
    "heart_rate_avg", "workout_duration_minutes",
]


def build_cluster_model(df: pd.DataFrame, k: int = 4):
    """Return (clustered_df, kmeans, scaler, feature_list, silhouette_score)."""
    X = df[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    df = df.copy()
    df["cluster_id"] = labels
    return df, kmeans, scaler, CLUSTER_FEATURES, sil


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("cluster_id")[
            ["steps", "sleep_hours", "heart_rate_avg",
             "workout_duration_minutes", "bmi"]
        ]
        .mean()
        .round(2)
    )
    agg["size"] = df.groupby("cluster_id").size()
    return agg.reset_index()


def cluster_label(row: pd.Series) -> str:
    if row["steps"] < 5000 and row["sleep_hours"] < 7:
        return "低活動量 + 低恢復"
    if row["steps"] > 9000 and row["sleep_hours"] >= 7:
        return "高活動量 + 良好恢復"
    if row["heart_rate_avg"] > 82:
        return "中等活動量 + 心肺負荷偏高"
    return "中等平衡型生活"


def explain_cluster_assignment(
    user_df: pd.DataFrame,
    cluster_row: pd.Series,
    profile_name: str,
) -> List[str]:
    u = user_df.iloc[0]
    steps_label = (
        "低活動量" if u["steps"] < 5000
        else "中等活動量" if u["steps"] < 9000
        else "高活動量"
    )
    sleep_label = (
        "略低於建議值" if u["sleep_hours"] < 7
        else "符合建議範圍" if u["sleep_hours"] <= 9
        else "高於一般建議範圍"
    )
    return [
        f"步數 {int(u['steps'])}，屬於{steps_label}（群組均值 {cluster_row['steps']:.0f}）",
        f"睡眠 {float(u['sleep_hours']):.1f} 小時，{sleep_label}（群組均值 {cluster_row['sleep_hours']:.1f}）",
        f"運動時長 {int(u['workout_duration_minutes'])} 分鐘，屬於低至中等運動量"
        f"（群組均值 {cluster_row['workout_duration_minutes']:.1f}）",
        f"平均心率 {int(u['heart_rate_avg'])}（群組均值 {cluster_row['heart_rate_avg']:.1f}），"
        f"BMI {float(u['bmi']):.1f}（群組均值 {cluster_row['bmi']:.1f}）",
        f"因此系統判定為「{profile_name}」。",
    ]


# ---------------------------------------------------------------------------
# Activity-level classifier
# ---------------------------------------------------------------------------

NUM_FEATURES: List[str] = [
    "age", "height_cm", "weight_kg", "steps",
    "sleep_hours", "heart_rate_avg", "workout_duration_minutes", "bmi",
]
CAT_FEATURES: List[str] = ["gender", "workout_type"]


def build_activity_classifier(df: pd.DataFrame):
    """Return (fitted_pipeline, num_cols, cat_cols, classification_report_dict)."""
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df["activity_level"].astype(str)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=250, random_state=42,
        min_samples_leaf=3, class_weight="balanced",
    )
    pipe = Pipeline([("pre", pre), ("model", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)
    return pipe, NUM_FEATURES, CAT_FEATURES, report


def get_top_feature_importance(
    pipe: Pipeline,
    topn: int = 3,
) -> List[Tuple[str, float]]:
    try:
        feat_names = pipe.named_steps["pre"].get_feature_names_out()
        importances = pipe.named_steps["model"].feature_importances_
        pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
        return [(n, float(v)) for n, v in pairs[:topn]]
    except Exception:
        return []


def explain_activity_level(
    user: UserInput,
    predicted_level: str,
    top_features: List[Tuple[str, float]],
) -> List[str]:
    lines: List[str] = []
    if top_features:
        lines.append("模型重要特徵（前 3 名）:")
        for name, score in top_features:
            lines.append(f"- {map_feature_name_to_zh(name)}（重要度 {score:.3f}）")

    if user.steps < 7000:
        lines.append("- 每日步數偏低，是主要影響因素之一")
    if user.workout_duration_minutes < 30:
        lines.append("- 運動時長低於較高活動等級常見範圍")
    if user.sleep_hours < 7:
        lines.append("- 睡眠時間略不足，可能影響恢復與活動狀態")
    if not lines:
        lines.append("- 目前活動、睡眠與心率特徵整體落在中高活動樣本區間")

    level_zh = {"Low": "低活動等級", "Medium": "中活動等級", "High": "高活動等級"}
    lines.append(f"- 綜合判定結果偏向「{level_zh.get(predicted_level, predicted_level)}」")
    return lines


# ---------------------------------------------------------------------------
# Calorie estimation
# ---------------------------------------------------------------------------

def estimate_daily_exercise_kcal(user: UserInput) -> float:
    """Simple formula-based calorie estimate (step burn + workout burn)."""
    step_kcal = user.steps * 0.04
    workout_kcal = user.workout_duration_minutes * 5.0
    return max(0.0, step_kcal + workout_kcal)


def predict_calories_ml(user: UserInput, fitness_path: str) -> Optional[float]:
    """
    Train a quick RandomForestRegressor on the fitness CSV and predict
    calories burned for *user*.  Returns None when data is unavailable.
    """
    try:
        df = pd.read_csv(fitness_path)
        required = [
            "age", "gender", "weight_kg", "height_cm",
            "heart_rate_avg", "workout_duration_minutes",
            "workout_type", "calories_burned",
        ]
        if any(c not in df.columns for c in required):
            return None

        work = df.copy()
        for col in ["age", "weight_kg", "height_cm", "heart_rate_avg"]:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work["duration_minutes"] = pd.to_numeric(
            work["workout_duration_minutes"], errors="coerce"
        )
        h = work["height_cm"] / 100.0
        work["bmi"] = work["weight_kg"] / (h ** 2)
        work["calories_burned"] = pd.to_numeric(work["calories_burned"], errors="coerce")
        work = work.dropna(subset=["calories_burned"])

        num_cols = ["age", "weight_kg", "height_cm", "bmi", "heart_rate_avg", "duration_minutes"]
        cat_cols = ["gender", "workout_type"]

        pre = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    num_cols,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    cat_cols,
                ),
            ]
        )
        pipe = Pipeline([
            ("pre", pre),
            ("model", RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)),
        ])
        pipe.fit(work[num_cols + cat_cols], work["calories_burned"])

        user_row = pd.DataFrame([{
            "age": user.age,
            "weight_kg": user.weight_kg,
            "height_cm": user.height_cm,
            "bmi": user.weight_kg / ((user.height_cm / 100.0) ** 2),
            "heart_rate_avg": user.heart_rate_avg,
            "duration_minutes": user.workout_duration_minutes,
            "gender": user.gender,
            "workout_type": user.workout_type,
        }])
        return max(0.0, float(pipe.predict(user_row)[0]))
    except Exception:
        return None