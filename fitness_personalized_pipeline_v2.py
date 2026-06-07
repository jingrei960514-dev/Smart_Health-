import argparse
import os
from google import genai
from dotenv import load_dotenv  # 新增這行

# 執行這行，程式就會自動去找旁邊的 .env 檔案，把密碼載入進來
load_dotenv()
from dataclasses import dataclass
from typing import Dict, List, Tuple
import unicodedata

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


FITNESS_PATH = "data/cleaned_fitness_data_v2.csv"
FOOD_PATH = "data/cleaned_nutrients_analysis.csv"
PROGRAM_SUMMARY_PATH = "data/program_summary.csv"
PROGRAM_FITNESS_PATH = "data/fitness_and_workout_dataset.csv"
PROGRAM_DETAIL_PATH = "data/programs_detailed_boostcamp_kaggle.csv"
PROGRAM_PROFILE_ZH_PATH = "data/program_profiles_zh.csv"
PROGRAM_PROFILE_PATH = "data/program_profiles.csv"


def ensure_columns(df: pd.DataFrame, defaults: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val
    return out


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
    goal_type: str
    target_change_kg: float
    daily_diet_adjust_kcal: float


def load_and_prepare_fitness(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_columns(
        df,
        {
            "age": 30,
            "gender": "Unknown",
            "height_cm": 170.0,
            "weight_kg": 65.0,
            "steps": 6000,
            "sleep_hours": 7.0,
            "heart_rate_avg": 75,
            "workout_type": "No Workout",
            "workout_duration_minutes": 0,
        },
    )
    df["workout_type"] = (
        df["workout_type"].astype(str).str.strip().replace({"nan": "No Workout", "": "No Workout"})
    )
    df.loc[df["workout_type"].isna(), "workout_type"] = "No Workout"

    # Derive BMI and activity score as robust, explainable features.
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


def build_cluster_model(df: pd.DataFrame, k: int = 4):
    cluster_features = [
        "age",
        "bmi",
        "steps",
        "sleep_hours",
        "heart_rate_avg",
        "workout_duration_minutes",
    ]
    X = df[cluster_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    df = df.copy()
    df["cluster_id"] = labels
    return df, kmeans, scaler, cluster_features, sil


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("cluster_id")[["steps", "sleep_hours", "heart_rate_avg", "workout_duration_minutes", "bmi"]]
        .mean()
        .round(2)
    )
    agg["size"] = df.groupby("cluster_id").size()
    return agg.reset_index()


def build_activity_classifier(df: pd.DataFrame):
    feature_cols_num = [
        "age",
        "height_cm",
        "weight_kg",
        "steps",
        "sleep_hours",
        "heart_rate_avg",
        "workout_duration_minutes",
        "bmi",
    ]
    feature_cols_cat = ["gender", "workout_type"]

    X = df[feature_cols_num + feature_cols_cat]
    y = df["activity_level"].astype(str)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=3,
        class_weight="balanced",
    )

    pipe = Pipeline([("pre", pre), ("model", clf)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)

    return pipe, feature_cols_num, feature_cols_cat, report


def recommend_foods(food_df: pd.DataFrame, mode: str = "balanced", topn: int = 5) -> pd.DataFrame:
    target = {
        "balanced": np.array([450, 30, 15, 45]),
        "fat_loss": np.array([380, 35, 10, 30]),
        "muscle_gain": np.array([550, 40, 18, 50]),
    }[mode]

    food_df = ensure_columns(
        food_df,
        {
            "Food": "Unknown Food",
            "Category": "Unknown Category",
            "Calories": 300.0,
            "Protein": 15.0,
            "Fat": 10.0,
            "Carbs": 35.0,
        },
    )
    nutrition_cols = ["Calories", "Protein", "Fat", "Carbs"]
    X = food_df[nutrition_cols].copy().astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    target_df = pd.DataFrame([target], columns=nutrition_cols)
    ts = scaler.transform(target_df)

    sim = (Xs @ ts.T).ravel() / (
        np.linalg.norm(Xs, axis=1) * np.linalg.norm(ts)
    )
    out = food_df.copy()
    out["score"] = sim
    return out.sort_values("score", ascending=False).head(topn)[
        ["Food", "Category", "Calories", "Protein", "Fat", "Carbs", "score"]
    ]


def localize_food_text(df: pd.DataFrame) -> pd.DataFrame:
    food_map = {
        "Peppers with beef and crumbs": "牛肉碎甜椒",
        "Flour": "麵粉",
        "Soybeans": "黃豆",
        "Wheat-germ cereal toasted": "烘烤小麥胚芽穀片",
        "Wheat germ": "小麥胚芽",
    }
    category_map = {
        "Vegetables R-Z": "蔬菜類",
        "Breads, cereals, fastfood,grains": "穀物與主食類",
        "Dairy products": "乳製品",
        "Meat, Poultry": "肉類與家禽",
        "Fats, Oils, Shortenings": "油脂類",
        "Fruits A-F": "水果類",
        "Fruits G-P": "水果類",
        "Fruits R-Z": "水果類",
    }

    out = df.copy()
    out["Food"] = out["Food"].map(food_map).fillna(out["Food"])
    out["Category"] = out["Category"].map(category_map).fillna(out["Category"])
    return out


def display_width(text: str) -> int:
    width = 0
    for ch in str(text):
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def pad_text(text: str, width: int) -> str:
    text = str(text)
    gap = max(0, width - display_width(text))
    return text + (" " * gap)


def print_aligned_table(df: pd.DataFrame) -> None:
    df_show = df.fillna("")
    headers = [str(c) for c in df_show.columns]
    rows = [[str(v) for v in row] for row in df_show.to_numpy()]
    col_widths = []
    for i, h in enumerate(headers):
        w = display_width(h)
        for r in rows:
            w = max(w, display_width(r[i]))
        col_widths.append(w)

    header_line = " | ".join(pad_text(h, col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * w for w in col_widths)
    print(header_line)
    print(sep_line)
    for r in rows:
        print(" | ".join(pad_text(v, col_widths[i]) for i, v in enumerate(r)))


def safe_get(mapping: Dict, key: str, default: str = "資料不足"):
    try:
        val = mapping[key]
        if val is None:
            return default
        return val
    except Exception:
        return default


def parse_list_like(value) -> List[str]:
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            arr = ast.literal_eval(s)
            if isinstance(arr, list):
                return [str(v).strip() for v in arr if str(v).strip()]
        except Exception:
            return [s]
    return [s]


def localize_level_text(text: str) -> str:
    mapping = {
        "Beginner": "初學",
        "Novice": "新手",
        "Intermediate": "中階",
        "Advanced": "進階",
    }
    out = str(text)
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


def localize_goal_text(text: str) -> str:
    mapping = {
        "Fat Loss": "減脂",
        "Muscle & Sculpting": "增肌塑形",
        "Bodyweight Fitness": "徒手健身",
        "Bodybuilding": "健美增肌",
        "Powerlifting": "健力",
        "Powerbuilding": "力量增肌",
        "Athletics": "體能表現",
        "Olympic Weightlifting": "奧林匹克舉重",
        "Strength": "力量提升",
        "Muscle": "增肌",
        "General Strength & Hypertrophy": "肌力與肌肥大",
        "Hypertrophy / Muscle Gain": "肌肥大與增肌",
        "Fat Loss / Conditioning": "減脂與體能",
        "Mobility": "活動度提升",
    }
    out = str(text)
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


def localize_equipment_text(text: str) -> str:
    mapping = {
        "Full Gym": "完整健身房",
        "Garage Gym": "家庭車庫健身",
        "At Home": "居家訓練",
        "No Equipment": "無器材",
        "Dumbbell": "啞鈴",
        "Barbell": "槓鈴",
        "Mixed": "混合器材",
    }
    out = str(text)
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


def localize_exercise_text(text: str) -> str:
    out = str(text)
    # Specific phrases first
    phrase_map = {
        "Leg Raise (Captain's Chair)": "羅馬椅抬腿",
        "Romanian Deadlift": "羅馬尼亞硬舉",
        "Abs Crunch (Weighted)": "負重捲腹",
        "Abs Crunch": "捲腹",
        "Incline Bench Press": "上斜臥推",
        "Bench Press": "臥推",
        "Tricep Rope Push Down": "繩索三頭肌下壓",
        "Tricep Pushdown": "三頭肌下壓",
        "Shoulder Press": "肩推",
        "Pull-Up": "引體向上",
        "Deadlift": "硬舉",
        "Face Pull": "臉拉",
        "Hammer Curl": "錘式彎舉",
        "Back Extension": "背伸展",
        "V-Up": "V字捲腹",
        "Walking Lunge": "行走弓箭步",
        "Goblet Squat": "高腳杯深蹲",
        "Squat": "深蹲",
        "Lunge": "弓箭步",
        "Dip": "雙槓撐體",
        "Row": "划船",
        "Run": "跑步",
        "Leg Raise": "抬腿",
    }
    for k, v in phrase_map.items():
        out = out.replace(k, v)

    # Parenthesis equipment translation
    out = out.replace("(Barbell)", "（槓鈴）")
    out = out.replace("(Dumbbell)", "（啞鈴）")
    out = out.replace("(Cable)", "（繩索）")
    out = out.replace("(Bodyweight)", "（徒手）")
    out = out.replace("(Machine)", "（機械式器材）")
    out = out.replace("(Assisted)", "（輔助）")
    return out


def load_program_candidates() -> pd.DataFrame:
    # Priority 1: program_profiles_zh.csv
    if os.path.exists(PROGRAM_PROFILE_ZH_PATH):
        df = pd.read_csv(PROGRAM_PROFILE_ZH_PATH)
        if "source_dataset" not in df.columns:
            df["source_dataset"] = "program_profiles_zh.csv"
        return df

    # Priority 2: program_profiles.csv
    if os.path.exists(PROGRAM_PROFILE_PATH):
        df = pd.read_csv(PROGRAM_PROFILE_PATH)
        if "source_dataset" not in df.columns:
            df["source_dataset"] = "program_profiles.csv"
        return df

    # Priority 3: fallback summary datasets
    frames = []
    for src, p in [("program_summary", PROGRAM_SUMMARY_PATH), ("fitness_and_workout", PROGRAM_FITNESS_PATH)]:
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        keep = [c for c in ["title", "goal", "level", "equipment", "program_length", "time_per_workout", "total_exercises", "description"] if c in df.columns]
        if not keep:
            continue
        part = df[keep].copy()
        part["source"] = src
        part["source_dataset"] = src
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["title", "goal", "level", "equipment", "program_length", "time_per_workout", "total_exercises", "description", "source"])

    pool = pd.concat(frames, ignore_index=True)
    pool = pool.drop_duplicates(subset=["title"], keep="first")
    return pool


def load_exercise_lookup() -> pd.DataFrame:
    if os.path.exists(PROGRAM_DETAIL_PATH):
        return pd.read_csv(PROGRAM_DETAIL_PATH, usecols=[c for c in ["title", "exercise_name"] if c in pd.read_csv(PROGRAM_DETAIL_PATH, nrows=0).columns])
    alt = "data/programs_detailed_bootcamp_kaggle.csv"
    if os.path.exists(alt):
        return pd.read_csv(alt, usecols=[c for c in ["title", "exercise_name"] if c in pd.read_csv(alt, nrows=0).columns])
    return pd.DataFrame(columns=["title", "exercise_name"])


def recommend_programs(user: UserInput, goal: GoalInput, predicted_level: str, profile_name: str, topn: int = 3) -> List[Dict[str, str]]:
    pool = load_program_candidates()
    if pool.empty:
        return []

    detail = load_exercise_lookup()
    ex_count = {}
    if not detail.empty and "title" in detail.columns:
        ex_count = detail.groupby("title").size().to_dict()

    goal_token = "fat" if goal.goal_type == "減重" else "muscle"
    level_pref = {"Low": "Beginner", "Medium": "Intermediate", "High": "Advanced"}.get(predicted_level, "Intermediate")
    if "低活動量" in profile_name:
        level_pref = "Beginner"
    elif "高活動量" in profile_name and predicted_level != "Low":
        level_pref = "Intermediate"

    target_time = 45 if predicted_level == "Low" else (60 if predicted_level == "Medium" else 75)
    target_len = 8 if goal.goal_type == "減重" else 12

    # Goal-aware weighting
    w_goal = 0.35 if goal.goal_type == "減重" else 0.30
    w_level = 0.22
    w_equipment = 0.15
    w_time = 0.18
    w_length = 0.10 if goal.goal_type == "減重" else 0.15

    scored = []
    for _, r in pool.iterrows():
        # Support both profile schema and fallback summary schema
        goals = parse_list_like(r.get("goal", r.get("primary_goal", "")))
        levels = parse_list_like(r.get("level", r.get("estimated_difficulty", "")))
        equipment = str(r.get("equipment", r.get("equipment_type", "")))
        time_pw = pd.to_numeric(r.get("time_per_workout", r.get("duration_minutes", np.nan)), errors="coerce")
        prog_len = pd.to_numeric(r.get("program_length", r.get("program_length_weeks", np.nan)), errors="coerce")
        train_days = pd.to_numeric(r.get("training_days_per_week", np.nan), errors="coerce")
        split_zh = str(r.get("training_split_type_zh", r.get("training_split_type", "資料不足")))
        muscles_zh = str(r.get("major_muscle_groups_zh", r.get("major_muscle_groups", "資料不足")))
        reps_ex_zh = str(r.get("representative_exercises_zh", r.get("representative_exercises", "資料不足")))
        reps_ex_zh = localize_exercise_text(reps_ex_zh)
        goal_zh = str(r.get("primary_goal_zh", localize_goal_text(str(r.get("primary_goal", r.get("goal", "資料不足")))))
                      )
        diff_zh = str(r.get("estimated_difficulty_zh", localize_level_text(str(r.get("estimated_difficulty", "資料不足")))))
        display_title_zh = str(r.get("display_title_zh", "個人化訓練計畫"))

        goal_score = 1.0 if any(goal_token in g.lower() for g in goals) else 0.25
        level_score = 1.0 if any(level_pref.lower() in lv.lower() for lv in levels) else 0.35
        equip_score = 1.0 if ("full gym" not in equipment.lower()) else 0.6
        time_score = 0.5 if pd.isna(time_pw) else max(0.0, 1.0 - abs(float(time_pw) - target_time) / 90.0)
        len_score = 0.5 if pd.isna(prog_len) else max(0.0, 1.0 - abs(float(prog_len) - target_len) / 16.0)

        score = (
            w_goal * goal_score
            + w_level * level_score
            + w_equipment * equip_score
            + w_time * time_score
            + w_length * len_score
        )

        reasons = []
        if goal_score >= 0.9:
            reasons.append(f"符合{goal.goal_type}目標")
        if level_score >= 0.9:
            reasons.append(f"適合 {level_pref} 程度")
        if equip_score >= 0.95:
            reasons.append("器材需求較彈性")
        if time_score >= 0.8:
            reasons.append(f"每次訓練時間接近 {target_time} 分鐘")
        if len_score >= 0.8:
            reasons.append("課表週期與目標時程相近")

        title = str(r.get("title", ""))
        user_goal_text = goal.goal_type
        reason_lines = [
            f"符合使用者目標：{user_goal_text}（課表主目標：{goal_zh}）",
            f"符合目前活動等級推估程度：{level_pref} 對應課表難度 {diff_zh}",
            f"課表週期適合作為中短期訓練計畫（{('資料不足' if pd.isna(prog_len) else f'{int(prog_len)} 週')})",
            f"訓練結構：{split_zh}，每週約 {('資料不足' if pd.isna(train_days) else int(train_days))} 天",
            f"主要肌群：{muscles_zh}",
            f"代表動作：{reps_ex_zh}",
        ]
        if pd.isna(time_pw):
            reason_lines.insert(2, "此課表缺少單次訓練時間資料，因此時間適配性僅作輔助參考。")
        else:
            reason_lines.insert(2, f"每次訓練時間接近使用者可用時間（目標 {target_time} 分鐘）")
        limit_lines = [
            "若目前使用者活動等級偏低，建議循序增加訓練量",
            "若器材不足，需替換部分器材動作",
        ]
        scored.append(
            {
                "display_title_zh": display_title_zh if display_title_zh and display_title_zh != "nan" else "個人化訓練計畫",
                "title": title,
                "goal": localize_goal_text(str(r.get("goal", "資料不足"))),
                "level": localize_level_text(str(r.get("level", diff_zh if diff_zh else "資料不足"))),
                "equipment": localize_equipment_text(equipment if equipment else "資料不足"),
                "program_length": "資料不足" if pd.isna(prog_len) else f"{float(prog_len):.0f} 週",
                "workout_duration": "資料不足" if pd.isna(time_pw) else f"{float(time_pw):.0f} 分鐘",
                "training_days_per_week": "資料不足" if pd.isna(train_days) else f"{int(train_days)} 天",
                "training_split_type_zh": split_zh,
                "major_muscle_groups_zh": muscles_zh,
                "representative_exercises_zh": reps_ex_zh,
                "primary_goal_zh": goal_zh,
                "estimated_difficulty_zh": diff_zh,
                "score": float(score),
                "reason_lines": reason_lines,
                "limit_lines": limit_lines,
                "exercise_detail_count": int(ex_count.get(title, 0)),
                "source_dataset": str(r.get("source_dataset", r.get("source", "資料不足"))),
            }
        )

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored[:topn]


def print_decision_report(
    user_df: pd.DataFrame,
    user: UserInput,
    args,
    sil: float,
    report: Dict,
    cluster_id: int,
    profile_name: str,
    cluster_reasons: List[str],
    predicted_level: str,
    top_features: List[Tuple[str, float]],
    level_reasons: List[str],
    goal_est: Dict[str, float],
    goal_mode: str,
    formula_calories: float,
    ml_calories,
    whatif: List[Dict[str, float]],
    best_case: str,
    actions: List[str],
    program_recs: List[Dict[str, str]],
    food_display: pd.DataFrame,
    food_reasons: List[str],
):
    level_map = {"Low": "低", "Medium": "中", "High": "高"}
    u = user_df.iloc[0].to_dict() if len(user_df) > 0 else {}

    print("=" * 60)
    print("可解釋個人化運動與營養決策支援系統")
    print("=" * 60)

    print("\n一、使用者輸入摘要")
    print("-" * 60)
    print(f"年齡：{safe_get(u, 'age')} 歲")
    print(f"性別：{safe_get(u, 'gender')}")
    bmi = safe_get(u, "bmi")
    print(f"BMI：{bmi if isinstance(bmi, str) else f'{float(bmi):.2f}'}")
    print(f"每日步數：{safe_get(u, 'steps')}")
    sleep = safe_get(u, "sleep_hours")
    print(f"睡眠時數：{sleep if isinstance(sleep, str) else f'{float(sleep):.1f}'} 小時")
    print(f"平均心率：{safe_get(u, 'heart_rate_avg')}")
    print(f"運動類型：{safe_get(u, 'workout_type')}")
    print(f"運動時長：{safe_get(u, 'workout_duration_minutes')} 分鐘")
    print(f"目標：{safe_get(goal_est, 'goal_type')}")
    print(f"目標變化：{safe_get(goal_est, 'target_change_kg', 0):.1f} kg")

    print("\n二、模型與方法摘要")
    print("-" * 60)
    macro_f1 = report.get("macro avg", {}).get("f1-score", "資料不足") if isinstance(report, dict) else "資料不足"
    macro_f1_str = macro_f1 if isinstance(macro_f1, str) else f"{macro_f1:.3f}"
    print(f"分群模型：KMeans，分群數 = {args.k}，Silhouette = {sil:.3f}")
    print(f"活動等級模型：Random Forest，Macro F1 = {macro_f1_str}")
    print("飲食推薦方法：Cosine Similarity")
    print("系統用途：根據生活型態、活動資料與營養資料，提供決策輔助建議。")

    print("\n三、生活型態分析")
    print("-" * 60)
    print(f"分群結果：{profile_name}（Cluster {cluster_id}）")
    print(f"活動等級：{level_map.get(predicted_level, predicted_level)}")
    print("\n分群解釋：")
    for line in cluster_reasons:
        if line.startswith("因此系統判定"):
            continue
        print(f"- {line}")

    print("\n活動等級解釋：")
    print("模型重要特徵前三名：")
    if top_features:
        for i, (name, score) in enumerate(top_features, 1):
            print(f"{i}. {map_feature_name_to_zh(name)}（重要度 {score:.3f}）")
    else:
        print("1. 資料不足")
    print("\n系統判斷：")
    summary_line = "每日步數與運動時長是影響活動等級的主要因素，因此目前被判定為低活動等級。"
    if predicted_level == "Medium":
        summary_line = "步數、睡眠與運動時長共同影響活動等級，目前判定為中活動等級。"
    if predicted_level == "High":
        summary_line = "步數與運動時長表現較佳，且恢復狀態可接受，因此判定為高活動等級。"
    print(summary_line)

    print("\n四、目標達成估算")
    print("-" * 60)
    print(f"目標：{safe_get(goal_est, 'goal_type')} {safe_get(goal_est, 'target_change_kg', 0):.1f} kg")
    if safe_get(goal_est, "goal_type") == "減重":
        print(f"估計總熱量差：約 {safe_get(goal_est, 'target_calorie_gap', 0):.0f} kcal")
        print(f"預估每日熱量差：約 {safe_get(goal_est, 'daily_calorie_gap', 0):.0f} kcal")
    print(f"預估達標時間：約 {safe_get(goal_est, 'estimated_days', 0):.0f} 天")
    print("\n提醒：")
    print(safe_get(goal_est, "note"))

    print("\n五、補充：ML 熱量預測參考")
    print("-" * 60)
    print(f"公式估算本次運動熱量：約 {formula_calories:.0f} kcal")
    if ml_calories is None:
        print("ML 模型預測本次運動熱量：資料不足，暫未啟用")
        print("兩者差異：資料不足")
    else:
        print(f"ML 模型預測本次運動熱量：約 {ml_calories:.0f} kcal")
        print(f"兩者差異：約 {abs(ml_calories - formula_calories):.0f} kcal")
    print("\n說明：")
    print("ML 預測值來自 calories prediction regression model，僅作為輔助參考。")
    print("目前達標天數仍以公式估算為主，避免模型不確定性影響主要建議。")
    print("此 ML 預測為輔助參考，非醫療或保證結果。")

    print("\n六、邊際效益分析")
    print("-" * 60)
    print(f"目前方案預估達標時間：{safe_get(goal_est, 'estimated_days', 0):.0f} 天")
    print("\n情境比較：")
    for i, s in enumerate(whatif, 1):
        short_name = s.get("name", "資料不足").replace("情境 A：", "").replace("情境 B：", "").replace("情境 C：", "")
        print(f"{i}. {short_name}")
        print(f"   新預估達標時間：{s.get('new_days', 0):.0f} 天")
        print(f"   改善：{s.get('improve_days', 0):.0f} 天\n")
    print("系統建議：")
    best_name = best_case.replace("情境 A：", "").replace("情境 B：", "").replace("情境 C：", "")
    print(f"對此使用者而言，「{best_name}」的邊際效益最大，可優先嘗試。")

    print("\n七、行動建議")
    print("-" * 60)
    for i, a in enumerate(actions, 1):
        print(f"{i}. {a}")

    print("\n八、Top 3 個人化課表推薦")
    print("-" * 60)
    if not program_recs:
        print("課表推薦資料不足")
    else:
        for i, rec in enumerate(program_recs, 1):
            print(f"推薦課表 {i}")
            print(f"課表名稱：{rec.get('display_title_zh', '個人化訓練計畫')}")
            print(f"原始課表名稱：{rec.get('title', '資料不足')}")
            print("")
            print("適合對象：")
            print(f"- {rec.get('level', '資料不足')}")
            print(f"- {rec.get('primary_goal_zh', rec.get('goal', '資料不足'))}")
            print(f"- {rec.get('equipment', '資料不足')}")
            print("")
            print("課表資訊：")
            print(f"- 課表長度：{rec.get('program_length', '資料不足')}")
            print(f"- 每週訓練天數：{rec.get('training_days_per_week', '資料不足')}")
            print(f"- 每次訓練：約 {rec.get('workout_duration', '資料不足')}")
            print(f"- 訓練方式：{rec.get('training_split_type_zh', '資料不足')}")
            print(f"- 主要訓練肌群：{rec.get('major_muscle_groups_zh', '資料不足')}")
            print(f"- 代表動作：{rec.get('representative_exercises_zh', '資料不足')}")
            print("")
            print(f"推薦分數：{rec.get('score', 0.0):.3f}")
            print("")
            print("推薦理由：")
            for rs in rec.get("reason_lines", ["資料不足"]):
                print(f"- {rs}")
            print("")
            print("限制：")
            for ls in rec.get("limit_lines", ["資料不足"]):
                print(f"- {ls}")
            print("")
            print(f"資料來源：{rec.get('source_dataset', '資料不足')}")
            print(f"可查詢動作明細：{rec.get('exercise_detail_count', 0)} 筆")
            print("")

    print("\n九、Top 5 營養輔助推薦")
    print("-" * 60)
    table_df = food_display.copy().reset_index(drop=True)
    table_df.insert(0, "排名", np.arange(1, len(table_df) + 1))
    table_df = table_df.rename(columns={"相似度分數": "相似度"})
    print_aligned_table(table_df.round(3))

    print("\n十、飲食推薦解釋")
    print("-" * 60)
    print("系統根據使用者目標營養向量與食物營養成分進行 Cosine Similarity 比對。")
    print("相似度越高，代表該食物的熱量、蛋白質、脂肪與碳水比例越接近使用者目標。")
    print(f"本次飲食推薦模式依使用者目標設定為：{goal_mode}。")
    print("\n推薦說明：")
    for line in food_reasons[2:5] if len(food_reasons) >= 5 else food_reasons:
        print(f"- {line}")
    print("\n注意：")
    print("以上為營養輔助建議，非醫療建議。")

    print("\n" + "=" * 60)


def cluster_label(row: pd.Series) -> str:
    if row["steps"] < 5000 and row["sleep_hours"] < 7:
        return "低活動量 + 低恢復"
    if row["steps"] > 9000 and row["sleep_hours"] >= 7:
        return "高活動量 + 良好恢復"
    if row["heart_rate_avg"] > 82:
        return "中等活動量 + 心肺負荷偏高"
    return "中等平衡型生活"


def generate_action_suggestions(user: UserInput, level: str) -> List[str]:
    actions = []
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


def categorize_steps(steps: float) -> str:
    if steps < 5000:
        return "低活動量"
    if steps < 9000:
        return "中等活動量"
    return "高活動量"


def categorize_sleep(hours: float) -> str:
    if hours < 7:
        return "略低於建議值"
    if hours <= 9:
        return "符合建議範圍"
    return "高於一般建議範圍"


# 以群組均值對照，產生可讀的分群說明。
def explain_cluster_assignment(user_df: pd.DataFrame, cluster_row: pd.Series, profile_name: str) -> List[str]:
    u = user_df.iloc[0]
    lines = [
        f"步數 {int(u['steps'])}，屬於{categorize_steps(float(u['steps']))}（群組均值 {cluster_row['steps']:.0f}）",
        f"睡眠 {float(u['sleep_hours']):.1f} 小時，{categorize_sleep(float(u['sleep_hours']))}（群組均值 {cluster_row['sleep_hours']:.1f}）",
        f"運動時長 {int(u['workout_duration_minutes'])} 分鐘，屬於低至中等運動量（群組均值 {cluster_row['workout_duration_minutes']:.1f}）",
        f"平均心率 {int(u['heart_rate_avg'])}（群組均值 {cluster_row['heart_rate_avg']:.1f}），BMI {float(u['bmi']):.1f}（群組均值 {cluster_row['bmi']:.1f}）",
        f"因此系統判定為「{profile_name}」。",
    ]
    return lines


def get_top_feature_importance(pipe: Pipeline, topn: int = 3) -> List[Tuple[str, float]]:
    try:
        pre = pipe.named_steps["pre"]
        model = pipe.named_steps["model"]
        feat_names = pre.get_feature_names_out()
        importances = model.feature_importances_
        pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
        return [(n, float(v)) for n, v in pairs[:topn]]
    except Exception:
        return []


def map_feature_name_to_zh(name: str) -> str:
    mapping = {
        "num__age": "年齡",
        "num__height_cm": "身高",
        "num__weight_kg": "體重",
        "num__steps": "步數",
        "num__sleep_hours": "睡眠時數",
        "num__heart_rate_avg": "平均心率",
        "num__workout_duration_minutes": "運動時長",
        "num__bmi": "BMI",
    }
    return mapping.get(name, name)


def explain_activity_level(user: UserInput, predicted_level: str, top_features: List[Tuple[str, float]]) -> List[str]:
    lines = []
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
    if predicted_level == "Low":
        lines.append("- 綜合判定結果偏向「低活動等級」")
    elif predicted_level == "Medium":
        lines.append("- 綜合判定結果偏向「中活動等級」")
    else:
        lines.append("- 綜合判定結果偏向「高活動等級」")
    return lines


def estimate_daily_exercise_kcal(user: UserInput) -> float:
    # 簡化估算：步行活動 + 訓練時長 的額外熱量消耗。
    step_kcal = user.steps * 0.04
    workout_kcal = user.workout_duration_minutes * 5.0
    return max(0.0, step_kcal + workout_kcal)


def predict_calories_ml(user: UserInput):
    """Baseline calories regression for reference only; returns None if unavailable."""
    try:
        df = pd.read_csv(FITNESS_PATH)
        req = ["age", "gender", "weight_kg", "height_cm", "heart_rate_avg", "workout_duration_minutes", "workout_type", "calories_burned"]
        if any(c not in df.columns for c in req):
            return None

        work = df.copy()
        work["age"] = pd.to_numeric(work["age"], errors="coerce")
        work["weight_kg"] = pd.to_numeric(work["weight_kg"], errors="coerce")
        work["height_cm"] = pd.to_numeric(work["height_cm"], errors="coerce")
        work["heart_rate_avg"] = pd.to_numeric(work["heart_rate_avg"], errors="coerce")
        work["duration_minutes"] = pd.to_numeric(work["workout_duration_minutes"], errors="coerce")
        h = work["height_cm"] / 100.0
        work["bmi"] = work["weight_kg"] / (h ** 2)
        work["calories_burned"] = pd.to_numeric(work["calories_burned"], errors="coerce")
        work = work.dropna(subset=["calories_burned"])

        num_cols = ["age", "weight_kg", "height_cm", "bmi", "heart_rate_avg", "duration_minutes"]
        cat_cols = ["gender", "workout_type"]
        X = work[num_cols + cat_cols]
        y = work["calories_burned"]

        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ]
        )
        model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X, y)

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


def estimate_goal_days(user: UserInput, goal: GoalInput) -> Dict[str, float]:
    goal_type = str(goal.goal_type).strip()
    if goal_type not in ("減重", "增肌"):
        goal_type = "減重"

    # 新增：1. 計算基礎代謝率 BMR (Mifflin-St Jeor 公式)
    if user.gender.lower() in ["male", "男"]:
        bmr = 10 * user.weight_kg + 6.25 * user.height_cm - 5 * user.age + 5
    else:
        bmr = 10 * user.weight_kg + 6.25 * user.height_cm - 5 * user.age - 161

    # 新增：2. 估算每日總消耗 TDEE (BMR * 久坐活動係數1.2 + 運動消耗)
    exercise_kcal = estimate_daily_exercise_kcal(user)
    tdee = (bmr * 1.2) + exercise_kcal

    if goal_type == "減重":
        total_gap = max(0.1, goal.target_change_kg) * 7700
        daily_gap = exercise_kcal + max(0.0, goal.daily_diet_adjust_kcal)
        daily_gap = max(daily_gap, 1.0)
        days = total_gap / daily_gap
        
        # 新增：3. 算出實際該吃的目標熱量
        target_intake = tdee - daily_gap
        # 加上安全底線防呆（男生不低於1500，女生不低於1200）
        min_intake = 1500 if user.gender.lower() in ["male", "男"] else 1200
        target_intake = max(min_intake, target_intake)

        note = "此為簡化估算，實際結果會受代謝、飲食執行度與身體狀況影響。"
        return {
            "goal_type": goal_type,
            "target_change_kg": goal.target_change_kg,
            "target_calorie_gap": total_gap,
            "daily_calorie_gap": daily_gap,
            "target_intake": target_intake,  # <-- 把這包進回傳字典
            "estimated_days": days,
            "note": note,
        }

    # 處理增肌的邏輯
    weekly_rate = 0.25
    days = (max(0.1, goal.target_change_kg) / weekly_rate) * 7
    target_intake = tdee + 300 # 假設增肌盈餘 300 大卡
    
    return {
        "goal_type": goal_type,
        "target_change_kg": goal.target_change_kg,
        "target_calorie_gap": 0.0,
        "daily_calorie_gap": 0.0,
        "target_intake": target_intake,  # <-- 增肌也回傳
        "estimated_days": days,
        "note": "增肌天數以每週約 0.25 kg 的保守假設估算，實際仍需依訓練與營養調整。",
    }

    weekly_rate = 0.25
    days = (max(0.1, goal.target_change_kg) / weekly_rate) * 7
    return {
        "goal_type": goal_type,
        "target_change_kg": goal.target_change_kg,
        "target_calorie_gap": 0.0,
        "daily_calorie_gap": 0.0,
        "estimated_days": days,
        "note": "增肌天數以每週約 0.25 kg 的保守假設估算，實際仍需依訓練與營養調整。",
    }


def what_if_analysis(user: UserInput, goal: GoalInput, base_days: float) -> Tuple[List[Dict[str, float]], str]:
    scenarios = []
    variants = [
        ("情境 A：每日步數 +2000", {"steps": user.steps + 2000}),
        ("情境 B：運動時間 +15 分鐘", {"workout_duration_minutes": user.workout_duration_minutes + 15}),
        ("情境 C：睡眠時間 +0.5 小時", {"sleep_hours": user.sleep_hours + 0.5}),
    ]

    for name, delta in variants:
        u2 = UserInput(**{**user.__dict__, **delta})
        est = estimate_goal_days(u2, goal)
        new_days = float(est["estimated_days"])
        scenarios.append(
            {
                "name": name,
                "new_days": new_days,
                "improve_days": max(0.0, base_days - new_days),
            }
        )

    best = max(scenarios, key=lambda x: x["improve_days"])
    return scenarios, best["name"]


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

def generate_llm_meal_plan(goal_est: Dict, food_rec: pd.DataFrame) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "系統未設定 API Key，略過 LLM 菜單生成。"

    # 【新版寫法】建立 Client 實例
    client = genai.Client(api_key=api_key)

    daily_kcal = goal_est.get("target_intake", 2000) 
    top_foods = ", ".join(food_rec["Food"].tolist())
    
    prompt = f"""
    你是一位專業的運動營養師。請根據以下條件，為使用者規劃一天的專屬健身菜單：
    - 每日目標熱量：約 {daily_kcal:.0f} 大卡
    - 系統優先推薦食材：{top_foods}（請盡量將這些食材融入菜單）
    
    請簡要輸出：
    1. 早、中、晚三餐的具體搭配
    2. 訓練前後的加餐建議
    """
    
    try:
        # 【新版寫法】呼叫 generate_content 的參數結構改變了
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"LLM 生成失敗：{e}"


def run_demo(args):
    fitness = load_and_prepare_fitness(FITNESS_PATH)
    food = pd.read_csv(FOOD_PATH)

    clustered, kmeans, scaler, cluster_features, sil = build_cluster_model(fitness, k=args.k)
    cluster_profile = summarize_clusters(clustered)

    clf, num_cols, cat_cols, report = build_activity_classifier(clustered)

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

    cX = user_df[cluster_features]
    cXs = scaler.transform(cX)
    cluster_id = int(kmeans.predict(cXs)[0])
    predicted_level = str(clf.predict(user_df[num_cols + cat_cols])[0])

    profile_row = cluster_profile[cluster_profile["cluster_id"] == cluster_id].iloc[0]
    profile_name = cluster_label(profile_row)
    cluster_reasons = explain_cluster_assignment(user_df, profile_row, profile_name)

    if goal.goal_type == "減重":
        food_mode = "fat_loss"
    elif goal.goal_type == "增肌":
        food_mode = "muscle_gain"
    else:
        food_mode = "balanced"
    food_rec = recommend_foods(food, mode=food_mode, topn=5)
    actions = generate_action_suggestions(user, predicted_level)
    top_features = get_top_feature_importance(clf, topn=3)
    level_reasons = explain_activity_level(user, predicted_level, top_features)
    goal_est = estimate_goal_days(user, goal)
    formula_calories = estimate_daily_exercise_kcal(user)
    ml_calories = predict_calories_ml(user)
    whatif, best_case = what_if_analysis(user, goal, float(goal_est["estimated_days"]))
    program_recs = recommend_programs(user, goal, predicted_level, profile_name, topn=3)
    food_reasons = explain_food_recommendation(localize_food_text(food_rec))

    input_display = user_df.rename(
        columns={
            "age": "年齡",
            "gender": "性別",
            "height_cm": "身高(cm)",
            "weight_kg": "體重(kg)",
            "steps": "步數",
            "sleep_hours": "睡眠時數",
            "heart_rate_avg": "平均心率",
            "workout_type": "運動類型",
            "workout_duration_minutes": "運動時長(分鐘)",
            "bmi": "BMI",
        }
    )
    llm_meal_plan = generate_llm_meal_plan(goal_est, food_rec)
    food_display = localize_food_text(food_rec).rename(
        columns={
            "Food": "食物",
            "Category": "類別",
            "Calories": "熱量",
            "Protein": "蛋白質",
            "Fat": "脂肪",
            "Carbs": "碳水",
            "score": "相似度分數",
        }
    )

    print_decision_report(
        user_df=user_df,
        user=user,
        args=args,
        sil=sil,
        report=report,
        cluster_id=cluster_id,
        profile_name=profile_name,
        cluster_reasons=cluster_reasons,
        predicted_level=predicted_level,
        top_features=top_features,
        level_reasons=level_reasons,
        goal_est=goal_est,
        goal_mode=food_mode,
        formula_calories=formula_calories,
        ml_calories=ml_calories,
        whatif=whatif,
        best_case=best_case,
        actions=actions,
        program_recs=program_recs,
        food_display=food_display,
        food_reasons=food_reasons,
    )
    print("\n十一、LLM 營養師專屬菜單建議")
    print("-" * 60)
    print(llm_meal_plan)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="健身個人化建議 pipeline 展示")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--age", type=int, default=24)
    p.add_argument("--gender", type=str, default="Male")
    p.add_argument("--height_cm", type=float, default=170.0)
    p.add_argument("--weight_kg", type=float, default=68.0)
    p.add_argument("--steps", type=int, default=6500)
    p.add_argument("--sleep_hours", type=float, default=6.5)
    p.add_argument("--heart_rate_avg", type=int, default=78)
    p.add_argument("--workout_type", type=str, default="Walking")
    p.add_argument("--workout_duration_minutes", type=int, default=25)
    p.add_argument("--goal_type", type=str, default="減重")
    p.add_argument("--target_change_kg", type=float, default=3.0)
    p.add_argument("--daily_diet_adjust_kcal", type=float, default=250.0)
    args = p.parse_args()

    run_demo(args)
