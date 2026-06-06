"""
utils/helpers.py — pure utility functions with no side effects.
"""
from __future__ import annotations

import unicodedata
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_width(text: str) -> int:
    """Return the terminal display width of *text* (CJK chars count as 2)."""
    width = 0
    for ch in str(text):
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def pad_text(text: str, width: int) -> str:
    text = str(text)
    gap = max(0, width - display_width(text))
    return text + (" " * gap)


def print_aligned_table(df: "pd.DataFrame") -> None:  # noqa: F821
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


def safe_get(mapping: Dict, key: str, default: Any = "資料不足") -> Any:
    try:
        val = mapping[key]
        return default if val is None else val
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Localization maps
# ---------------------------------------------------------------------------

_FOOD_MAP: Dict[str, str] = {
    "Peppers with beef and crumbs": "牛肉碎甜椒",
    "Flour": "麵粉",
    "Soybeans": "黃豆",
    "Wheat-germ cereal toasted": "烘烤小麥胚芽穀片",
    "Wheat germ": "小麥胚芽",
}

_CATEGORY_MAP: Dict[str, str] = {
    "Vegetables R-Z": "蔬菜類",
    "Breads, cereals, fastfood,grains": "穀物與主食類",
    "Dairy products": "乳製品",
    "Meat, Poultry": "肉類與家禽",
    "Fats, Oils, Shortenings": "油脂類",
    "Fruits A-F": "水果類",
    "Fruits G-P": "水果類",
    "Fruits R-Z": "水果類",
}

_LEVEL_MAP: Dict[str, str] = {
    "Beginner": "初學",
    "Novice": "新手",
    "Intermediate": "中階",
    "Advanced": "進階",
}

_GOAL_MAP: Dict[str, str] = {
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

_EQUIPMENT_MAP: Dict[str, str] = {
    "Full Gym": "完整健身房",
    "Garage Gym": "家庭車庫健身",
    "At Home": "居家訓練",
    "No Equipment": "無器材",
    "Dumbbell": "啞鈴",
    "Barbell": "槓鈴",
    "Mixed": "混合器材",
}

_EXERCISE_PHRASE_MAP: Dict[str, str] = {
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

_EXERCISE_EQUIPMENT_MAP: Dict[str, str] = {
    "(Barbell)": "（槓鈴）",
    "(Dumbbell)": "（啞鈴）",
    "(Cable)": "（繩索）",
    "(Bodyweight)": "（徒手）",
    "(Machine)": "（機械式器材）",
    "(Assisted)": "（輔助）",
}

_FEATURE_NAME_ZH: Dict[str, str] = {
    "num__age": "年齡",
    "num__height_cm": "身高",
    "num__weight_kg": "體重",
    "num__steps": "步數",
    "num__sleep_hours": "睡眠時數",
    "num__heart_rate_avg": "平均心率",
    "num__workout_duration_minutes": "運動時長",
    "num__bmi": "BMI",
}


def _replace_all(text: str, mapping: Dict[str, str]) -> str:
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def localize_food_text(df: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
    out = df.copy()
    out["Food"] = out["Food"].map(_FOOD_MAP).fillna(out["Food"])
    out["Category"] = out["Category"].map(_CATEGORY_MAP).fillna(out["Category"])
    return out


def localize_level_text(text: str) -> str:
    return _replace_all(str(text), _LEVEL_MAP)


def localize_goal_text(text: str) -> str:
    return _replace_all(str(text), _GOAL_MAP)


def localize_equipment_text(text: str) -> str:
    return _replace_all(str(text), _EQUIPMENT_MAP)


def localize_exercise_text(text: str) -> str:
    out = _replace_all(str(text), _EXERCISE_PHRASE_MAP)
    return _replace_all(out, _EXERCISE_EQUIPMENT_MAP)


def map_feature_name_to_zh(name: str) -> str:
    return _FEATURE_NAME_ZH.get(name, name)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

import ast  # noqa: E402 (stdlib; intentional late import to keep top clean)


def parse_list_like(value: Any) -> List[str]:
    if value is None:
        return []
    import pandas as pd  # local import to keep module lightweight
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = ast.literal_eval(s)
            if isinstance(arr, list):
                return [str(v).strip() for v in arr if str(v).strip()]
        except Exception:
            return [s]
    return [s]