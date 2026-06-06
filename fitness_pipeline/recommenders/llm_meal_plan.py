"""
recommenders/llm_meal_plan.py — Gemini-backed meal plan generation.

Fixes applied:
- timeout: wraps the API call in a threading.Timer so the whole program
  never hangs indefinitely on a network problem.
- retry: up to MAX_RETRIES attempts with exponential back-off before
  returning a graceful error string.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# 呼叫此函式，它會自動尋找專案根目錄的 .env 檔並載入變數
load_dotenv()

from fitness_pipeline.models.schemas import GoalEstimate

MAX_RETRIES = 2
TIMEOUT_SECONDS = 20
BACKOFF_BASE = 1.5   # wait = BACKOFF_BASE ** attempt seconds


def generate_llm_meal_plan(
    goal_est: GoalEstimate,
    food_rec: pd.DataFrame,
) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "系統未設定 GEMINI_API_KEY，略過 LLM 菜單生成。"

    try:
        from google import genai  # optional dependency
    except ImportError:
        return "google-genai 套件未安裝，略過 LLM 菜單生成。"

    client = genai.Client(api_key=api_key)
    top_foods = ", ".join(food_rec["Food"].tolist())
    prompt = (
        f"你是一位專業的運動營養師。請根據以下條件，為使用者規劃一天的專屬健身菜單：\n"
        f"- 每日目標熱量：約 {goal_est.target_intake:.0f} 大卡\n"
        f"- 系統優先推薦食材：{top_foods}（請盡量將這些食材融入菜單）\n\n"
        f"請簡要輸出：\n"
        f"1. 早、中、晚三餐的具體搭配\n"
        f"2. 訓練前後的加餐建議\n"
    )

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        result: list[Optional[str]] = [None]
        error_box: list[Optional[Exception]] = [None]

        def _call() -> None:
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                result[0] = resp.text
            except Exception as exc:          # noqa: BLE001
                error_box[0] = exc

        import threading
        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=TIMEOUT_SECONDS)

        if t.is_alive():
            # Thread is still blocked — treat as timeout
            last_error = TimeoutError(
                f"Gemini API 未在 {TIMEOUT_SECONDS} 秒內回應（第 {attempt + 1} 次嘗試）"
            )
        elif error_box[0] is not None:
            last_error = error_box[0]
        else:
            return result[0] or "（LLM 回應為空）"

        # Back-off before retry
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_BASE ** attempt)

    return f"LLM 菜單生成失敗（已重試 {MAX_RETRIES} 次）：{last_error}"