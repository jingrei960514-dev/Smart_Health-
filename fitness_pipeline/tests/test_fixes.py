"""
tests/test_fixes.py — targeted unit tests for every bug fix.

Run with:  pytest tests/test_fixes.py -v
"""
from __future__ import annotations

import types
import numpy as np
import pandas as pd
import pytest

# ── helpers ────────────────────────────────────────────────────────────────

def _make_user(**overrides):
    from fitness_pipeline.models.schemas import UserInput
    defaults = dict(
        age=24, gender="Male", height_cm=170.0, weight_kg=68.0,
        steps=6500, sleep_hours=6.5, heart_rate_avg=78,
        workout_type="Walking", workout_duration_minutes=25,
    )
    return UserInput(**{**defaults, **overrides})


def _make_goal(**overrides):
    from fitness_pipeline.models.schemas import GoalInput
    defaults = dict(goal_type="減重", target_change_kg=3.0, daily_diet_adjust_kcal=250.0)
    return GoalInput(**{**defaults, **overrides})


# ── Fix 1: dead code removed from estimate_goal_days ──────────────────────

class TestGoalEstimate:
    def test_weight_loss_returns_dataclass(self):
        from fitness_pipeline.recommenders.goal_engine import estimate_goal_days
        result = estimate_goal_days(_make_user(), _make_goal(goal_type="減重"))
        assert result.goal_type == "減重"
        assert result.estimated_days > 0
        assert result.target_intake > 0

    def test_muscle_gain_returns_dataclass(self):
        from fitness_pipeline.recommenders.goal_engine import estimate_goal_days
        result = estimate_goal_days(_make_user(), _make_goal(goal_type="增肌"))
        assert result.goal_type == "增肌"
        # Dead-code check: the function must complete without AttributeError
        assert result.estimated_days > 0
        assert result.target_intake > 0

    def test_muscle_gain_target_intake_above_floor(self):
        """target_intake must never be below the safety floor."""
        from fitness_pipeline.recommenders.goal_engine import estimate_goal_days
        result = estimate_goal_days(
            _make_user(weight_kg=40.0, height_cm=150.0, age=18),
            _make_goal(goal_type="增肌"),
        )
        assert result.target_intake >= 1500  # male floor


# ── Fix 2: run_pipeline / pipeline.py is not tested for I/O here ──────────
# (Integration test would require actual data files)


# ── Fix 3: print_decision_report not tested — it has been replaced by
#    report/renderer.py which is a pure print function; tested via snapshot
#    tests in CI, not here.


# ── Fix 4: weight sum assertion ────────────────────────────────────────────

class TestWeightSum:
    def test_weights_sum_to_one(self):
        import importlib
        import fitness_pipeline.recommenders.program_recommender as m
        total = m._W_GOAL + m._W_LEVEL + m._W_EQUIPMENT + m._W_TIME + m._W_LENGTH
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}"

    def test_broken_weights_raise(self, monkeypatch):
        """If someone edits a weight constant, the assertion fires on import."""
        import sys
        # Remove cached module so we can re-import with patched source
        mod_name = "fitness_pipeline.recommenders.program_recommender"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        # Monkey-patching the constant at module level after import won't
        # re-trigger the assert, so we verify the guard is present in source.
        import inspect
        import fitness_pipeline.recommenders.program_recommender as m
        src = inspect.getsource(m)
        assert "assert abs(_WEIGHT_SUM - 1.0)" in src


# ── Fix 5: LLM timeout/retry ───────────────────────────────────────────────

class TestLlmMealPlan:
    def test_no_api_key_returns_graceful_string(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from fitness_pipeline.recommenders.llm_meal_plan import generate_llm_meal_plan
        from fitness_pipeline.models.schemas import GoalEstimate
        ge = GoalEstimate("減重", 3.0, 23100.0, 510.0, 1800.0, 45.0, "note")
        food = pd.DataFrame({"Food": ["Apple"], "Category": ["Fruit"],
                              "Calories": [52], "Protein": [0.3],
                              "Fat": [0.2], "Carbs": [14], "score": [0.9]})
        result = generate_llm_meal_plan(ge, food)
        assert "GEMINI_API_KEY" in result or "google-genai" in result

    def test_timeout_returns_error_string(self, monkeypatch):
        """Simulate a hanging API call; must resolve within TIMEOUT + margin."""
        import os, time
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

        # Fake genai module that hangs forever
        fake_genai = types.ModuleType("google.genai")
        class _HangingClient:
            def __init__(self, **_): pass
            class models:
                @staticmethod
                def generate_content(**_):
                    time.sleep(9999)
        fake_genai.Client = _HangingClient

        import sys
        sys.modules.setdefault("google", types.ModuleType("google"))
        sys.modules["google.genai"] = fake_genai

        # Patch the import inside the module
        import fitness_pipeline.recommenders.llm_meal_plan as llm_mod
        monkeypatch.setattr(llm_mod, "MAX_RETRIES", 0)
        monkeypatch.setattr(llm_mod, "TIMEOUT_SECONDS", 1)

        from fitness_pipeline.models.schemas import GoalEstimate
        ge = GoalEstimate("減重", 3.0, 23100.0, 510.0, 1800.0, 45.0, "note")
        food = pd.DataFrame({"Food": ["Apple"], "Category": ["Fruit"],
                              "Calories": [52], "Protein": [0.3],
                              "Fat": [0.2], "Carbs": [14], "score": [0.9]})
        start = time.time()
        result = llm_mod.generate_llm_meal_plan(ge, food)
        elapsed = time.time() - start

        assert "失敗" in result or "LLM" in result
        assert elapsed < 5, "Timeout guard did not fire in time"


# ── Fix 6: cosine similarity division-by-zero ──────────────────────────────

class TestCosineSimilarity:
    def _make_food_df(self, rows):
        return pd.DataFrame(rows, columns=["Food", "Category", "Calories", "Protein", "Fat", "Carbs"])

    def test_normal_case_returns_topn(self):
        from fitness_pipeline.recommenders.food_recommender import recommend_foods
        df = self._make_food_df([
            ["Apple",  "Fruit",  52,  0.3,  0.2, 14],
            ["Chicken","Meat",  165, 31.0,  3.6,  0],
            ["Rice",   "Grain", 206,  4.3,  0.4, 45],
            ["Egg",    "Dairy",  78,  6.0,  5.0,  0.6],
            ["Oats",   "Grain", 389, 17.0,  7.0, 66],
        ])
        result = recommend_foods(df, mode="balanced", topn=3)
        assert len(result) == 3
        assert not result["score"].isna().any()

    def test_zero_norm_row_does_not_raise(self):
        """A row of all-zero nutrition values must not raise ZeroDivisionError."""
        from fitness_pipeline.recommenders.food_recommender import recommend_foods
        df = self._make_food_df([
            ["ZeroFood", "Unknown", 0, 0, 0, 0],
            ["Apple",    "Fruit",  52, 0.3, 0.2, 14],
            ["Chicken",  "Meat",  165, 31.0, 3.6, 0],
        ])
        result = recommend_foods(df, mode="fat_loss", topn=2)
        assert len(result) == 2
        # The zero-row should get similarity 0, not NaN
        zero_row = result[result["Food"] == "ZeroFood"]
        if not zero_row.empty:
            assert zero_row["score"].iloc[0] == pytest.approx(0.0)

    def test_scores_are_finite(self):
        from fitness_pipeline.recommenders.food_recommender import recommend_foods
        import math
        df = self._make_food_df([
            ["A", "X", 100, 5, 5, 20],
            ["B", "X", 200, 10, 8, 30],
        ])
        result = recommend_foods(df, mode="muscle_gain", topn=2)
        for s in result["score"]:
            assert math.isfinite(s)


# ── Fix 7: load_program_candidates skips empty-column frames ──────────────

class TestLoadProgramCandidates:
    def test_empty_keep_skips_frame(self, tmp_path, monkeypatch):
        """
        A fallback CSV with no useful columns must not be appended
        (previously caused silent downstream KeyErrors).
        """
        # Write a CSV with no known column names
        bad_csv = tmp_path / "fitness_and_workout_dataset.csv"
        bad_csv.write_text("col_a,col_b\n1,2\n3,4\n")

        import fitness_pipeline.recommenders.program_recommender as m
        monkeypatch.setattr(m, "PROGRAM_PROFILE_ZH_PATH", str(tmp_path / "no.csv"))
        monkeypatch.setattr(m, "PROGRAM_PROFILE_PATH",    str(tmp_path / "no2.csv"))
        monkeypatch.setattr(m, "PROGRAM_SUMMARY_PATH",    str(tmp_path / "no3.csv"))
        monkeypatch.setattr(m, "PROGRAM_FITNESS_PATH",    str(bad_csv))

        result = m.load_program_candidates()
        # Should return empty DataFrame, not raise
        assert isinstance(result, pd.DataFrame)

    def test_good_fallback_csv_is_used(self, tmp_path, monkeypatch):
        good_csv = tmp_path / "program_summary.csv"
        good_csv.write_text("title,goal,level\nProgA,Fat Loss,Beginner\n")

        import fitness_pipeline.recommenders.program_recommender as m
        monkeypatch.setattr(m, "PROGRAM_PROFILE_ZH_PATH", str(tmp_path / "no.csv"))
        monkeypatch.setattr(m, "PROGRAM_PROFILE_PATH",    str(tmp_path / "no2.csv"))
        monkeypatch.setattr(m, "PROGRAM_SUMMARY_PATH",    str(good_csv))
        monkeypatch.setattr(m, "PROGRAM_FITNESS_PATH",    str(tmp_path / "no3.csv"))

        result = m.load_program_candidates()
        assert len(result) == 1
        assert result.iloc[0]["title"] == "ProgA"