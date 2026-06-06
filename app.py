import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── 從新套件 import ──────────────────────────────────────────────────────────
from fitness_pipeline.models.schemas import GoalInput, UserInput
from fitness_pipeline.models.fitness_models import (
    build_activity_classifier,
    build_cluster_model,
    cluster_label,
    explain_cluster_assignment,
    load_and_prepare_fitness,
    summarize_clusters,
    NUM_FEATURES,
    CAT_FEATURES,
    CLUSTER_FEATURES,
)
from fitness_pipeline.recommenders.food_recommender import (
    recommend_foods,
    build_food_display,
)
from fitness_pipeline.recommenders.goal_engine import estimate_goal_days
from fitness_pipeline.recommenders.llm_meal_plan import generate_llm_meal_plan
from fitness_pipeline.recommenders.program_recommender import recommend_programs

# ── 資料路徑（對應你目前的結構：data/ 在根目錄）───────────────────────────────
_BASE        = os.path.join(os.path.dirname(__file__), "data")
FITNESS_PATH = os.path.join(_BASE, "cleaned_fitness_data_v2.csv")
FOOD_PATH    = os.path.join(_BASE, "cleaned_nutrients_analysis.csv")

# ── 網頁基本設定 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="個人化健身與營養決策系統",
    page_icon="🏋️‍♂️",
    layout="wide",
)

st.title("🏋️‍♂️ AI 健身與營養決策支援系統")
st.markdown("這套系統將根據您的生活型態、活動數據與目標，提供可解釋的訓練建議與專屬菜單。")

# ── 側邊欄：使用者輸入 ────────────────────────────────────────────────────────
st.sidebar.header("👤 你的個人資料")
age            = st.sidebar.number_input("年齡", min_value=10, max_value=100, value=24)
gender         = st.sidebar.selectbox("性別", ["Male", "Female"])
height_cm      = st.sidebar.number_input("身高 (cm)", value=170.0)
weight_kg      = st.sidebar.number_input("體重 (kg)", value=68.0)

st.sidebar.header("📊 活動與生活數據")
steps              = st.sidebar.slider("每日步數", 0, 30000, 6500, step=500)
sleep_hours        = st.sidebar.slider("睡眠時數", 0.0, 12.0, 6.5, step=0.5)
heart_rate_avg     = st.sidebar.number_input("平均心率", value=78)
workout_type       = st.sidebar.selectbox("運動類型", ["Walking", "Weightlifting", "Running", "No Workout"])
workout_duration   = st.sidebar.slider("運動時長 (分鐘)", 0, 180, 25, step=5)

st.sidebar.header("🎯 健身目標")
goal_type              = st.sidebar.radio("目標", ["減重", "增肌"])
target_change_kg       = st.sidebar.number_input("目標變化量 (kg)", value=3.0, step=0.5)
daily_diet_adjust_kcal = st.sidebar.number_input("每日飲食調整預期 (kcal)", value=250.0, step=50.0)

# ── 執行按鈕 ──────────────────────────────────────────────────────────────────
if st.sidebar.button("產生個人化分析報告", type="primary"):

    with st.spinner("模型運算與資料分析中，請稍候..."):

        # 1. 封裝 dataclass
        user = UserInput(
            age=age, gender=gender, height_cm=height_cm, weight_kg=weight_kg,
            steps=steps, sleep_hours=sleep_hours, heart_rate_avg=heart_rate_avg,
            workout_type=workout_type, workout_duration_minutes=workout_duration,
        )
        goal = GoalInput(
            goal_type=goal_type,
            target_change_kg=target_change_kg,
            daily_diet_adjust_kcal=daily_diet_adjust_kcal,
        )

        # 2. 讀取資料與訓練模型
        fitness = load_and_prepare_fitness(FITNESS_PATH)
        food_df = pd.read_csv(FOOD_PATH)

        clustered, kmeans, scaler, _feats, sil = build_cluster_model(fitness, k=4)
        cluster_profile = summarize_clusters(clustered)
        clf, _num, _cat, report = build_activity_classifier(clustered)

        # 3. 建立使用者 DataFrame
        user_df = pd.DataFrame([user.__dict__])
        user_df["bmi"] = user_df["weight_kg"] / ((user_df["height_cm"] / 100.0) ** 2)

        # 4. 分群與活動等級預測
        cXs = scaler.transform(user_df[CLUSTER_FEATURES])
        cluster_id = int(kmeans.predict(cXs)[0])
        predicted_level = str(clf.predict(user_df[NUM_FEATURES + CAT_FEATURES])[0])

        level_map_zh = {"Low": "低活動量", "Medium": "中活動量", "High": "高活動量"}
        predicted_level_zh = level_map_zh.get(predicted_level, predicted_level)

        profile_row  = cluster_profile[cluster_profile["cluster_id"] == cluster_id].iloc[0]
        profile_name = cluster_label(profile_row)

        # 5. 推薦與估算
        food_mode_map = {"減重": "fat_loss", "增肌": "muscle_gain"}
        food_mode = food_mode_map.get(goal.goal_type, "balanced")

        food_rec     = recommend_foods(food_df, mode=food_mode, topn=5)
        food_display = build_food_display(food_rec)          # 已幫你 rename 欄位
        goal_est     = estimate_goal_days(user, goal)        # 回傳 GoalEstimate dataclass
        program_recs = recommend_programs(user, goal, predicted_level, profile_name, topn=3)
        llm_meal_plan = generate_llm_meal_plan(goal_est, food_rec)

    # ── 結果呈現 ──────────────────────────────────────────────────────────────
    st.divider()

    # 一、生活型態分析
    st.header("一、生活型態與活動分析")
    col1, col2, col3 = st.columns(3)
    col1.metric("判定活動等級", predicted_level_zh)
    col2.metric("生活型態分群", profile_name)
    col3.metric("BMI 指數", f"{user_df['bmi'].iloc[0]:.1f}")

    # 二、目標達成預估
    # ⚠️ goal_est 現在是 dataclass，用 . 取屬性，不是 .get()
    st.header("二、目標達成預估")
    st.info(f"預估達標時間：約 **{goal_est.estimated_days:.0f}** 天")
    st.success(f"建議每日目標攝取熱量：約 **{goal_est.target_intake:.0f}** 大卡")
    st.caption(goal_est.note)

    # 三、營養推薦
    st.header("三、營養輔助推薦清單")
    st.dataframe(food_display.round(3), use_container_width=True, hide_index=True)
    st.caption("以上表格依據 Cosine Similarity 計算得出，相似度越高代表營養配比越接近您的目標。")

    # 四、課表推薦
    # ⚠️ program_recs 現在是 ProgramRec dataclass list，用 . 取屬性，不是 .get()
    st.header("四、Top 3 個人化課表推薦")
    if not program_recs:
        st.warning("目前資料庫中沒有適合的課表。")
    else:
        for i, rec in enumerate(program_recs, 1):
            with st.expander(f"推薦課表 {i}：{rec.display_title_zh}"):
                st.write(f"**適合對象：** {rec.level} / {rec.primary_goal_zh} / {rec.equipment}")
                st.write(f"**課表週期：** {rec.program_length}（每週 {rec.training_days_per_week}）")
                st.write(f"**主要肌群：** {rec.major_muscle_groups_zh}")
                st.write(f"**代表動作：** {rec.representative_exercises_zh}")
                st.markdown("---")
                st.write("**推薦理由：**")
                for reason in rec.reason_lines:
                    st.write(f"- {reason}")
                st.write("**限制：**")
                for limit in rec.limit_lines:
                    st.write(f"- {limit}")

    # 五、LLM 菜單
    st.header("五、LLM 營養師專屬菜單")
    st.markdown(llm_meal_plan)