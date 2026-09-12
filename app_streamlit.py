import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))

from src.predict import StudentScorePredictor
from src.data_preprocessing import load_data, EDUCATION_ORDER

st.set_page_config(
    page_title="Student Score Predictor & ML Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        text-align: center;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2563EB;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎓 Student Math Score Predictor & ML Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production Machine Learning Pipeline & Interactive Dashboard</div>', unsafe_allow_html=True)

# Load predictor and dataset
@st.cache_resource
def get_predictor():
    return StudentScorePredictor()

@st.cache_data
def get_dataset():
    return load_data("StudentScore.xls")

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"⚠️ Model not loaded: {str(e)}. Please run `python train_pipeline.py` first.")
    st.stop()

df = get_dataset()

# Sidebar Navigation
st.sidebar.header("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🎯 Predictor UI", "📊 Model Benchmark & Analytics", "🔍 Exploratory Data Analysis (EDA)"])

# Page 1: Interactive Predictor UI
if page == "🎯 Predictor UI":
    st.subheader("🎯 Input Student Information & Predict Math Score")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 👤 Demographic & Academic Profile")
        gender = st.selectbox("Gender", options=["female", "male"])
        race = st.selectbox("Race / Ethnicity", options=["group A", "group B", "group C", "group D", "group E"])
        parent_edu = st.selectbox("Parental Level of Education", options=EDUCATION_ORDER)
        lunch = st.selectbox("Lunch Type", options=["standard", "free/reduced"])
        test_prep = st.selectbox("Test Preparation Course", options=["none", "completed"])

    with col2:
        st.markdown("### 📝 Current Assessment Scores")
        reading_score = st.slider("Reading Score (0 - 100)", min_value=0, max_value=100, value=70)
        writing_score = st.slider("Writing Score (0 - 100)", min_value=0, max_value=100, value=70)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Predict Math Score", use_container_width=True):
            input_dict = {
                "gender": gender,
                "race/ethnicity": race,
                "parental level of education": parent_edu,
                "lunch": lunch,
                "test preparation course": test_prep,
                "reading score": reading_score,
                "writing score": writing_score
            }
            
            result = predictor.predict_single(input_dict)
            pred_score = result["predicted_math_score"]
            rounded_score = result["rounded_math_score"]
            
            st.markdown("---")
            st.markdown("### 🏆 Prediction Result")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Predicted Score", f"{pred_score} / 100")
            res_col2.metric("Rounded Score", f"{rounded_score} / 100")
            
            avg_rw = (reading_score + writing_score) / 2.0
            diff = pred_score - avg_rw
            res_col3.metric("Math vs Avg R/W", f"{diff:+.1f} pts")
            
            if pred_score >= 80:
                st.success("🌟 Excellent performance prediction! High likelihood of honors math achievement.")
            elif pred_score >= 60:
                st.info("👍 Good performance prediction. Student is performing at grade level.")
            else:
                st.warning("⚠️ Below average score predicted. Consider providing additional math tutoring.")

# Page 2: Model Benchmark & Analytics
elif page == "📊 Model Benchmark & Analytics":
    st.subheader("📊 Machine Learning Model Comparison & Evaluation")
    
    metrics_file = os.path.join("models", "metrics_summary.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)
            
        st.markdown(f"**Winning Algorithm:** `{metrics_data.get('best_model_name', 'N/A')}`")
        
        test_metrics = metrics_data.get("test_set_metrics", {})
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Test R² Score", f"{test_metrics.get('R2', 0):.4f}")
        m_col2.metric("Mean Absolute Error (MAE)", f"{test_metrics.get('MAE', 0):.2f} pts")
        m_col3.metric("Root Mean Squared Error (RMSE)", f"{test_metrics.get('RMSE', 0):.2f} pts")
        m_col4.metric("MAPE", f"{test_metrics.get('MAPE', 0):.2f}%")
        
        st.markdown("---")
        st.markdown("### 📈 Cross-Validation Leaderboard")
        cv_df = pd.DataFrame(metrics_data.get("cv_benchmark_summary", []))
        st.dataframe(cv_df, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🖼️ Saved Visualizations")
    img_col1, img_col2 = st.columns(2)
    
    plot_comp = os.path.join("plots", "model_comparison.png")
    plot_res = os.path.join("plots", "residual_analysis.png")
    plot_fi = os.path.join("plots", "feature_importance.png")
    
    if os.path.exists(plot_comp):
        img_col1.image(plot_comp, caption="Model Comparison (CV R²)", use_column_width=True)
    if os.path.exists(plot_res):
        img_col2.image(plot_res, caption="Residual Analysis & Actual vs Predicted", use_column_width=True)
        
    if os.path.exists(plot_fi):
        st.image(plot_fi, caption="Feature Importances", use_column_width=True)

# Page 3: Exploratory Data Analysis
elif page == "🔍 Exploratory Data Analysis (EDA)":
    st.subheader("🔍 Exploratory Data Analysis of Student Scores")
    
    st.markdown("### 📊 Dataset Overview")
    st.write(f"Total Rows: **{len(df)}** | Total Features: **{len(df.columns)}**")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📈 Score Distributions & Relationships")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown("#### Score Distribution by Gender")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="gender", y="math score", palette="Set2", ax=ax)
        ax.set_title("Math Score Distribution by Gender")
        st.pyplot(fig)
        
    with eda_col2:
        st.markdown("#### Score Distribution by Lunch Type")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="lunch", y="math score", palette="Set1", ax=ax)
        ax.set_title("Math Score Distribution by Lunch Type")
        st.pyplot(fig)
        
    st.markdown("#### Reading vs Writing vs Math Score Correlation")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[["math score", "reading score", "writing score"]].corr(), annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)
