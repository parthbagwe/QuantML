import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from src.preprocess import load_and_clean
from src.features import add_features

st.set_page_config(page_title="Stock Predictor", layout="wide", page_icon="📈")

# Custom dark CSS
st.markdown("""
<style>
    .stApp { background: #0a0e17; color: #c9d1e0; }
    .metric-card { background: #0d1220; border: 1px solid #1e2a3a; padding: 1rem; border-radius: 8px; }
</style>""", unsafe_allow_html=True)

st.title("QuantML · Stock Price Predictor")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Configuration")
    uploaded = st.file_uploader("Upload Kaggle CSV", type=['csv'])
    model_choice = st.selectbox("ML Model", ['XGBoost','Random Forest','Decision Tree','Linear Regression'])
    pred_days = st.slider("Forecast days", 7, 90, 30)
    run_btn = st.button("Run Prediction")

if uploaded:
    df = load_and_clean(uploaded)
    df = add_features(df)

    # Load selected model
    model_map = {
        'XGBoost': 'models/xgboost_model.pkl',
        'Random Forest': 'models/rf_model.pkl',
        'Decision Tree': 'models/dt_model.pkl',
        'Linear Regression': 'models/linear_model.pkl'
    }
    model = joblib.load(model_map[model_choice])

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    metrics = json.load(open('models/metrics.json'))
    m = metrics[model_choice.lower().replace(' ','_')]
    col1.metric("R² Score", m['R2'])
    col2.metric("RMSE", f"${m['RMSE']:.2f}")
    col3.metric("MAE", f"${m['MAE']:.2f}")
    col4.metric("Directional Accuracy", f"{m['Directional_Accuracy']*100:.1f}%")

    # Candlestick chart with Plotly
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'], open=df['open'],
        high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#22c55e',
        decreasing_line_color='#ef4444'
    )])
    fig.update_layout(
        paper_bgcolor='#0d1220', plot_bgcolor='#080c15',
        font_color='#c9d1e0', xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Actual vs predicted
    FEATURES = ['close_lag1','ma_20','ma_50','rsi_14','macd','volume_ratio',
                'bb_width','daily_return','high_low_range','ema_12']
    preds = model.predict(df[FEATURES].dropna())
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=df['close'].values[-len(preds):],
                              name='Actual', line=dict(color='#00d4aa')))
    fig2.add_trace(go.Scatter(y=preds, name='Predicted',
                              line=dict(color='#f59e0b', dash='dot')))
    fig2.update_layout(paper_bgcolor='#0d1220', plot_bgcolor='#080c15',
                       font_color='#c9d1e0')
    st.plotly_chart(fig2, use_container_width=True)