"""
streamlit_app.py
================
Complete 5-tab Streamlit frontend.

Tabs:
  1. Dashboard    — upload, KPIs, candlestick, actual vs predicted
  2. Model Compare — all 4 models side by side
  3. Analytics    — MA, MACD, RSI, heatmap, volume
  4. Forecast     — N-day price forecast with confidence bands
  5. About        — architecture + deployment guide
"""

import io, json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="QuantML · Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #0a0e17; color: #c9d1e0; }
.block-container { padding: 1.5rem 2rem; }
section[data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid #1e2a3a;
}
[data-testid="metric-container"] {
    background: #0d1220;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] { color: #4a5a6a !important; font-size: 11px; }
[data-testid="stMetricValue"] { color: #e2e8f4 !important; font-size: 22px; }
.stTabs [data-baseweb="tab-list"] {
    background: #0d1220;
    border-bottom: 1px solid #1e2a3a;
}
.stTabs [data-baseweb="tab"] { color: #4a5a6a; font-size: 13px; }
.stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa !important;
    background: transparent !important;
}
.stButton > button {
    background: #00d4aa; color: #041511;
    border: none; border-radius: 6px; font-weight: 500;
}
.stButton > button:hover { background: #00bfa0; }
hr { border-color: #1e2a3a; }
code { background: #111827 !important; color: #00d4aa !important; }
pre  { background: #080c15 !important; color: #00d4aa !important; }
</style>
""", unsafe_allow_html=True)

# Plotly dark theme applied to every chart
PL = dict(
    paper_bgcolor="#0d1220", plot_bgcolor="#080c15",
    font_color="#c9d1e0",
    xaxis=dict(gridcolor="#1e2a3a", zerolinecolor="#1e2a3a"),
    yaxis=dict(gridcolor="#1e2a3a", zerolinecolor="#1e2a3a"),
    legend=dict(bgcolor="#0d1220", bordercolor="#1e2a3a", borderwidth=1),
    margin=dict(l=50, r=30, t=40, b=40),
)

COLOURS = {
    "xgboost":           "#00d4aa",
    "random_forest":     "#3b82f6",
    "decision_tree":     "#f59e0b",
    "linear_regression": "#6b7a8a",
    "actual":            "#e2e8f4",
}

# Session state defaults
for key in ["df", "last_result", "last_model", "all_results", "forecast"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ── Cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def process_file(file_bytes: bytes, name: str) -> pd.DataFrame:
    import tempfile
    from src.preprocess import load_and_clean
    from src.features import add_features
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.write(file_bytes)
    tmp.close()
    df = load_and_clean(tmp.name)
    os.unlink(tmp.name)
    return add_features(df)


# ── Chart builders ────────────────────────────────────────────────────────────

def chart_candlestick(df, n=90):
    sub = df.tail(n)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=sub["date"], open=sub["open"], high=sub["high"],
        low=sub["low"], close=sub["close"],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#166534", decreasing_fillcolor="#7f1d1d",
        name="OHLC", showlegend=False,
    ), row=1, col=1)
    for col, colour, dash, lbl in [
        ("ma_20","#00d4aa","solid","MA-20"),
        ("ma_50","#f59e0b","dash","MA-50"),
        ("ma_200","#8b5cf6","dot","MA-200"),
    ]:
        if col in sub.columns and sub[col].notna().any():
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub[col], name=lbl,
                line=dict(color=colour, width=1.2, dash=dash), opacity=0.9,
            ), row=1, col=1)
    vol_c = ["#166534" if c >= o else "#7f1d1d"
             for c, o in zip(sub["close"], sub["open"])]
    fig.add_trace(go.Bar(
        x=sub["date"], y=sub["volume"], name="Volume",
        marker_color=vol_c, showlegend=False,
    ), row=2, col=1)
    fig.update_layout(**PL, xaxis_rangeslider_visible=False,
                      title="Candlestick + Volume", height=480)
    return fig


def chart_actual_vs_predicted(result, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result["dates"], y=result["actual"],
        name="Actual", line=dict(color="#e2e8f4", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=result["dates"], y=result["predicted"],
        name=f"Predicted ({model_name})",
        line=dict(color=COLOURS.get(model_name, "#00d4aa"), width=1.5, dash="dot"),
    ))
    fig.update_layout(**PL, title="Actual vs Predicted Close Price", height=360)
    return fig


def chart_model_comparison(all_results):
    names  = list(all_results.keys())
    labels = [n.replace("_"," ").title() for n in names]
    cols   = [COLOURS.get(n,"#888") for n in names]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["R² Score","RMSE","MAE"])
    for col_idx, key in enumerate(["R2","RMSE","MAE"], 1):
        vals = [all_results[n]["metrics"][key] for n in names]
        fig.add_trace(go.Bar(
            x=labels, y=vals, marker_color=cols, showlegend=False,
            text=[f"{v:.3f}" for v in vals], textposition="outside",
        ), row=1, col=col_idx)
    fig.update_layout(**PL, height=380, title="Model Performance Comparison")
    return fig


def chart_forecast(fc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc["dates"] + fc["dates"][::-1],
        y=fc["upper"] + fc["lower"][::-1],
        fill="toself", fillcolor="rgba(0,212,170,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence band",
    ))
    fig.add_trace(go.Scatter(
        x=fc["dates"], y=fc["forecast"], name="Forecast",
        line=dict(color="#00d4aa", width=2, dash="dash"),
    ))
    trend_sym = "▲" if fc["trend"] == "Bullish" else "▼"
    fig.update_layout(
        **PL,
        title=f"{len(fc['dates'])}-Day Forecast  {trend_sym} {fc['trend']} "
              f"({fc['price_change_pct']:+.2f}%)",
        height=380,
    )
    return fig


def chart_moving_averages(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"], name="Close",
        line=dict(color="rgba(226,232,244,0.5)", width=0.8),
    ))
    for col, colour, dash, lbl in [
        ("ma_20","#00d4aa","solid","MA-20"),
        ("ma_50","#f59e0b","dash","MA-50"),
        ("ma_200","#8b5cf6","dot","MA-200"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col], name=lbl,
                line=dict(color=colour, width=1.4, dash=dash),
            ))
    fig.update_layout(**PL, title="Moving Average Analysis", height=360)
    return fig


def chart_rsi(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["rsi_14"], name="RSI-14",
        line=dict(color="#f59e0b", width=1.4),
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                  annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e",
                  annotation_text="Oversold (30)")
    fig.update_layout(**PL, title="RSI-14", height=260, yaxis_range=[0,100])
    return fig


def chart_macd(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["macd"], name="MACD",
        line=dict(color="#00d4aa", width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["macd_signal"], name="Signal",
        line=dict(color="#f59e0b", width=1.2, dash="dot"),
    ))
    fig.add_trace(go.Bar(
        x=df["date"], y=df["macd_hist"], name="Histogram",
        marker_color=["#22c55e" if v >= 0 else "#ef4444"
                      for v in df["macd_hist"]],
        opacity=0.7,
    ))
    fig.update_layout(**PL, title="MACD", height=260)
    return fig


def chart_heatmap(df):
    cols = [c for c in [
        "close","open","high","low","volume","ma_20","ma_50",
        "rsi_14","macd","bb_width","daily_return","volume_ratio","close_lag1",
    ] if c in df.columns]
    corr = df[cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=cols, y=cols, colorscale="RdYlGn", zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text}",
        textfont=dict(size=9), colorbar=dict(thickness=12),
    ))
    fig.update_layout(**PL, title="Feature Correlation Matrix", height=480)
    return fig


def chart_volume(df):
    fig = go.Figure(go.Bar(
        x=df["date"], y=df["volume"],
        marker_color=["#00d4aa" if c >= o else "#ef4444"
                      for c, o in zip(df["close"], df["open"])],
        opacity=0.8, name="Volume",
    ))
    if "volume_ma20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["volume_ma20"], name="Volume MA-20",
            line=dict(color="#f59e0b", width=1.4),
        ))
    fig.update_layout(**PL, title="Volume Analysis", height=300)
    return fig


def chart_feature_importance(model, feature_cols, model_name):
    import numpy as np
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return None
    top = 15
    idx   = np.argsort(imp)[-top:]
    feats = [feature_cols[i].replace("_"," ") for i in idx]
    vals  = imp[idx]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker=dict(color=vals, colorscale=[[0,"#1a3a2a"],[1,"#00d4aa"]]),
        text=[f"{v:.4f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        **PL,
        title=f"Feature Importance — {model_name.replace('_',' ').title()}",
        height=460,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### QuantML")
    st.markdown("*Stock Price Predictor*")
    st.divider()

    uploaded   = st.file_uploader("Upload Kaggle CSV", type=["csv"])
    use_sample = st.button("Load AAPL sample (yfinance)")
    st.divider()

    model_name    = st.selectbox(
        "ML Model",
        ["xgboost","random_forest","decision_tree","linear_regression"],
        format_func=lambda x: x.replace("_"," ").title(),
    )
    forecast_days = st.slider("Forecast days", 7, 90, 30, step=7)
    run_btn       = st.button("▶  Run Prediction", use_container_width=True)

    st.divider()
    st.caption("Train models first:")
    st.code("python run_training.py")
    st.caption("Then launch:")
    st.code("streamlit run streamlit_app.py")


# ── Data loading ──────────────────────────────────────────────────────────────

if uploaded is not None:
    with st.spinner("Processing..."):
        df = process_file(uploaded.read(), uploaded.name)
        st.session_state.df = df
        st.sidebar.success(f"Loaded {len(df):,} rows")

elif use_sample:
    with st.spinner("Downloading AAPL via yfinance..."):
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            raw = yf.download(
                "AAPL",
                start=(datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d"),
                end=datetime.today().strftime("%Y-%m-%d"),
                auto_adjust=True, progress=False,
            )
            raw.reset_index(inplace=True)
            raw.columns = [str(c).lower() for c in raw.columns]
            buf = io.StringIO()
            raw.to_csv(buf, index=False)
            buf.seek(0)
            from src.preprocess import load_and_clean
            from src.features import add_features
            df = add_features(load_and_clean(buf))
            st.session_state.df = df
            st.sidebar.success(f"AAPL loaded: {len(df):,} rows")
        except Exception as e:
            st.sidebar.error(f"Download failed: {e}")

df = st.session_state.df


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🏆 Model Compare",
    "📈 Analytics",
    "🔮 Forecast",
    "ℹ️ About",
])


# ════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════
with tab1:
    if df is None:
        st.info("Upload a Kaggle CSV or click **Load AAPL sample** in the sidebar.")
        st.markdown("""
**Supported columns:** Date, Open, High, Low, Close, Adj Close, Volume

**Free datasets:**
- [Kaggle NIFTY-50](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data)
- [Kaggle S&P 500](https://www.kaggle.com/datasets/camnugent/sandp500)
        """)
    else:
        from src.preprocess import get_summary_stats
        s = get_summary_stats(df)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Latest Price",  f"${s['price_latest']:,.2f}")
        c2.metric("Total Return",  f"{s['price_change_pct']:+.1f}%",
                  delta=f"{s['price_change_pct']:+.1f}%")
        c3.metric("Price Range",   f"${s['price_min']:,.0f}–${s['price_max']:,.0f}")
        c4.metric("Data Points",   f"{s['rows']:,}")
        c5.metric("Period",        f"{s['date_start'][:7]}→{s['date_end'][:7]}")

        st.plotly_chart(chart_candlestick(df), use_container_width=True)
        st.divider()

        if run_btn:
            with st.spinner(f"Running {model_name}..."):
                try:
                    from src.predict import predict_test_set
                    result = predict_test_set(df, model_name)
                    st.session_state.last_result = result
                    st.session_state.last_model  = model_name
                except FileNotFoundError:
                    st.error("Models not found. Run `python run_training.py` first.")

        result = st.session_state.last_result
        if result:
            m = result["metrics"]
            st.subheader("Model Performance")
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("R² Score",       f"{m['R2']:.4f}")
            c2.metric("RMSE",           f"${m['RMSE']:.2f}")
            c3.metric("MAE",            f"${m['MAE']:.2f}")
            c4.metric("MAPE",           f"{m['MAPE']:.2f}%")
            c5.metric("Dir. Accuracy",  f"{m['Directional_Accuracy']:.1f}%")
            st.plotly_chart(
                chart_actual_vs_predicted(result, st.session_state.last_model),
                use_container_width=True,
            )


# ════════════════════════════════════════
# TAB 2 — MODEL COMPARE
# ════════════════════════════════════════
with tab2:
    if df is None:
        st.info("Upload a dataset first.")
    else:
        if st.button("Compare all 4 models"):
            with st.spinner("Running all models..."):
                try:
                    from src.predict import all_models_predict_test
                    st.session_state.all_results = all_models_predict_test(df)
                except Exception as e:
                    st.error(str(e))

        all_r = st.session_state.all_results
        if all_r:
            st.plotly_chart(chart_model_comparison(all_r), use_container_width=True)

            # Metrics table
            rows = []
            for name, r in all_r.items():
                m = r["metrics"]
                rows.append({
                    "Model":            name.replace("_"," ").title(),
                    "R²":               m["R2"],
                    "RMSE":             m["RMSE"],
                    "MAE":              m["MAE"],
                    "MAPE (%)":         m["MAPE"],
                    "Dir. Accuracy (%)":m["Directional_Accuracy"],
                })
            st.dataframe(
                pd.DataFrame(rows).set_index("Model"),
                use_container_width=True,
            )

            # All models overlaid
            fig = go.Figure()
            first = list(all_r.values())[0]
            fig.add_trace(go.Scatter(
                x=first["dates"], y=first["actual"],
                name="Actual", line=dict(color="#e2e8f4", width=1.5),
            ))
            for name, r in all_r.items():
                fig.add_trace(go.Scatter(
                    x=r["dates"], y=r["predicted"],
                    name=name.replace("_"," ").title(),
                    line=dict(color=COLOURS.get(name,"#888"),
                              width=1.2, dash="dot"),
                ))
            fig.update_layout(**PL, height=400, title="All Models vs Actual")
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance for best model
            best = max(all_r, key=lambda k: all_r[k]["metrics"]["R2"])
            try:
                from src.train import load_model, load_feature_columns
                bm   = load_model(best)
                fcols = [f for f in load_feature_columns() if f in df.columns]
                fi   = chart_feature_importance(bm, fcols, best)
                if fi:
                    st.plotly_chart(fi, use_container_width=True)
            except Exception:
                pass


# ════════════════════════════════════════
# TAB 3 — ANALYTICS
# ════════════════════════════════════════
with tab3:
    if df is None:
        st.info("Upload a dataset first.")
    else:
        st.plotly_chart(chart_moving_averages(df), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if "rsi_14" in df.columns:
                st.plotly_chart(chart_rsi(df), use_container_width=True)
        with col2:
            if "macd" in df.columns:
                st.plotly_chart(chart_macd(df), use_container_width=True)

        st.plotly_chart(chart_heatmap(df), use_container_width=True)
        st.plotly_chart(chart_volume(df),  use_container_width=True)


# ════════════════════════════════════════
# TAB 4 — FORECAST
# ════════════════════════════════════════
with tab4:
    if df is None:
        st.info("Upload a dataset first.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("Generate Forecast"):
                with st.spinner(f"Forecasting {forecast_days} days..."):
                    try:
                        from src.predict import forecast_n_days
                        fc = forecast_n_days(df, model_name, forecast_days)
                        st.session_state.forecast = fc
                    except FileNotFoundError:
                        st.error("Models not found. Run run_training.py first.")

            fc = st.session_state.forecast
            if fc:
                st.plotly_chart(chart_forecast(fc), use_container_width=True)
                days_show = min(7, len(fc["dates"]))
                st.subheader(f"Next {days_show} days")
                st.dataframe(pd.DataFrame({
                    "Date":       fc["dates"][:days_show],
                    "Forecast":   [f"${p:,.2f}" for p in fc["forecast"][:days_show]],
                    "Low (95%)":  [f"${p:,.2f}" for p in fc["lower"][:days_show]],
                    "High (95%)": [f"${p:,.2f}" for p in fc["upper"][:days_show]],
                }), use_container_width=True, hide_index=True)

        with col2:
            fc = st.session_state.forecast
            if fc:
                st.metric("Starting price",
                          f"${fc['last_known_price']:,.2f}")
                st.metric(f"Day-{forecast_days} forecast",
                          f"${fc['forecast'][-1]:,.2f}",
                          delta=f"{fc['price_change_pct']:+.2f}%")
                st.metric("Trend", fc["trend"])
                if "rsi_14" in df.columns:
                    rsi = df["rsi_14"].iloc[-1]
                    st.metric("RSI-14", f"{rsi:.1f}",
                              delta="Overbought" if rsi > 70 else
                                    "Oversold"   if rsi < 30 else "Neutral")
                if "macd" in df.columns:
                    sig = "BUY" if df["macd"].iloc[-1] > df["macd_signal"].iloc[-1] else "SELL"
                    st.metric("MACD Signal", sig)


# ════════════════════════════════════════
# TAB 5 — ABOUT
# ════════════════════════════════════════
with tab5:
    st.markdown("""
## QuantML · Stock Price Predictor

### Project structure

    stock-predictor/
    ├── data/raw/              ← Kaggle CSV uploads
    ├── data/processed/        ← cleaned + featured data
    ├── src/
    │   ├── preprocess.py      ← cleaning, normalization
    │   ├── features.py        ← 28 technical indicators
    │   ├── train.py           ← TimeSeriesSplit training
    │   ├── evaluate.py        ← metrics + matplotlib charts
    │   └── predict.py         ← inference + N-day forecast
    ├── models/                ← saved .pkl files + metrics.json
    ├── app/
    │   ├── main.py            ← FastAPI REST API
    │   └── schemas.py         ← Pydantic validation
    ├── streamlit_app.py       ← this file
    ├── run_training.py        ← one-command training
    └── requirements.txt

### Commands

    # Install
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

    # Train models
    python run_training.py
    python run_training.py --csv myfile.csv

    # Run dashboard
    streamlit run streamlit_app.py

    # Run API
    uvicorn app.main:app --reload

### Deploy to Streamlit Cloud

    git init
    git add .
    git commit -m "feat: stock predictor"
    git remote add origin https://github.com/YOUR_NAME/stock-predictor.git
    git push -u origin main

Then go to share.streamlit.io → New app → select repo → streamlit_app.py → Deploy
""")
