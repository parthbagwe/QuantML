"""
streamlit_app.py
================
Self-contained — trains all 4 models in-browser when data is uploaded.
No .pkl files needed. Works on Streamlit Cloud out of the box.
"""

import io, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

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
</style>
""", unsafe_allow_html=True)

PL = dict(
    paper_bgcolor="#0d1220", plot_bgcolor="#080c15",
    font_color="#c9d1e0",
    xaxis=dict(gridcolor="#1e2a3a", zerolinecolor="#1e2a3a"),
    yaxis=dict(gridcolor="#1e2a3a", zerolinecolor="#1e2a3a"),
    legend=dict(bgcolor="#0d1220", bordercolor="#1e2a3a", borderwidth=1),
    margin=dict(l=50, r=30, t=40, b=40),
)

COLOURS = {
    "XGBoost":            "#00d4aa",
    "Random Forest":      "#3b82f6",
    "Decision Tree":      "#f59e0b",
    "Linear Regression":  "#6b7a8a",
}

# ── Session state ─────────────────────────────────────────────────────────────
for k in ["df", "models", "feature_cols", "metrics", "X_test", "y_test",
          "dates_test", "forecast"]:
    if k not in st.session_state:
        st.session_state[k] = None


# ════════════════════════════════════════
# DATA PROCESSING
# ════════════════════════════════════════

COLUMN_ALIASES = {
    "Date":"date","date":"date","Datetime":"date","timestamp":"date",
    "Open":"open","open":"open",
    "High":"high","high":"high",
    "Low":"low","low":"low",
    "Close":"close","close":"close",
    "Adj Close":"adj_close","adj close":"adj_close","adj_close":"adj_close",
    "Volume":"volume","volume":"volume",
}

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns=COLUMN_ALIASES, inplace=True)

    # Try to find a date column even if not in aliases
    if "date" not in df.columns:
        for col in df.columns:
            try:
                pd.to_datetime(df[col].iloc[:5])
                df.rename(columns={col: "date"}, inplace=True)
                break
            except Exception:
                continue

    keep = [c for c in ["date","open","high","low","close","adj_close","volume"]
            if c in df.columns]
    df = df[keep].copy()

    if "date" not in df.columns:
        st.error("No date column found. Columns in your file: " + str(list(df.columns)))
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df.sort_values("date", inplace=True)
    df.drop_duplicates(subset="date", keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if "adj_close" in df.columns:
        df["close"] = df["adj_close"].fillna(df["close"])
        df.drop(columns=["adj_close"], inplace=True)

    df.dropna(subset=["close"], inplace=True)

    for col in ["open","high","low","volume"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
        else:
            df[col] = df["close"]

    df = df[df["close"] > 0]
    df.reset_index(drop=True, inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["close_lag1"]  = df["close"].shift(1)
    df["close_lag5"]  = df["close"].shift(5)
    df["close_lag10"] = df["close"].shift(10)
    df["close_lag20"] = df["close"].shift(20)
    df["ma_20"]       = df["close"].rolling(20).mean()
    df["ma_50"]       = df["close"].rolling(50).mean()
    df["ma_200"]      = df["close"].rolling(200).mean()
    df["ema_12"]      = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"]      = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

    ma20           = df["close"].rolling(20).mean()
    std20          = df["close"].rolling(20).std()
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / df["bb_width"].replace(0, np.nan)

    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv

    df["volume_ma20"]    = df["volume"].rolling(20).mean()
    df["volume_ratio"]   = (df["volume"] / df["volume_ma20"].replace(0, np.nan)).fillna(1.0)
    df["daily_return"]   = df["close"].pct_change()
    df["log_return"]     = np.log(df["close"] / df["close"].shift(1))
    df["rolling_vol_20"] = df["daily_return"].rolling(20).std()
    df["high_low_range"] = df["high"] - df["low"]
    df["open_close_delta"] = df["close"] - df["open"]
    df["true_range"]     = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"]  - df["close"].shift(1))
        )
    )
    df["close_vs_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["close_vs_ma50"] = (df["close"] - df["ma_50"]) / df["ma_50"]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


FEATURE_COLS = [
    "close_lag1","close_lag5","close_lag10","close_lag20",
    "ma_20","ma_50","ema_12","ema_26",
    "macd","macd_signal","macd_hist","rsi_14",
    "bb_upper","bb_lower","bb_width","bb_pct",
    "obv","volume_ma20","volume_ratio",
    "daily_return","log_return","rolling_vol_20",
    "high_low_range","open_close_delta","true_range",
    "close_vs_ma20","close_vs_ma50","volume",
]


# ════════════════════════════════════════
# TRAINING — runs in-app, no .pkl needed
# ════════════════════════════════════════

def compute_metrics(y_true, y_pred) -> dict:
    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae     = float(mean_absolute_error(y_true, y_pred))
    r2      = float(r2_score(y_true, y_pred))
    nonzero = y_true != 0
    mape    = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    dir_acc = float(np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0)) * 100)
    return {
        "R2": round(r2, 4), "RMSE": round(rmse, 4),
        "MAE": round(mae, 4), "MAPE": round(mape, 2),
        "Directional_Accuracy": round(dir_acc, 2),
    }


def train_all(df: pd.DataFrame):
    feat_cols = [f for f in FEATURE_COLS if f in df.columns]
    X = df[feat_cols].values
    y = df["close"].values

    tscv   = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    tr_idx, te_idx = splits[-1]

    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]
    dates_test      = df["date"].iloc[te_idx].reset_index(drop=True)

    model_defs = {
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Decision Tree":     DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
        "XGBoost":           XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                          subsample=0.8, colsample_bytree=0.8,
                                          random_state=42, verbosity=0,
                                          eval_metric="rmse", early_stopping_rounds=20),
    }

    trained = {}
    metrics = {}
    progress = st.progress(0, text="Training models...")

    for i, (name, model) in enumerate(model_defs.items()):
        progress.progress((i) / len(model_defs), text=f"Training {name}...")
        if name == "XGBoost":
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train)
        preds          = model.predict(X_test)
        trained[name]  = model
        metrics[name]  = compute_metrics(y_test, preds)

    progress.progress(1.0, text="All models trained!")
    progress.empty()

    return trained, metrics, X_test, y_test, dates_test, feat_cols


# ════════════════════════════════════════
# PREDICTION & FORECAST
# ════════════════════════════════════════

def predict(model, df: pd.DataFrame, feat_cols: list) -> dict:
    feat_cols = [f for f in feat_cols if f in df.columns]
    X      = df[feat_cols].values
    y      = df["close"].values
    split  = int(len(X) * 0.8)
    X_test = X[split:]
    y_test = y[split:]
    dates  = df["date"].iloc[split:].reset_index(drop=True)
    preds  = model.predict(X_test)
    return {
        "dates":     dates.astype(str).tolist(),
        "actual":    y_test.tolist(),
        "predicted": preds.tolist(),
        "metrics":   compute_metrics(y_test, preds),
    }


def forecast_n(model, df: pd.DataFrame, feat_cols: list, n_days: int) -> dict:
    feat_cols  = [f for f in feat_cols if f in df.columns]
    col_idx    = {name: i for i, name in enumerate(feat_cols)}
    last_close = float(df["close"].iloc[-1])
    last_date  = df["date"].iloc[-1]
    current    = df[feat_cols].iloc[-1].values.copy().astype(float)

    prices = []
    prev   = last_close

    for step in range(n_days):
        pred = float(model.predict(current.reshape(1, -1))[0])
        prices.append(pred)
        if "close_lag1"  in col_idx: current[col_idx["close_lag1"]]  = pred
        if "close_lag5"  in col_idx and step >= 5:  current[col_idx["close_lag5"]]  = prices[-5]
        if "close_lag10" in col_idx and step >= 10: current[col_idx["close_lag10"]] = prices[-10]
        if "close_lag20" in col_idx and step >= 20: current[col_idx["close_lag20"]] = prices[-20]
        if "daily_return" in col_idx and prev:
            current[col_idx["daily_return"]] = (pred - prev) / prev
        prev = pred

    fc_dates = pd.date_range(start=last_date, periods=n_days + 1, freq="B")[1:]
    bands    = [0.02 + 0.002 * i for i in range(n_days)]
    upper    = [p * (1 + b) for p, b in zip(prices, bands)]
    lower    = [p * (1 - b) for p, b in zip(prices, bands)]
    chg      = round((prices[-1] - last_close) / last_close * 100, 2)

    return {
        "dates":            [str(d.date()) for d in fc_dates],
        "forecast":         [round(p, 2) for p in prices],
        "upper":            [round(u, 2) for u in upper],
        "lower":            [round(l, 2) for l in lower],
        "last_known_price": round(last_close, 2),
        "price_change_pct": chg,
        "trend":            "Bullish" if chg > 0 else "Bearish",
    }


# ════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════

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
        x=sub["date"], y=sub["volume"],
        marker_color=vol_c, showlegend=False, name="Volume",
    ), row=2, col=1)
    fig.update_layout(**PL, xaxis_rangeslider_visible=False,
                      title="Candlestick + Volume", height=480)
    return fig


def chart_avp(result, model_name):
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
    fig.update_layout(**PL, title="Actual vs Predicted", height=360)
    return fig


def chart_all_models(all_results):
    names  = list(all_results.keys())
    labels = names
    cols   = [COLOURS.get(n, "#888") for n in names]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["R² Score", "RMSE", "MAE"])
    for ci, key in enumerate(["R2","RMSE","MAE"], 1):
        vals = [all_results[n]["metrics"][key] for n in names]
        fig.add_trace(go.Bar(
            x=labels, y=vals, marker_color=cols, showlegend=False,
            text=[f"{v:.3f}" for v in vals], textposition="outside",
        ), row=1, col=ci)
    fig.update_layout(**PL, height=380, title="Model Comparison")
    return fig


def chart_overlay(all_results):
    fig = go.Figure()
    first = list(all_results.values())[0]
    fig.add_trace(go.Scatter(
        x=first["dates"], y=first["actual"],
        name="Actual", line=dict(color="#e2e8f4", width=1.8),
    ))
    for name, r in all_results.items():
        fig.add_trace(go.Scatter(
            x=r["dates"], y=r["predicted"],
            name=name,
            line=dict(color=COLOURS.get(name,"#888"), width=1.3, dash="dot"),
        ))
    fig.update_layout(**PL, height=420, title="All Models vs Actual")
    return fig


def chart_forecast(fc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc["dates"] + fc["dates"][::-1],
        y=fc["upper"] + fc["lower"][::-1],
        fill="toself", fillcolor="rgba(0,212,170,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence band",
    ))
    fig.add_trace(go.Scatter(
        x=fc["dates"], y=fc["forecast"], name="Forecast",
        line=dict(color="#00d4aa", width=2, dash="dash"),
    ))
    sym = "▲" if fc["trend"] == "Bullish" else "▼"
    fig.update_layout(
        **PL,
        title=f"Forecast  {sym} {fc['trend']} ({fc['price_change_pct']:+.2f}%)",
        height=380,
    )
    return fig


def chart_ma(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close",
                             line=dict(color="rgba(226,232,244,0.5)", width=0.8)))
    for col, colour, dash, lbl in [
        ("ma_20","#00d4aa","solid","MA-20"),
        ("ma_50","#f59e0b","dash","MA-50"),
        ("ma_200","#8b5cf6","dot","MA-200"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], name=lbl,
                                     line=dict(color=colour, width=1.4, dash=dash)))
    fig.update_layout(**PL, title="Moving Averages", height=360)
    return fig


def chart_rsi(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["rsi_14"], name="RSI-14",
                             line=dict(color="#f59e0b", width=1.4)))
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                  annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e",
                  annotation_text="Oversold")
    fig.update_layout(**PL, title="RSI-14", height=260, yaxis_range=[0,100])
    return fig


def chart_macd(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["macd"], name="MACD",
                             line=dict(color="#00d4aa", width=1.2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal",
                             line=dict(color="#f59e0b", width=1.2, dash="dot")))
    fig.add_trace(go.Bar(x=df["date"], y=df["macd_hist"], name="Histogram",
                         marker_color=["#22c55e" if v >= 0 else "#ef4444"
                                       for v in df["macd_hist"]], opacity=0.7))
    fig.update_layout(**PL, title="MACD", height=260)
    return fig


def chart_heatmap(df):
    cols = [c for c in ["close","open","high","low","volume","ma_20","ma_50",
                        "rsi_14","macd","bb_width","daily_return",
                        "volume_ratio","close_lag1"] if c in df.columns]
    corr = df[cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=cols, y=cols,
        colorscale="RdYlGn", zmin=-1, zmax=1,
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
        fig.add_trace(go.Scatter(x=df["date"], y=df["volume_ma20"],
                                 name="Vol MA-20",
                                 line=dict(color="#f59e0b", width=1.4)))
    fig.update_layout(**PL, title="Volume", height=300)
    return fig


def chart_feat_importance(model, feat_cols, model_name):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return None
    top = 15
    idx   = np.argsort(imp)[-top:]
    feats = [feat_cols[i].replace("_"," ") for i in idx]
    vals  = imp[idx]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker=dict(color=vals, colorscale=[[0,"#1a3a2a"],[1,"#00d4aa"]]),
        text=[f"{v:.4f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(**PL,
                      title=f"Feature Importance — {model_name}",
                      height=460)
    return fig


# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════

with st.sidebar:
    st.markdown("### QuantML")
    st.markdown("*Stock Price Predictor*")
    st.divider()

    uploaded   = st.file_uploader("Upload Kaggle CSV", type=["csv"])
    use_sample = st.button("Load AAPL sample (yfinance)")
    st.divider()

    model_choice  = st.selectbox(
        "ML Model",
        ["XGBoost", "Random Forest", "Decision Tree", "Linear Regression"],
    )
    forecast_days = st.slider("Forecast days", 7, 90, 30, step=7)
    run_btn       = st.button("▶  Run Prediction", use_container_width=True)
    compare_btn   = st.button("⚖  Compare All Models", use_container_width=True)


# ════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════

if uploaded is not None:
    with st.spinner("Loading and cleaning data..."):
        try:
            raw = pd.read_csv(uploaded)
            df  = clean(raw)
            df  = add_features(df)
            st.session_state.df      = df
            st.session_state.models  = None   # reset models when new data loaded
            st.session_state.metrics = None
            st.sidebar.success(f"Loaded {len(df):,} rows")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

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
            df  = clean(raw)
            df  = add_features(df)
            st.session_state.df      = df
            st.session_state.models  = None
            st.session_state.metrics = None
            st.sidebar.success(f"AAPL loaded: {len(df):,} rows")
        except Exception as e:
            st.sidebar.error(f"Download failed: {e}")

df = st.session_state.df


# ════════════════════════════════════════
# TRAIN when Run or Compare is clicked
# ════════════════════════════════════════

if (run_btn or compare_btn) and df is not None:
    if st.session_state.models is None:
        with st.spinner("Training all 4 models on your data..."):
            trained, metrics, X_test, y_test, dates_test, feat_cols = train_all(df)
            st.session_state.models      = trained
            st.session_state.metrics     = metrics
            st.session_state.X_test      = X_test
            st.session_state.y_test      = y_test
            st.session_state.dates_test  = dates_test
            st.session_state.feat_cols   = feat_cols
            st.success("All 4 models trained and ready.")


# ════════════════════════════════════════
# TABS
# ════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🏆 Model Compare",
    "📈 Analytics",
    "🔮 Forecast",
    "ℹ️ About",
])


# ── TAB 1 DASHBOARD ──────────────────────────────────────────────────────────
with tab1:
    if df is None:
        st.info("Upload a Kaggle CSV or click Load AAPL sample in the sidebar.")
    else:
        latest  = float(df["close"].iloc[-1])
        oldest  = float(df["close"].iloc[0])
        chg_pct = round((latest - oldest) / oldest * 100, 2)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Latest Price",  f"${latest:,.2f}")
        c2.metric("Total Return",  f"{chg_pct:+.1f}%", delta=f"{chg_pct:+.1f}%")
        c3.metric("Price Range",   f"${df['close'].min():,.0f}–${df['close'].max():,.0f}")
        c4.metric("Data Points",   f"{len(df):,}")
        c5.metric("Period",
                  f"{str(df['date'].min().date())[:7]} → {str(df['date'].max().date())[:7]}")

        st.plotly_chart(chart_candlestick(df), use_container_width=True)
        st.divider()

        models  = st.session_state.models
        metrics = st.session_state.metrics

        if models is None:
            st.info("Click **▶ Run Prediction** in the sidebar to train models and see predictions.")
        else:
            result = predict(models[model_choice], df, st.session_state.feat_cols)
            m      = result["metrics"]
            st.subheader(f"Model: {model_choice}")
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("R² Score",      f"{m['R2']:.4f}")
            c2.metric("RMSE",          f"${m['RMSE']:.2f}")
            c3.metric("MAE",           f"${m['MAE']:.2f}")
            c4.metric("MAPE",          f"{m['MAPE']:.2f}%")
            c5.metric("Dir. Accuracy", f"{m['Directional_Accuracy']:.1f}%")
            st.plotly_chart(chart_avp(result, model_choice), use_container_width=True)


# ── TAB 2 MODEL COMPARE ───────────────────────────────────────────────────────
with tab2:
    if df is None:
        st.info("Upload a dataset first.")
    elif st.session_state.models is None:
        st.info("Click **⚖ Compare All Models** in the sidebar to train and compare.")
    else:
        models  = st.session_state.models
        metrics = st.session_state.metrics
        feat_cols = st.session_state.feat_cols

        # Build per-model results for overlay chart
        all_results = {
            name: predict(model, df, feat_cols)
            for name, model in models.items()
        }

        st.plotly_chart(chart_all_models(all_results), use_container_width=True)

        # Metrics table
        rows = []
        for name, r in all_results.items():
            m = r["metrics"]
            rows.append({
                "Model":              name,
                "R²":                 m["R2"],
                "RMSE":               m["RMSE"],
                "MAE":                m["MAE"],
                "MAPE (%)":           m["MAPE"],
                "Dir. Accuracy (%)":  m["Directional_Accuracy"],
            })
        st.dataframe(
            pd.DataFrame(rows).set_index("Model"),
            use_container_width=True,
        )

        st.plotly_chart(chart_overlay(all_results), use_container_width=True)

        # Feature importance for best model
        best = max(metrics, key=lambda k: metrics[k]["R2"])
        st.subheader(f"Feature Importance — {best} (best R²)")
        fi = chart_feat_importance(models[best], feat_cols, best)
        if fi:
            st.plotly_chart(fi, use_container_width=True)


# ── TAB 3 ANALYTICS ───────────────────────────────────────────────────────────
with tab3:
    if df is None:
        st.info("Upload a dataset first.")
    else:
        st.plotly_chart(chart_ma(df),      use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_rsi(df),  use_container_width=True)
        with c2:
            st.plotly_chart(chart_macd(df), use_container_width=True)
        st.plotly_chart(chart_heatmap(df), use_container_width=True)
        st.plotly_chart(chart_volume(df),  use_container_width=True)


# ── TAB 4 FORECAST ────────────────────────────────────────────────────────────
with tab4:
    if df is None:
        st.info("Upload a dataset first.")
    elif st.session_state.models is None:
        st.info("Click **▶ Run Prediction** in the sidebar to train models first.")
    else:
        models    = st.session_state.models
        feat_cols = st.session_state.feat_cols

        if st.button("Generate Forecast"):
            with st.spinner(f"Forecasting {forecast_days} days with {model_choice}..."):
                fc = forecast_n(models[model_choice], df, feat_cols, forecast_days)
                st.session_state.forecast = fc

        fc = st.session_state.forecast
        if fc:
            col1, col2 = st.columns([2, 1])
            with col1:
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
                st.metric("Starting price",      f"${fc['last_known_price']:,.2f}")
                st.metric(f"Day-{forecast_days}", f"${fc['forecast'][-1]:,.2f}",
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


# ── TAB 5 ABOUT ───────────────────────────────────────────────────────────────
with tab5:
    st.markdown("""
## QuantML · Stock Price Predictor

All 4 models train directly in the browser when you upload data.
No pre-trained files needed. Works on Streamlit Cloud out of the box.

### How to use
1. Upload any Kaggle OHLCV CSV (or click Load AAPL sample)
2. Click **Run Prediction** to train all 4 models
3. Click **Compare All Models** to see side-by-side comparison
4. Go to Forecast tab and click Generate Forecast

### Models
- **XGBoost** — best accuracy, gradient boosted trees
- **Random Forest** — robust ensemble, handles noise well
- **Decision Tree** — fast, interpretable
- **Linear Regression** — baseline comparison

### Deploy
    git add . && git commit -m "fix: self-contained training"
    git push
    # Streamlit Cloud auto-redeploys
""")
