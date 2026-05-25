"""
evaluate.py
===========
Metrics computation and chart generation.

Metrics:
  R²   — 1.0 = perfect, 0 = same as predicting the mean
  RMSE — same unit as price, penalizes large errors more
  MAE  — average absolute error, more interpretable
  MAPE — percentage error, scale-independent
  Directional Accuracy — % of days the model got up/down correct
                         (most trading-relevant metric)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dark fintech colour palette
C = {
    "teal":   "#00d4aa", "amber": "#f59e0b", "blue":  "#3b82f6",
    "red":    "#ef4444", "purple":"#8b5cf6", "bg":    "#0d1220",
    "surface":"#111827", "grid": "#1e2a3a",  "text":  "#c9d1e0",
    "muted":  "#4a5a6a",
}

MODEL_COLOURS = {
    "xgboost":           C["teal"],
    "random_forest":     C["blue"],
    "decision_tree":     C["amber"],
    "linear_regression": "#6b7a8a",
}

plt.rcParams.update({
    "figure.facecolor": C["bg"],    "axes.facecolor":   C["surface"],
    "axes.edgecolor":   C["grid"],  "axes.labelcolor":  C["text"],
    "text.color":       C["text"],  "xtick.color":      C["muted"],
    "ytick.color":      C["muted"], "grid.color":       C["grid"],
    "grid.linestyle":   "--",       "grid.linewidth":   0.5,
    "font.family":      "monospace","legend.facecolor":  C["surface"],
    "legend.edgecolor": C["grid"],
})


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all evaluation metrics for one model."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    nonzero = y_true != 0
    mape = float(
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    )

    actual_dir = np.diff(y_true) > 0
    pred_dir   = np.diff(y_pred) > 0
    dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)

    return {
        "R2":                   round(r2, 4),
        "RMSE":                 round(rmse, 4),
        "MAE":                  round(mae, 4),
        "MAPE":                 round(mape, 2),
        "Directional_Accuracy": round(dir_acc, 2),
    }


def evaluate_all(models: dict, X_test: np.ndarray, y_test: np.ndarray,
                 output_dir: str = "models") -> dict:
    """Evaluate all trained models and save metrics.json."""
    all_metrics = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        all_metrics[name] = metrics
        print(f"{name:25s}  R²={metrics['R2']:.4f}  "
              f"RMSE={metrics['RMSE']:.4f}  "
              f"Dir={metrics['Directional_Accuracy']:.1f}%")

    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved → {path}")
    return all_metrics


# ── Chart generators ──────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_true, predictions: dict,
                              dates=None, save_path=None):
    """Line chart: actual vs all model predictions."""
    N   = min(120, len(y_true))
    fig, ax = plt.subplots(figsize=(12, 5))
    x = dates.iloc[-N:] if dates is not None else range(N)

    ax.plot(x, y_true[-N:], color=C["text"], linewidth=1.5,
            label="Actual", zorder=5)
    for name, preds in predictions.items():
        ax.plot(x, preds[-N:],
                color=MODEL_COLOURS.get(name, C["muted"]),
                linewidth=1.2, linestyle="--", alpha=0.85,
                label=name.replace("_", " ").title())

    ax.set_title("Actual vs Predicted — Close Price", fontsize=13, pad=12)
    ax.set_ylabel("Price")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_model_comparison_bar(metrics: dict, save_path=None):
    """Grouped horizontal bars comparing R², RMSE, MAE."""
    names  = list(metrics.keys())
    labels = [n.replace("_", " ").title() for n in names]
    r2s    = [metrics[n]["R2"]   for n in names]
    rmses  = [metrics[n]["RMSE"] for n in names]
    maes   = [metrics[n]["MAE"]  for n in names]
    cols   = [MODEL_COLOURS.get(n, "#888") for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, title in zip(
        axes,
        [r2s, rmses, maes],
        ["R² Score (↑ better)", "RMSE (↓ better)", "MAE (↓ better)"]
    ):
        bars = ax.barh(labels, vals, color=cols, height=0.5, alpha=0.9)
        ax.set_title(title, fontsize=10, pad=8)
        ax.grid(True, axis="x", alpha=0.3)
        mv = max(vals) if vals else 1
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + mv * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9, color=C["text"])

    fig.suptitle("Model Performance Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_feature_importance(model, feature_cols: list, top_n: int = 15,
                             model_name: str = "xgboost", save_path=None):
    """Horizontal bar chart of top-N feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return None

    indices = np.argsort(importances)[-top_n:]
    feats   = [feature_cols[i] for i in indices]
    vals    = importances[indices]
    max_val = vals.max()
    colours = [plt.cm.YlGnBu(0.3 + 0.7 * v / max_val) for v in vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feats, vals, color=colours, height=0.6)
    ax.set_title(f"Feature Importance — {model_name.replace('_',' ').title()}",
                 fontsize=12, pad=10)
    ax.set_xlabel("Importance score")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color=C["text"])
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path=None):
    """Seaborn heatmap of feature correlations."""
    cols = [c for c in [
        "close","open","high","low","volume",
        "ma_20","ma_50","rsi_14","macd","bb_width",
        "daily_return","volume_ratio","close_lag1",
    ] if c in df.columns]

    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, ax=ax, cmap="RdYlGn", center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.5, linecolor=C["grid"],
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, pad=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_candlestick(df: pd.DataFrame, n: int = 60, save_path=None):
    """
    Matplotlib candlestick chart.
    Green candle = close >= open (bullish), Red = bearish.
    Wicks = high and low of the day.
    """
    subset = df.tail(n).reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})

    for i, row in subset.iterrows():
        colour = C["teal"] if row["close"] >= row["open"] else C["red"]
        body_bottom = min(row["open"], row["close"])
        body_height = abs(row["close"] - row["open"])
        ax1.add_patch(plt.Rectangle(
            (i - 0.35, body_bottom), 0.7, max(body_height, 0.01),
            color=colour, alpha=0.9
        ))
        ax1.plot([i, i], [row["low"], row["high"]], color=colour, linewidth=0.7)

    for col, colour, dash, lbl in [
        ("ma_20",  C["amber"],  "solid", "MA-20"),
        ("ma_50",  C["blue"],   "dash",  "MA-50"),
        ("ma_200", C["purple"], "dot",   "MA-200"),
    ]:
        if col in subset.columns and subset[col].notna().any():
            ax1.plot(subset.index, subset[col], color=colour,
                     linewidth=1.2, linestyle=dash, label=lbl, alpha=0.9)

    ax1.set_xlim(-1, n)
    ax1.set_xticks([])
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title(f"Candlestick — Last {n} Trading Days", fontsize=12)
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)

    vol_cols = [C["teal"] if row["close"] >= row["open"] else C["red"]
                for _, row in subset.iterrows()]
    ax2.bar(subset.index, subset["volume"], color=vol_cols, alpha=0.7, width=0.8)
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.3)

    tick_pos = range(0, n, 10)
    ax2.set_xticks(list(tick_pos))
    ax2.set_xticklabels(
        [str(subset["date"].iloc[i].date()) if i < len(subset) else ""
         for i in tick_pos],
        fontsize=7, rotation=30
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_moving_averages(df: pd.DataFrame, save_path=None):
    """MA-20, MA-50, MA-200 overlaid on price with golden-cross shading."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["date"], df["close"], color=C["text"],
            linewidth=0.8, alpha=0.6, label="Close")

    for col, colour, dash, lbl in [
        ("ma_20",  C["teal"],   "solid", "MA-20"),
        ("ma_50",  C["amber"],  "dashed","MA-50"),
        ("ma_200", C["purple"], "dotted","MA-200"),
    ]:
        if col in df.columns:
            ax.plot(df["date"], df[col], color=colour,
                    linewidth=1.4, linestyle=dash, label=lbl)

    if "ma_20" in df.columns and "ma_50" in df.columns:
        bullish = df["ma_20"] > df["ma_50"]
        ax.fill_between(df["date"], df["close"].min(), df["close"].max(),
                        where=bullish, alpha=0.04, color=C["teal"],
                        label="Bullish zone (MA20 > MA50)")

    ax.set_title("Moving Average Analysis", fontsize=13)
    ax.set_ylabel("Price")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_forecast(df: pd.DataFrame, forecast: np.ndarray,
                   confidence_band: float = 0.04, save_path=None):
    """Historical price + N-day forecast with shaded confidence band."""
    n_hist     = min(90, len(df))
    n_forecast = len(forecast)
    hist_dates = df["date"].iloc[-n_hist:]
    last_date  = df["date"].iloc[-1]
    fc_dates   = pd.date_range(start=last_date, periods=n_forecast + 1, freq="B")[1:]

    upper = forecast * (1 + confidence_band)
    lower = forecast * (1 - confidence_band)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hist_dates, df["close"].iloc[-n_hist:], color=C["text"],
            linewidth=1.2, label="Historical")
    ax.plot(fc_dates, forecast, color=C["teal"], linewidth=2,
            linestyle="--", label="Forecast")
    ax.fill_between(fc_dates, lower, upper,
                    color=C["teal"], alpha=0.15, label="±4% band")
    ax.axvline(x=last_date, color=C["muted"], linewidth=1, linestyle=":")
    ax.set_title(f"{n_forecast}-Day Price Forecast", fontsize=13)
    ax.set_ylabel("Price")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig
