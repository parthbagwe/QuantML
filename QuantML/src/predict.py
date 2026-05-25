"""
predict.py
==========
Prediction engine.

predict_test_set()  — predict on the last 20% of the dataset
forecast_n_days()   — iterative N-day ahead forecast
all_models_predict_test() — run all 4 models simultaneously
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from src.features import get_feature_columns
from src.evaluate import compute_metrics

MODEL_MAP = {
    "xgboost":           "models/xgboost_model.pkl",
    "random_forest":     "models/random_forest_model.pkl",
    "decision_tree":     "models/decision_tree_model.pkl",
    "linear_regression": "models/linear_regression_model.pkl",
}


def load_model(model_name: str):
    key  = model_name.lower().replace(" ", "_")
    path = MODEL_MAP.get(key)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"Model '{model_name}' not found. Run python run_training.py first."
        )
    return joblib.load(path)


def load_feature_cols() -> list:
    path = "models/feature_columns.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return get_feature_columns()


def predict_test_set(df: pd.DataFrame, model_name: str) -> dict:
    """
    Run inference on the last 20% of data (the test portion).
    Returns actual prices, predicted prices, dates, and all metrics.
    """
    model        = load_model(model_name)
    feature_cols = [f for f in load_feature_cols() if f in df.columns]

    X     = df[feature_cols].values
    y     = df["close"].values
    dates = df["date"]

    split  = int(len(X) * 0.8)
    X_test = X[split:]
    y_test = y[split:]
    dates_test = dates.iloc[split:].reset_index(drop=True)

    preds   = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)

    return {
        "dates":     dates_test.astype(str).tolist(),
        "actual":    y_test.tolist(),
        "predicted": preds.tolist(),
        "metrics":   metrics,
    }


def forecast_n_days(df: pd.DataFrame, model_name: str, n_days: int = 30) -> dict:
    """
    Iterative N-day ahead forecast.

    Strategy:
      1. Start from the last known feature row.
      2. Predict day t+1.
      3. Feed the prediction back as close_lag1, update other lags.
      4. Repeat for n_days steps.

    Confidence band widens with horizon — uncertainty grows each step.
    """
    model        = load_model(model_name)
    feature_cols = [f for f in load_feature_cols() if f in df.columns]
    col_idx      = {name: i for i, name in enumerate(feature_cols)}

    last_close = float(df["close"].iloc[-1])
    last_date  = df["date"].iloc[-1]
    current    = df[feature_cols].iloc[-1].values.copy().astype(float)

    forecast_prices = []
    prev_close      = last_close

    for step in range(n_days):
        pred = float(model.predict(current.reshape(1, -1))[0])
        forecast_prices.append(pred)

        # Feed prediction back as lag features for next step
        if "close_lag1"  in col_idx:
            current[col_idx["close_lag1"]]  = pred
        if "close_lag5"  in col_idx and step >= 5:
            current[col_idx["close_lag5"]]  = forecast_prices[-5]
        if "close_lag10" in col_idx and step >= 10:
            current[col_idx["close_lag10"]] = forecast_prices[-10]
        if "close_lag20" in col_idx and step >= 20:
            current[col_idx["close_lag20"]] = forecast_prices[-20]
        if "daily_return" in col_idx and prev_close:
            current[col_idx["daily_return"]] = (pred - prev_close) / prev_close

        prev_close = pred

    # Business day dates for forecast
    fc_dates = pd.date_range(start=last_date, periods=n_days + 1, freq="B")[1:]

    # Confidence band: ±2% day 1, grows to ±8% at day 30
    bands = [0.02 + 0.002 * i for i in range(n_days)]
    upper = [p * (1 + b) for p, b in zip(forecast_prices, bands)]
    lower = [p * (1 - b) for p, b in zip(forecast_prices, bands)]

    price_change_pct = round(
        (forecast_prices[-1] - last_close) / last_close * 100, 2
    )

    return {
        "dates":              [str(d.date()) for d in fc_dates],
        "forecast":           [round(p, 2) for p in forecast_prices],
        "upper":              [round(u, 2) for u in upper],
        "lower":              [round(l, 2) for l in lower],
        "last_known_price":   round(last_close, 2),
        "price_change_pct":   price_change_pct,
        "trend":              "Bullish" if price_change_pct > 0 else "Bearish",
    }


def all_models_predict_test(df: pd.DataFrame) -> dict:
    """Run all 4 models simultaneously and return combined results."""
    results = {}
    for name in MODEL_MAP:
        try:
            results[name] = predict_test_set(df, name)
        except FileNotFoundError:
            pass
    return results
