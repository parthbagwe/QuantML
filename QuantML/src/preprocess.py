"""
preprocess.py
=============
Handles all data ingestion, cleaning, and normalization for stock CSV files.
Supports Kaggle OHLCV format and yfinance downloads.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Kaggle datasets use different capitalizations — this normalizes all of them
COLUMN_ALIASES = {
    "date": "date", "Date": "date", "Datetime": "date", "timestamp": "date",
    "open": "open", "Open": "open",
    "high": "high", "High": "high",
    "low": "low",   "Low": "low",
    "close": "close", "Close": "close",
    "adj close": "adj_close", "Adj Close": "adj_close", "adj_close": "adj_close",
    "volume": "volume", "Volume": "volume",
}


def load_and_clean(filepath_or_buffer) -> pd.DataFrame:
    """
    Load a stock CSV from disk or an uploaded buffer.
    Steps:
      1. Column normalization
      2. Date parsing and chronological sort
      3. Duplicate removal
      4. Missing value handling (forward fill)
      5. Bad data removal (negative prices, zero volume)
      6. IQR outlier removal
      7. Adjusted Close substitution
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    if isinstance(filepath_or_buffer, str):
        df = pd.read_csv(filepath_or_buffer)
    else:
        df = pd.read_csv(filepath_or_buffer)

    # ── 2. Normalize column names ─────────────────────────────────────────────
    df.rename(columns=COLUMN_ALIASES, inplace=True)
    keep = [c for c in ["date","open","high","low","close","adj_close","volume"]
            if c in df.columns]
    df = df[keep].copy()

    # ── 3. Parse dates and sort chronologically ───────────────────────────────
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 4. Remove duplicate dates ─────────────────────────────────────────────
    df.drop_duplicates(subset="date", keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 5. Use Adjusted Close as primary price if available ───────────────────
    # Adj Close corrects for stock splits and dividend payouts
    if "adj_close" in df.columns:
        df["close"] = df["adj_close"].fillna(df["close"])
        df.drop(columns=["adj_close"], inplace=True)

    # ── 6. Handle missing values ──────────────────────────────────────────────
    # Drop rows missing the target (close price)
    df.dropna(subset=["close"], inplace=True)

    # Forward-fill OHLV — preserves time structure better than dropping rows
    # Dropping rows would create gaps that break rolling window calculations
    for col in ["open", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Synthesize missing columns from close if completely absent
    if "open"   not in df.columns: df["open"]   = df["close"]
    if "high"   not in df.columns: df["high"]   = df["close"]
    if "low"    not in df.columns: df["low"]    = df["close"]
    if "volume" not in df.columns: df["volume"] = 0

    # ── 7. Remove bad data ────────────────────────────────────────────────────
    df = df[df["close"] > 0]
    df = df[df["volume"] >= 0]
    df = df[df["high"] >= df["low"]]

    # ── 8. IQR outlier removal on close price ─────────────────────────────────
    Q1 = df["close"].quantile(0.01)
    Q3 = df["close"].quantile(0.99)
    df = df[(df["close"] >= Q1) & (df["close"] <= Q3)]

    df.reset_index(drop=True, inplace=True)
    return df


def normalize_prices(df: pd.DataFrame, scaler_path: str = "models/price_scaler.pkl"):
    """
    Apply MinMaxScaler to OHLCV columns.
    Saves the scaler to disk so prediction uses the SAME scale.
    """
    cols_to_scale = [c for c in ["open","high","low","close","volume"]
                     if c in df.columns]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return df_scaled, scaler


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return summary statistics for display in the dashboard."""
    return {
        "rows":             len(df),
        "date_start":       str(df["date"].min().date()),
        "date_end":         str(df["date"].max().date()),
        "price_min":        round(float(df["close"].min()), 2),
        "price_max":        round(float(df["close"].max()), 2),
        "price_mean":       round(float(df["close"].mean()), 2),
        "price_latest":     round(float(df["close"].iloc[-1]), 2),
        "price_change_pct": round(
            float((df["close"].iloc[-1] - df["close"].iloc[0])
                  / df["close"].iloc[0] * 100), 2
        ),
        "avg_daily_volume": round(float(df["volume"].mean()), 0),
    }