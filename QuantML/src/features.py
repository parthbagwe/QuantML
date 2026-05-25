"""
features.py
===========
Generates all 28 technical indicator features from clean OHLCV data.

Indicators:
  MA-20, MA-50, MA-200, EMA-12, EMA-26
  MACD, MACD signal, MACD histogram
  RSI-14 (Wilder's smoothing)
  Bollinger Bands: upper, lower, width, %B
  OBV (On-Balance Volume)
  Volume MA-20, Volume Ratio
  Lag features: close lag 1, 5, 10, 20
  Daily return, log return, rolling volatility 20
  High-Low range, Open-Close delta, True Range
  Close vs MA-20, Close vs MA-50
"""

import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "close_lag1", "close_lag5", "close_lag10", "close_lag20",
    "ma_20", "ma_50", "ema_12", "ema_26",
    "macd", "macd_signal", "macd_hist",
    "rsi_14",
    "bb_upper", "bb_lower", "bb_width", "bb_pct",
    "obv",
    "volume_ma20", "volume_ratio",
    "daily_return", "log_return", "rolling_vol_20",
    "high_low_range", "open_close_delta", "true_range",
    "close_vs_ma20", "close_vs_ma50",
    "volume",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the DataFrame.
    Drops rows with NaN produced by rolling windows (first ~200 rows).
    """
    df = df.copy()

    # ── Lag features ──────────────────────────────────────────────────────────
    # Previous close prices — the strongest single predictor of next close
    df["close_lag1"]  = df["close"].shift(1)
    df["close_lag5"]  = df["close"].shift(5)
    df["close_lag10"] = df["close"].shift(10)
    df["close_lag20"] = df["close"].shift(20)

    # ── Simple moving averages ────────────────────────────────────────────────
    # MA-20 = short-term trend, MA-50 = medium, MA-200 = long-term
    df["ma_20"]  = df["close"].rolling(window=20).mean()
    df["ma_50"]  = df["close"].rolling(window=50).mean()
    df["ma_200"] = df["close"].rolling(window=200).mean()

    # ── Exponential moving averages ───────────────────────────────────────────
    # EMA weights recent prices more than SMA
    # EMA-12 (fast) and EMA-26 (slow) are used to calculate MACD
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # ── MACD ──────────────────────────────────────────────────────────────────
    # MACD > 0 → short-term momentum is bullish
    # MACD crossing above signal line → buy signal
    # Histogram = momentum of momentum (acceleration)
    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── RSI-14 ────────────────────────────────────────────────────────────────
    # Measures speed and change of price movements. Range: 0–100
    # RSI > 70 = overbought (may fall), RSI < 30 = oversold (may rise)
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()   # Wilder's smoothing
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    # Price near upper band → overbought, near lower band → oversold
    # BB Width = volatility measure (wider = more volatile)
    # BB %B = position of price within the bands (0 = lower, 1 = upper)
    ma20           = df["close"].rolling(20).mean()
    std20          = df["close"].rolling(20).std()
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / df["bb_width"].replace(0, np.nan)

    # ── On-Balance Volume ─────────────────────────────────────────────────────
    # Rising OBV with rising price = confirmed uptrend
    # Divergence between OBV and price often predicts reversals
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv

    # ── Volume features ───────────────────────────────────────────────────────
    # Volume ratio > 1.5 = volume spike, often signals a strong move
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = (df["volume"] / df["volume_ma20"].replace(0, np.nan)).fillna(1.0)

    # ── Return features ───────────────────────────────────────────────────────
    df["daily_return"]   = df["close"].pct_change()
    df["log_return"]     = np.log(df["close"] / df["close"].shift(1))
    df["rolling_vol_20"] = df["daily_return"].rolling(20).std()

    # ── Price-derived features ────────────────────────────────────────────────
    df["high_low_range"]   = df["high"] - df["low"]
    df["open_close_delta"] = df["close"] - df["open"]
    df["true_range"]       = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"]  - df["close"].shift(1))
        )
    )

    # ── Relative position to moving averages ──────────────────────────────────
    # Positive = price above MA (bullish), Negative = below (bearish)
    df["close_vs_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["close_vs_ma50"] = (df["close"] - df["ma_50"]) / df["ma_50"]

    # Drop NaN rows created by rolling windows
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_feature_columns() -> list:
    return FEATURE_COLUMNS
