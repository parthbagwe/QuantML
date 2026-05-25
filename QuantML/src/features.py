import pandas as pd
import numpy as np

def add_features(df):

    # --- Lag features: yesterday's close is the single strongest predictor ---
    # The model learns "if price was X yesterday, it's probably near X today"
    df['close_lag1'] = df['close'].shift(1)    # 1 day ago
    df['close_lag5'] = df['close'].shift(5)    # 1 week ago
    df['close_lag20'] = df['close'].shift(20)  # 1 month ago

    # --- Moving averages: smooth out noise, reveal trend direction ---
    df['ma_20'] = df['close'].rolling(20).mean()   # short-term trend
    df['ma_50'] = df['close'].rolling(50).mean()   # medium-term
    df['ma_200'] = df['close'].rolling(200).mean() # long-term (slower)

    # --- Exponential MA: weights recent prices more heavily ---
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # --- MACD: momentum indicator ---
    # When MACD crosses above signal line → bullish signal
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # --- RSI: measures overbought/oversold (0-100) ---
    # Above 70 = overbought (price may drop), below 30 = oversold (may rise)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands: volatility envelope around price ---
    # Price touching upper band = overextended, lower band = support
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = ma20 + 2 * std20
    df['bb_lower'] = ma20 - 2 * std20
    df['bb_width'] = df['bb_upper'] - df['bb_lower']  # volatility proxy

    # --- Price-derived features ---
    df['daily_return'] = df['close'].pct_change()          # % change day over day
    df['high_low_range'] = df['high'] - df['low']          # intraday volatility
    df['open_close_delta'] = df['close'] - df['open']      # direction of day

    # --- Volume features ---
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']  # volume spike indicator

    # Drop NaN rows created by rolling windows (first 200 rows will be NaN)
    df.dropna(inplace=True)

    return df