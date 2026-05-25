from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib, json

FEATURES = [
    'close_lag1','close_lag5','close_lag20',
    'ma_20','ma_50','ema_12','ema_26',
    'macd','macd_signal','rsi_14',
    'bb_upper','bb_lower','bb_width',
    'daily_return','high_low_range','open_close_delta',
    'volume','volume_ratio'
]

def train(df):
    X = df[FEATURES]
    y = df['close']

    # TimeSeriesSplit: NEVER use random train/test split on time series
    # Random split leaks future data into training → artificially inflated R²
    # TimeSeriesSplit always trains on past, tests on future
    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        'linear': LinearRegression(),

        'decision_tree': DecisionTreeRegressor(
            max_depth=8,          # shallow tree = less overfitting
            min_samples_leaf=10,  # each leaf needs 10 data points minimum
            random_state=42
        ),

        'random_forest': RandomForestRegressor(
            n_estimators=200,     # 200 decision trees
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,            # use all CPU cores
            random_state=42
        ),

        'xgboost': XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,   # lower = more careful = better generalisation
            max_depth=6,
            subsample=0.8,        # each tree trained on 80% of data
            colsample_bytree=0.8, # each tree uses 80% of features
            reg_lambda=1.5,       # L2 regularisation — prevents overfitting
            early_stopping_rounds=30,  # stops if no improvement for 30 rounds
            eval_metric='rmse',
            random_state=42
        )
    }

    # Use last fold as final train/test
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, model in models.items():
        if name == 'xgboost':
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      verbose=False)
        else:
            model.fit(X_train, y_train)

        joblib.dump(model, f'models/{name}_model.pkl')
        print(f"Saved {name}")

    return models, X_test, y_test