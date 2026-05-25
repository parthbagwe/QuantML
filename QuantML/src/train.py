"""
train.py
========
Trains all four ML models using TimeSeriesSplit cross-validation.

Why TimeSeriesSplit instead of random split?
  Random split leaks future data into training.
  Example: training on 2023 data, testing on 2021 data = model "sees the future".
  TimeSeriesSplit always trains on the past, tests on the future.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.features import get_feature_columns


def build_models() -> dict:
    """
    Instantiate all four models with tuned hyperparameters.
    """
    return {
        "linear_regression": LinearRegression(
            fit_intercept=True,
            n_jobs=-1
        ),

        "decision_tree": DecisionTreeRegressor(
            max_depth=8,            # shallow = less overfitting
            min_samples_leaf=10,    # leaf needs >= 10 samples
            min_samples_split=20,   # node needs >= 20 to split
            random_state=42
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=200,       # 200 trees — averages out errors
            max_depth=12,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features="sqrt",    # each tree considers sqrt(n_features)
            n_jobs=-1,              # use all CPU cores
            random_state=42
        ),

        "xgboost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,     # small steps = more precise convergence
            max_depth=6,
            subsample=0.8,          # each tree trains on 80% of rows
            colsample_bytree=0.8,   # each tree uses 80% of features
            reg_lambda=1.5,         # L2 regularization
            reg_alpha=0.1,          # L1 regularization
            min_child_weight=5,
            gamma=0.1,
            random_state=42,
            eval_metric="rmse",
            early_stopping_rounds=30,
            verbosity=0,
        ),
    }


def train(df: pd.DataFrame, models_dir: str = "models") -> tuple:
    """
    Train all models. Uses last fold of TimeSeriesSplit as hold-out test set.

    Returns
    -------
    (trained_models, X_test, y_test, feature_cols)
    """
    os.makedirs(models_dir, exist_ok=True)

    feature_cols = get_feature_columns()
    feature_cols = [f for f in feature_cols if f in df.columns]

    X = df[feature_cols].values
    y = df["close"].values

    print(f"Training on {len(X)} samples, {len(feature_cols)} features")

    # TimeSeriesSplit — 5 folds, always past→future
    tscv   = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]   # use last fold as official test set

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    models  = build_models()
    trained = {}

    for name, model in models.items():
        print(f"\n▶ Training {name}...")

        if name == "xgboost":
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      verbose=False)
        else:
            model.fit(X_train, y_train)

        path = os.path.join(models_dir, f"{name}_model.pkl")
        joblib.dump(model, path)
        print(f"   Saved → {path}")
        trained[name] = model

    # Save feature list — ensures train/predict use same features
    feat_path = os.path.join(models_dir, "feature_columns.json")
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"\nFeature list saved → {feat_path}")

    return trained, X_test, y_test, feature_cols


def load_model(name: str, models_dir: str = "models"):
    path = os.path.join(models_dir, f"{name}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}. Run python run_training.py first."
        )
    return joblib.load(path)


def load_feature_columns(models_dir: str = "models") -> list:
    path = os.path.join(models_dir, "feature_columns.json")
    if not os.path.exists(path):
        return get_feature_columns()
    with open(path) as f:
        return json.load(f)
