"""
run_training.py
===============
One-command full training pipeline.

Usage:
  python run_training.py                          # downloads AAPL via yfinance
  python run_training.py --csv data/raw/AAPL.csv  # use local Kaggle CSV
  python run_training.py --symbol TSLA --years 5  # different stock
"""

import argparse, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--csv",    type=str, default=None)
parser.add_argument("--symbol", type=str, default="AAPL")
parser.add_argument("--years",  type=int, default=5)
args = parser.parse_args()


def main():
    print("=" * 60)
    print("  QuantML — Stock Predictor — Training Pipeline")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("static/charts", exist_ok=True)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    from src.preprocess import load_and_clean

    if args.csv:
        print(f"\n[1/5] Loading: {args.csv}")
        df_raw = load_and_clean(args.csv)
    else:
        print(f"\n[1/5] Downloading {args.symbol} ({args.years} years)...")
        try:
            import yfinance as yf
        except ImportError:
            print("ERROR: pip install yfinance")
            sys.exit(1)

        from datetime import datetime, timedelta
        end   = datetime.today()
        start = end - timedelta(days=args.years * 365)
        raw   = yf.download(args.symbol,
                            start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"),
                            auto_adjust=True, progress=False)
        if raw.empty:
            print(f"ERROR: No data for {args.symbol}")
            sys.exit(1)

        raw.reset_index(inplace=True)
        raw.columns = [str(c).lower() for c in raw.columns]
        csv_path = f"data/raw/{args.symbol}.csv"
        raw.to_csv(csv_path, index=False)
        print(f"   Saved → {csv_path}")
        df_raw = load_and_clean(csv_path)

    print(f"   Rows: {len(df_raw)} | "
          f"{df_raw['date'].min().date()} → {df_raw['date'].max().date()}")

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    print("\n[2/5] Engineering features...")
    from src.features import add_features
    df = add_features(df_raw)
    print(f"   {df.shape[1]} columns, {len(df)} usable rows")
    df.to_csv(f"data/processed/{args.symbol}_featured.csv", index=False)

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print("\n[3/5] Training models...")
    from src.train import train
    models, X_test, y_test, feature_cols = train(df)

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    print("\n[4/5] Evaluating...")
    from src.evaluate import evaluate_all
    metrics = evaluate_all(models, X_test, y_test)

    print("\n── Results ──")
    for name, m in metrics.items():
        print(f"   {name:25s}  R²={m['R2']:.4f}  "
              f"RMSE={m['RMSE']:.4f}  Dir={m['Directional_Accuracy']:.1f}%")

    # ── Step 5: Generate charts ───────────────────────────────────────────────
    print("\n[5/5] Generating charts → static/charts/")
    from src.evaluate import (
        plot_actual_vs_predicted, plot_model_comparison_bar,
        plot_candlestick, plot_moving_averages,
        plot_correlation_heatmap, plot_feature_importance,
    )
    import numpy as np

    feat_present = [f for f in feature_cols if f in df.columns]
    X_all  = df[feat_present].values
    split  = int(len(X_all) * 0.8)
    X_t    = X_all[split:]
    y_t    = df["close"].values[split:]
    dates_t = df["date"].iloc[split:].reset_index(drop=True)

    all_preds = {n: m.predict(X_t) for n, m in models.items()}

    plot_actual_vs_predicted(y_t, all_preds, dates_t,
                              save_path="static/charts/actual_vs_predicted.png")
    plot_model_comparison_bar(metrics,
                               save_path="static/charts/model_comparison.png")
    plot_candlestick(df,          save_path="static/charts/candlestick.png")
    plot_moving_averages(df,      save_path="static/charts/moving_averages.png")
    plot_correlation_heatmap(df,  save_path="static/charts/correlation_heatmap.png")

    best = max(metrics, key=lambda k: metrics[k]["R2"])
    plot_feature_importance(models[best], feat_present, model_name=best,
                             save_path="static/charts/feature_importance.png")

    print("\n" + "=" * 60)
    print("  Done. Launch the app:")
    print("  streamlit run streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
