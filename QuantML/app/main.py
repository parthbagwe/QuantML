"""
main.py
=======
FastAPI REST backend.

Endpoints:
  POST /upload-and-predict  — upload CSV, pick model, get test-set predictions
  POST /forecast            — upload CSV, get N-day forecast
  POST /compare-all         — run all 4 models, compare results
  GET  /metrics             — load saved metrics.json
  GET  /health              — health check

Run locally:
  uvicorn app.main:app --reload --port 8000
  Visit: http://localhost:8000/docs
"""

import io, json, os, tempfile
import pandas as pd

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.preprocess import load_and_clean
from src.features import add_features
from src.predict import predict_test_set, forecast_n_days, all_models_predict_test


app = FastAPI(
    title="QuantML Stock Predictor API",
    description="ML-powered stock price prediction using OHLCV data",
    version="1.0.0",
)

# Allow frontend (different port/domain) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # replace * with your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


def _process_upload(file_bytes: bytes) -> pd.DataFrame:
    """Clean and feature-engineer an uploaded CSV."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.write(file_bytes)
    tmp.close()
    df = load_and_clean(tmp.name)
    os.unlink(tmp.name)
    return add_features(df)


@app.get("/", response_class=HTMLResponse)
def root():
    if os.path.exists("templates/index.html"):
        return FileResponse("templates/index.html")
    return HTMLResponse(
        "<h2>QuantML API running. Visit <a href='/docs'>/docs</a></h2>"
    )


@app.get("/health")
def health():
    return {"status": "ok", "models_exist": os.path.exists("models")}


@app.post("/upload-and-predict")
async def upload_and_predict(
    file:       UploadFile = File(...),
    model_name: str        = Form(default="xgboost"),
):
    """Upload Kaggle CSV and get actual vs predicted prices for the test set."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")
    contents = await file.read()
    try:
        df = _process_upload(contents)
    except Exception as e:
        raise HTTPException(422, f"Failed to process CSV: {e}")
    try:
        result = predict_test_set(df, model_name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {
        "model_name": model_name,
        **result,
        "dataset_info": {
            "rows_total": len(df),
            "date_start": str(df["date"].min().date()),
            "date_end":   str(df["date"].max().date()),
        }
    }


@app.post("/forecast")
async def forecast(
    file:       UploadFile = File(...),
    model_name: str        = Form(default="xgboost"),
    n_days:     int        = Form(default=30),
):
    """Upload CSV and receive an N-day future price forecast."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")
    if not 1 <= n_days <= 180:
        raise HTTPException(422, "n_days must be 1–180.")
    contents = await file.read()
    try:
        df = _process_upload(contents)
    except Exception as e:
        raise HTTPException(422, f"Failed to process CSV: {e}")
    try:
        result = forecast_n_days(df, model_name, n_days)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"model_name": model_name, **result}


@app.post("/compare-all")
async def compare_all(file: UploadFile = File(...)):
    """Run all 4 models on the uploaded dataset."""
    contents = await file.read()
    try:
        df = _process_upload(contents)
    except Exception as e:
        raise HTTPException(422, f"Failed to process CSV: {e}")
    return all_models_predict_test(df)


@app.get("/metrics")
def get_saved_metrics():
    """Return saved metrics.json from last training run."""
    path = "models/metrics.json"
    if not os.path.exists(path):
        raise HTTPException(
            404, "No metrics found. Run python run_training.py first."
        )
    with open(path) as f:
        return json.load(f)
