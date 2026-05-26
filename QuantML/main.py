from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
import pandas as pd

from preprocess import load_and_clean
from features import add_features
from predict import (
    predict_test_set,
    forecast_n_days,
)

app = FastAPI(
    title="QuantML API",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Helpers ----------

def process_csv(file_bytes: bytes):
    """
    Convert uploaded CSV -> cleaned dataframe -> feature engineered dataframe
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(file_bytes)
    tmp.close()

    df = load_and_clean(tmp.name)
    df = add_features(df)

    os.unlink(tmp.name)

    return df


# ---------- Routes ----------

@app.get("/")
def home():
    return {
        "message": "QuantML Backend Running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/upload-and-predict")
async def upload_and_predict(
    file: UploadFile = File(...),
    model_name: str = Form("xgboost")
):

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed"
        )

    contents = await file.read()

    try:
        df = process_csv(contents)

        result = predict_test_set(df, model_name)

        return {
            "success": True,
            "model": model_name,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/forecast")
async def forecast(
    file: UploadFile = File(...),
    model_name: str = Form("xgboost"),
    n_days: int = Form(30)
):

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed"
        )

    contents = await file.read()

    try:
        df = process_csv(contents)

        result = forecast_n_days(
            df=df,
            model_name=model_name,
            n_days=n_days
        )

        return {
            "success": True,
            "model": model_name,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/metrics")
def get_metrics():

    metrics_path = "models/metrics.json"

    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="metrics.json not found"
        )

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return metrics