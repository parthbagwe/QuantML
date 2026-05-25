from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.predict import predict_next_n_days
import pandas as pd, io

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.post("/upload-and-predict")
async def predict(file: UploadFile = File(...), model: str = "xgboost", days: int = 30):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode()))
    result = predict_next_n_days(df, model_name=model, n_days=days)
    return result

@app.get("/metrics")
def get_metrics():
    import json
    return json.load(open("models/metrics.json"))