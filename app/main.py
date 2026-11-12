
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Estimates customer churn probability.",
    version="1.0.0",
)

model = None
threshold = None

if MODEL_PATH.exists() and METADATA_PATH.exists():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, encoding="utf-8") as file:
        threshold = float(json.load(file)["threshold"])


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(customer: Customer):
    if model is None or threshold is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Train the model with `python -m src.train` first.",
        )

    data = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(data)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "churn_probability": probability,
        "prediction": prediction,
        "result": "Likely to churn" if prediction else "Likely to stay",
        "threshold": threshold,
    }
