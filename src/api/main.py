from fastapi import FastAPI
from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse
)

import joblib
import pandas as pd


app = FastAPI(
    title="Credit Risk API"
)


model = joblib.load(
    "models/best_model.pkl"
)


@app.get("/")
def home():

    return {
        "message": "Credit Risk API Running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: PredictionRequest):

    df = pd.DataFrame(
        [data.dict()]
    )

    probability = (
        model.predict_proba(df)[0][1]
    )

    return PredictionResponse(
        risk_probability=float(
            probability
        )
    )