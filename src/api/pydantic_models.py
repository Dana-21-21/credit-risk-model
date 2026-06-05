from pydantic import BaseModel


class PredictionRequest(BaseModel):

    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float


class PredictionResponse(BaseModel):

    risk_probability: float