from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_FILE = Path(__file__).with_name("model.pkl")


class LoanRequest(BaseModel):
    gender: str = Field(..., example="Male")
    married: str = Field(..., example="yes")
    education: str = Field(..., example="Graduate")
    self_employed: str = Field(..., example="No")
    applicant_income: float = Field(..., example=5000)
    coapplicant_income: float = Field(..., example=1500)
    loan_amount: float = Field(..., example=100)
    credit_history: int = Field(..., ge=0, le=1, example=1)


class LoanResponse(BaseModel):
    prediction: str
    probability: float
    model: str = Field("loan-approval-logistic-regression")


def load_model():
    package = joblib.load(MODEL_FILE)
    return package["pipeline"], package["target_encoder"], package["feature_columns"]


app = FastAPI(
    title="Loan Approval API",
    description="Predict loan approval status and probability using a trained logistic regression model.",
    version="1.0.0",
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # পরে চাইলে তোমার Vercel domain দিতে পারো
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline, target_encoder, feature_columns = load_model()


@app.get("/")
def root():
    return {"status": "ok", "service": "loan-approval-api"}


@app.post("/predict", response_model=LoanResponse)
def predict(payload: LoanRequest):
    request_data = {
        "Gender": [payload.gender],
        "Married": [payload.married],
        "Education": [payload.education],
        "Self_Employed": [payload.self_employed],
        "ApplicantIncome": [payload.applicant_income],
        "CoapplicantIncome": [payload.coapplicant_income],
        "LoanAmount": [payload.loan_amount],
        "Credit_History": [payload.credit_history],
    }

    input_df = pd.DataFrame(request_data, columns=feature_columns)
    proba = pipeline.predict_proba(input_df)[0, 1]
    class_idx = pipeline.predict(input_df)[0]
    prediction_label = target_encoder.inverse_transform([class_idx])[0]

    return LoanResponse(
        prediction=prediction_label,
        probability=round(float(proba), 4),
    )
