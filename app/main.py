
from fastapi import FastAPI
from pydantic import BaseModel

from app.predict import predict_performance


class EmployeeInput(BaseModel):
    EmpDepartment: str
    EmpJobRole: str
    EmpEnvironmentSatisfaction: int
    EmpLastSalaryHikePercent: int
    YearsSinceLastPromotion: int
    ExperienceYearsInCurrentRole: int
    EmpWorkLifeBalance: int
    YearsWithCurrManager: int
    ExperienceYearsAtThisCompany: int
    EmpJobLevel: int


app = FastAPI(
    title="Employee Performance Prediction API",
    description="API for predicting employee performance using a trained machine learning model.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Employee Performance Prediction API is running"
    }


@app.post("/predict")
def predict(data: EmployeeInput):

    prediction = predict_performance(data.model_dump())

    return {
        "prediction": prediction
    }

