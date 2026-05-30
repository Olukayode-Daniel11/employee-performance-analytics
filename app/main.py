
from fastapi import FastAPI

from predict import predict_performance

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Employee Performance Prediction API is running"
    }

@app.post("/predict")
def predict(data: dict):

    prediction = predict_performance(data)

    return {
        "prediction": prediction
    }
