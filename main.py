import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Iris Classifier API", version="1.0")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

target_names = ["setosa", "versicolor", "virginica"]

class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Prediction(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict

@app.get("/")
def root():
    return {"message": "Iris Classifier API", "version": "1.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "RandomForestClassifier"}

@app.post("/predict", response_model=Prediction)
def predict(features: Features):
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    return Prediction(
        prediction=target_names[prediction],
        confidence=round(float(probabilities[prediction]), 3),
        probabilities={
            name: round(float(prob), 3)
            for name, prob in zip(target_names, probabilities)
        }
    )
