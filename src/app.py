from fastapi import FastAPI
from src.predict import predict

app = FastAPI()

@app.post("/predict")
def get_prediction(data: list):
    result = predict(data)
    return {"fraud": int(result[0])}