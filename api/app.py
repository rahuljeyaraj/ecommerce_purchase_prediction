from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# -------------------------
# Create app
# -------------------------
app = FastAPI()

# -------------------------
# Load model
# -------------------------
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------
# Define input schema
# -------------------------
class InputData(BaseModel):
    cart: int
    view: int
    session_duration_sec: float
    unique_products: int
    avg_price: float


# -------------------------
# Health check (optional)
# -------------------------
@app.get("/")
def read_root():
    return {"message": "API is running"}


# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "purchase_prediction": int(prediction),
        "purchase_probability": float(probability),
    }